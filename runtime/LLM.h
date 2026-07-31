#pragma once
#include <cuda_runtime.h>
#include <array>
#include <vector>
#include <unordered_map>
#include <initializer_list>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include "../tensor/Arena.h"
#include "../utils/safetensor_loader.h"
#include "../kernel/sampling/Argmax.h"
#include "Engine.h"
#include "../kv/KV_pool.h"
#include "../tensor/Tensor.h"
#include "Scheduler.h"
#include "Driver.h"
#include "../tokenizer/llama_tokenizer.h"
#include "OpRecord.h"

inline constexpr uint64_t SAFETENSOR_LOAD_BUFFER_BYTES = 1ull << 30;

class LLM {
	friend class Module;
public:
	// wait=false: 非阻塞取当前输入; wait=true: server idle 时阻塞等待。
	// idle 等待返回 nullopt 时，server 退出。
	using ReceiveFn = std::function<std::optional<std::string>(bool wait)>;
	// 请求完成后立即回调，只返回新生成的文本。
	using EmitFn = std::function<void(int request_id, std::string generated_text)>;

	struct InferenceBuffers {
		int *h_inputs;
		int *d_tokens;
		int *h_tokens;
		int input_stride;
		int token_stride;
	};

	explicit LLM(int max_tensors)
	: arena_(max_tensors){
		cudaStreamCreate(&stream_);
	}

	~LLM(){
		cudaStreamSynchronize(stream_);
		for (auto &cache : graphs_)
			for (auto &[shape, exec] : cache) cudaGraphExecDestroy(exec);
		cudaStreamDestroy(stream_);
	}

	cudaGraphExec_t bake(ExecutionPhase, const GraphShape &shape) {
		cudaGraph_t graph;
		cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
		for (const auto &op : program_) op.forward(shape, stream_);
		cudaStreamEndCapture(stream_, &graph);
		cudaGraphExec_t exec;
		cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
		cudaGraphDestroy(graph);
		return exec;
	}

	Arena &arena() {return arena_;}
	Tensor *parameter(std::string name, std::initializer_list<int> shape, Dtype dtype) {
		Tensor *tensor = arena_.alloc(shape, dtype);
		parameters_.emplace(std::move(name), tensor);
		return tensor;
	}
	const std::unordered_map<std::string, Tensor *> &parameters() const {return parameters_;}
	cudaStream_t stream() const {return stream_;}

	// std::vector<std::string> need_transpose = {".q_proj.weight", ".k_proj.weight", ... ,"lm_head.weight"};
	// llm.load_safetensor(path, [&](const std::string &name) {
	// 	for (const std::string &pattern : need_transpose){
	// 		if (name.find(pattern) != std::string::npos) return 1;
	// 	}
	// 	return 0;
	// }
	template<typename F>
	void load_safetensor(const std::string &path, F transpose) {
		struct LoadItem {
			std::string name;
			Tensor *tensor;
			layer_info info;
		};

		auto infos = weight_loader(path);
		std::vector<LoadItem> items;
		items.reserve(parameters_.size());
		for (const auto &parameter : parameters_)
			items.push_back({parameter.first, parameter.second, infos.at(parameter.first)});
		std::sort(items.begin(), items.end(), [](const LoadItem &a, const LoadItem &b) {
			return a.info.start < b.info.start;
		});

		std::ifstream file(path, std::ios::binary);
		uint64_t header_size = safetensor_detail::read_header_size(file);
		uint64_t data_start = 8 + header_size;

		for (size_t i = 0; i < items.size();) {
			uint64_t batch_start = items[i].info.start;
			uint64_t batch_end = batch_start;
			size_t j = i;
			while (j < items.size() && items[j].info.end - batch_start <= SAFETENSOR_LOAD_BUFFER_BYTES) {
				batch_end = items[j].info.end;
				++j;
			}

			std::vector<char> buffer(static_cast<size_t>(batch_end - batch_start));
			file.seekg(static_cast<std::streamoff>(data_start + batch_start));
			file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));

			for (size_t k = i; k < j; ++k) {
				const LoadItem &item = items[k];
				size_t bytes = static_cast<size_t>(item.info.end - item.info.start);
				const char *source = buffer.data() + (item.info.start - batch_start);
				if (!transpose(item.name)) {
					cudaMemcpy(item.tensor->ptr, source, bytes, cudaMemcpyHostToDevice);
					continue;
				}

				size_t rows = static_cast<size_t>(item.info.shape[0]);
				size_t cols = static_cast<size_t>(item.info.shape[1]);
				size_t element_bytes = dtype_size(item.info.d);
				std::vector<char> transposed(bytes);
				for (size_t r = 0; r < rows; ++r)
					for (size_t c = 0; c < cols; ++c)
						std::memcpy(
							transposed.data() + (c * rows + r) * element_bytes,
							source + (r * cols + c) * element_bytes,
							element_bytes);
				cudaMemcpy(item.tensor->ptr, transposed.data(), bytes, cudaMemcpyHostToDevice);
			}
			i = j;
		}
	}

	void forward(ExecutionPhase phase, const GraphShape &shape) {
		auto &cache = graphs_[phase == ExecutionPhase::PREFILL ? 0 : 1];
		auto it = cache.find(shape);
		if (it == cache.end()) {
			cudaStreamSynchronize(stream_);
			auto exec = bake(phase, shape);
			it = cache.emplace(shape, exec).first;
		}
		cudaGraphLaunch(it->second, stream_);
	}

	size_t num_graphs(ExecutionPhase phase) const {return graphs_[phase == ExecutionPhase::PREFILL ? 0 : 1].size();}

	void inference(const ScheduledBatch &batch, int parity, Engine &engine, const InferenceBuffers &buffers) {
		const StepPlan &plan = batch.plan;
		int batch_size = static_cast<int>(plan.req_ids.size());
		int total_tokens = static_cast<int>(plan.ids.size());
		int *h_input = buffers.h_inputs + parity * buffers.input_stride;
		std::copy(plan.ids.begin(), plan.ids.end(), h_input);
		cudaMemcpyAsync(program_.front().input()->ptr, h_input, total_tokens * sizeof(int), cudaMemcpyHostToDevice, stream_);
		GraphShape shape = engine.prepare(plan.slots.data(), batch_size, plan.lens.data(), stream_);
		forward(batch.phase, shape);
		Tensor *logits = program_.back().output();
		int *d_tokens = buffers.d_tokens + parity * buffers.token_stride;
		int *h_tokens = buffers.h_tokens + parity * buffers.token_stride;
		launch_argmax_last_token(d_tokens, logits->ptr, logits->dtype, engine.cu_seqlens(), batch_size, logits->shape[1], stream_);
		cudaMemcpyAsync(h_tokens, d_tokens, batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream_);
	}

	void server(const ReceiveFn &receive, const EmitFn &emit, int max_new_tokens, Engine &engine, KV_Pool &pool,
	            SchedulerConfig config, const std::string &tokenizer_path) {
		InferenceBuffers buffers;
		buffers.input_stride = config.max_num_batched_tokens;
		buffers.token_stride = config.max_num_seqs;
		cudaHostAlloc(&buffers.h_inputs, 	2 * buffers.input_stride * sizeof(int), 0);
		cudaMalloc(&buffers.d_tokens, 2 * buffers.token_stride * sizeof(int));
		cudaHostAlloc(&buffers.h_tokens, 2 * buffers.token_stride * sizeof(int), 0);

		LocalKvHandoff handoff;
		Scheduler scheduler(pool, config, handoff);
		auto submit = [&](const std::string &input) {
			scheduler.add_request(tokenize(input, tokenizer_path), max_new_tokens);
		};
		auto on_finished = [&](std::vector<Request> &&done) {
			for (Request &request : done) {
				std::vector<int> generated(
					request.token_ids.begin() + request.prompt_len,
					request.token_ids.end());
				emit(request.id, detokenize(generated, tokenizer_path));
			}
		};
		auto launch = [&](const ScheduledBatch &batch, int parity) {
			inference(batch, parity, engine, buffers);
		};

		{
			PipelineDriver driver(scheduler, buffers.h_tokens, buffers.token_stride, stream_, launch, on_finished);
			while (true) {
				std::optional<std::string> input = receive(true);
				if (!input) break;
				submit(*input);
				while (true) {
					while ((input = receive(false))) submit(*input);
					if (driver.pump() == PipelineDriver::Pump::IDLE) break;
				}
			}
		}

		cudaFreeHost(buffers.h_inputs);
		cudaFree(buffers.d_tokens);
		cudaFreeHost(buffers.h_tokens);
	}

	void finalize() {arena_.finalize();}


private:
	template<typename T>
	void append(T &module, std::initializer_list<Tensor *> inputs,
	            std::initializer_list<Tensor *> outputs) {
		program_.emplace_back(module, inputs, outputs);
	}

	Arena arena_;
	std::vector<OpRecord> program_;
	std::unordered_map<std::string, Tensor *> parameters_;
	cudaStream_t stream_;
	std::array<std::unordered_map<GraphShape, cudaGraphExec_t, GraphShapeHash>, 2> graphs_;

};
