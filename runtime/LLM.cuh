#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <initializer_list>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include "../tensor/Arena.cuh"
#include "../utils/safetensor_loader.h"
#include "Engine.h"
#include "../kv/KV_pool.h"
#include "../tensor/Tensor.h"
#include "Scheduler.h"
#include "../tokenizer/llama_tokenizer.h"
#include "OpRecord.h"

inline constexpr uint64_t SAFETENSOR_LOAD_BUFFER_BYTES = 1ull << 30;

class LLM {
	friend class Module;
public:
	explicit LLM(int max_tensors)
	: arena_(max_tensors){
		cudaStreamCreate(&stream_);
	}

	~LLM(){
		cudaStreamSynchronize(stream_);
		for (auto &[shape, exec] : graphs_) cudaGraphExecDestroy(exec);
		cudaStreamDestroy(stream_);
	}

	cudaGraphExec_t bake(const GraphShape &shape) {
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

	void forward(const GraphShape &shape) {
		auto it = graphs_.find(shape);
		if (it == graphs_.end()) {
			cudaStreamSynchronize(stream_);
			auto exec = bake(shape);
			it = graphs_.emplace(shape, exec).first;
		}
		cudaGraphLaunch(it->second, stream_);
	}
	void finalize() {arena_.finalize();}


private:
	template<typename T>
	void append(T &module) {
		program_.emplace_back(module);
	}

	Arena arena_;
	std::vector<OpRecord> program_;
	std::unordered_map<std::string, Tensor *> parameters_;
	cudaStream_t stream_;
	std::unordered_map<GraphShape, cudaGraphExec_t, GraphShapeHash> graphs_;

};
