// 五层 decoder-only 小模型端到端测试：
// Scheduler -> PipelineDriver -> Embedding -> 5 x Transformer -> Final RMSNorm -> LM Head -> argmax。
// 覆盖在线 continuous batching、chunked prefill、纯 P/D batch 与两套 GraphShape 图族。
//
// 编译: nvcc -O2 -arch=sm_120 test_five_layer_model.cu -o test_five_layer_model
#include "../../model/modules/Embedding.h"
#include "../../model/modules/Transformer.h"
#include "../../model/modules/RMSNorm.h"
#include "../../model/modules/LMHead.h"
#include "../Engine.h"
#include "../Scheduler.h"
#include "../Driver.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

constexpr int NUM_LAYERS = 5;
constexpr int NH = 4, NKV = 2, HS = 32;
constexpr int HIDDEN = NH * HS;
constexpr int FC_DIM = 256;
constexpr int VOCAB = 128;
constexpr int MAX_TOKENS = 8;
constexpr int MAX_SEQS = 4;
constexpr int MAX_SEQ_LEN = 32;
constexpr int BLOCKS = 8;
constexpr int NREQ = 3;
const int PROMPT_LEN[NREQ] = {12, 5, 9};
const int MAX_NEW[NREQ] = {3, 4, 2};

static void check_cuda(cudaError_t err) {
	if (err != cudaSuccess) {
		std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
		std::abort();
	}
}

static void randomize_parameters(LLM &llm) {
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
	for (Tensor *tensor : llm.parameter_storages()) {
		assert(tensor->dtype == Dtype::BF16);
		std::vector<__nv_bfloat16> values(tensor->numel());
		for (__nv_bfloat16 &value : values) value = __float2bfloat16(dist(rng));
		check_cuda(cudaMemcpy(tensor->ptr, values.data(), tensor->bytes(), cudaMemcpyHostToDevice));
	}
}

// 每个 batch 条目只对本步最后一行 logits 做 argmax。
__global__ void sample_argmax(int *tokens, const __nv_bfloat16 *logits, const int *rows) {
	int b = blockIdx.x;
	const __nv_bfloat16 *row = logits + (size_t)rows[b] * VOCAB;
	int best = 0;
	for (int v = 1; v < VOCAB; ++v)
		if (__bfloat162float(row[v]) > __bfloat162float(row[best])) best = v;
	tokens[b] = best;
}

int main() {
	const size_t layer_bytes = (size_t)BLOCKS * KV_BLOCK_SIZE * NKV * HS * sizeof(__nv_bfloat16);
	const size_t kv_bytes = NUM_LAYERS * layer_bytes;
	void *k_base, *v_base;
	check_cuda(cudaMalloc(&k_base, kv_bytes));
	check_cuda(cudaMalloc(&v_base, kv_bytes));
	{
		LLM llm(/*max_tensors=*/128);
		cudaStream_t stream = llm.stream();
		Engine engine(llm.arena(), NH, NKV, HS, MAX_SEQS, MAX_SEQ_LEN);
		Tensor *token_ids = llm.arena().alloc({MAX_TOKENS}, Dtype::I32);
		Embedding<__nv_bfloat16> embedding(llm, token_ids, MAX_TOKENS, VOCAB, HIDDEN,
		                           "model.embed_tokens");

		std::vector<std::unique_ptr<Transformer<__nv_bfloat16>>> layers;
		layers.reserve(NUM_LAYERS);
		Tensor *x = embedding.out();
		for (int layer = 0; layer < NUM_LAYERS; ++layer) {
			layers.emplace_back(std::make_unique<Transformer<__nv_bfloat16>>(
				llm, engine, x, MAX_TOKENS, HIDDEN, layer, FC_DIM,
				/*qkv_bias=*/false, /*o_bias=*/false, /*fc_bias=*/false,
				1e-5f, "model.layers." + std::to_string(layer)));
			x = layers.back()->out();
		}
		RMSNorm<__nv_bfloat16> final_norm(llm, x, MAX_TOKENS, HIDDEN, 1e-5f, "model.norm");
		LMHead<__nv_bfloat16> lm_head(llm, final_norm.out(), MAX_TOKENS, HIDDEN, VOCAB,
		                          "lm_head");

		llm.finalize();
		KV_Pool pool(k_base, v_base, Dtype::BF16, NKV * HS, kv_bytes,
		             NUM_LAYERS, MAX_SEQS, MAX_SEQ_LEN);
		engine.bind_pool(&pool);
		engine.init_rope(stream);
		check_cuda(cudaStreamSynchronize(stream));
		randomize_parameters(llm);
		assert(llm.parameters().size() == 48);

		LocalKvHandoff handoff;
		Scheduler scheduler(pool, {MAX_SEQS, MAX_TOKENS, /*eos=*/-1}, handoff);
		auto submit_request = [&](int r) {
			std::vector<int> prompt(PROMPT_LEN[r]);
			for (int i = 0; i < PROMPT_LEN[r]; ++i) prompt[i] = (r * 31 + i * 7 + 3) % VOCAB;
			assert(scheduler.add_request(std::move(prompt), MAX_NEW[r]) == r);
		};
		// 只预先提交 r0；r1/r2 会在 GPU 流水已经启动后动态加入。
		submit_request(0);

		int *d_sample, *h_sample, *d_rows, *h_rows;
		check_cuda(cudaMalloc(&d_sample, 2 * MAX_SEQS * sizeof(int)));
		check_cuda(cudaHostAlloc(&h_sample, 2 * MAX_SEQS * sizeof(int), 0));
		check_cuda(cudaMalloc(&d_rows, 2 * MAX_SEQS * sizeof(int)));
		check_cuda(cudaHostAlloc(&h_rows, 2 * MAX_SEQS * sizeof(int), 0));

		bool saw_prefill = false, saw_decode = false, saw_multi_batch = false;
		bool saw_r0_decode = false, saw_late_prefill = false;
		bool submitted_late = false;
		int launched_steps = 0;
		std::map<std::tuple<int, int, int>, int> shape_counts;
		std::vector<std::vector<int>> generated(NREQ);

		auto launch = [&](const ScheduledBatch &batch, int parity) {
			const StepPlan &plan = batch.plan;
			int B = (int)plan.req_ids.size();
			int total = 0;
			for (int b = 0; b < B; ++b) {
				if (batch.phase == ExecutionPhase::DECODE) assert(plan.lens[b] == 1);
				saw_r0_decode |= batch.phase == ExecutionPhase::DECODE && plan.req_ids[b] == 0;
				saw_late_prefill |= batch.phase == ExecutionPhase::PREFILL && plan.req_ids[b] >= 1;
				total += plan.lens[b];
				h_rows[parity * MAX_SEQS + b] = total - 1;
			}
			assert(total == (int)plan.ids.size());
			saw_prefill |= batch.phase == ExecutionPhase::PREFILL;
			saw_decode |= batch.phase == ExecutionPhase::DECODE;
			saw_multi_batch |= B > 1;
			shape_counts[{batch.phase == ExecutionPhase::PREFILL ? 0 : 1, B, total}]++;
			++launched_steps;

			check_cuda(cudaMemcpyAsync(token_ids->ptr, plan.ids.data(), total * sizeof(int),
			                               cudaMemcpyHostToDevice, stream));
			check_cuda(cudaMemcpyAsync(d_rows + parity * MAX_SEQS,
			                               h_rows + parity * MAX_SEQS, B * sizeof(int),
			                               cudaMemcpyHostToDevice, stream));
			GraphShape shape = engine.prepare(plan.slots.data(), B, plan.lens.data(), stream);
			assert(shape.batch == B && shape.total_tokens == total);
			llm.forward(batch.phase, shape);
			sample_argmax<<<B, 1, 0, stream>>>(
				d_sample + parity * MAX_SEQS,
				static_cast<const __nv_bfloat16 *>(lm_head.out()->ptr),
				d_rows + parity * MAX_SEQS);
			check_cuda(cudaMemcpyAsync(h_sample + parity * MAX_SEQS,
			                               d_sample + parity * MAX_SEQS, B * sizeof(int),
			                               cudaMemcpyDeviceToHost, stream));
		};

		auto on_finished = [&](std::vector<Request> &&done) {
			for (auto &request : done)
				generated[request.id] = std::vector<int>(
					request.token_ids.begin() + request.prompt_len,
					request.token_ids.end());
		};

		{
			PipelineDriver driver(scheduler, h_sample, MAX_SEQS, stream, launch, on_finished);
			while (true) {
				PipelineDriver::Pump state = driver.pump();
				if (state == PipelineDriver::Pump::IDLE) break;
				if (state != PipelineDriver::Pump::LAUNCHED) continue;

				if (!submitted_late && launched_steps >= 2) {
					submit_request(1);
					submit_request(2);
					submitted_late = true;
				}
			}
		}

		assert(saw_prefill && saw_decode && saw_multi_batch);
		assert(saw_r0_decode && saw_late_prefill);
		assert(shape_counts.size() >= 3);
		bool replayed_shape = false;
		for (const auto &entry : shape_counts) replayed_shape |= entry.second > 1;
		assert(replayed_shape);
		assert(llm.num_graphs(ExecutionPhase::PREFILL) > 0);
		assert(llm.num_graphs(ExecutionPhase::DECODE) > 0);
		for (int r = 0; r < NREQ; ++r) {
			assert((int)generated[r].size() == MAX_NEW[r]);
			for (int token : generated[r]) assert(token >= 0 && token < VOCAB);
		}
		assert(scheduler.num_requests() == 0);
		assert(scheduler.num_preemptions() == 0);
		assert((int)pool.num_free_blocks() == BLOCKS);
		assert((int)pool.num_free_slots() == MAX_SEQS);

		check_cuda(cudaFreeHost(h_rows));
		check_cuda(cudaFree(d_rows));
		check_cuda(cudaFreeHost(h_sample));
		check_cuda(cudaFree(d_sample));
	}

	check_cuda(cudaFree(k_base));
	check_cuda(cudaFree(v_base));
	std::printf("test_five_layer_model PASS: online continuous batching + split prefill/decode graph replay\n");
	return 0;
}
