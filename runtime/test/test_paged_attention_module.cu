// PagedAttention module + CUDA Graph 测试:
// 第一步 bake {B=1,total=3}，第二步用相同 shape replay，但更新输入和 KV 元数据。
//
// 编译: nvcc -O2 -arch=sm_120 test_paged_attention_module.cu -o test_paged_attention_module
#include "../../model/modules/PagedAttention.h"
#include "../GraphShape.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

constexpr int NH = 1, NKV = 1, HS = 4;
constexpr int QS = NH * HS, KS = NKV * HS;
constexpr int TOKENS = 3;
constexpr int MAX_SEQS = 1, MAX_SEQ_LEN = KV_BLOCK_SIZE;

static void check_cuda(cudaError_t err) {
	if (err != cudaSuccess) {
		std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
		std::abort();
	}
}

static void check_output(const std::vector<float> &out,
	                     const std::vector<float> &expected) {
	assert(out.size() == expected.size() * QS);
	for (size_t t = 0; t < expected.size(); ++t)
		for (int d = 0; d < QS; ++d)
			assert(std::fabs(out[t * QS + d] - expected[t]) < 1e-6f);
}

int main() {
	cudaStream_t prepare_stream;
	check_cuda(cudaStreamCreate(&prepare_stream));

	const size_t kv_bytes = (size_t)KV_BLOCK_SIZE * KS * sizeof(float);
	void *k_base, *v_base;
	check_cuda(cudaMalloc(&k_base, kv_bytes));
	check_cuda(cudaMalloc(&v_base, kv_bytes));
	LLM llm(/*max_tensors=*/7); // Engine meta/cos/sin + q/k/v/out
	Engine engine(llm.arena(), NH, NKV, HS, MAX_SEQS, MAX_SEQ_LEN);
	Tensor *q = llm.arena().alloc({TOKENS, QS}, Dtype::F32);
	Tensor *k = llm.arena().alloc({TOKENS, KS}, Dtype::F32);
	Tensor *v = llm.arena().alloc({TOKENS, KS}, Dtype::F32);
	PagedAttention<float> attention(llm, engine, q, k, v, /*layer=*/0);
	llm.finalize();
	KV_Pool pool(k_base, v_base, Dtype::F32, KS, kv_bytes,
	             /*num_layers=*/1, MAX_SEQS, MAX_SEQ_LEN);
	engine.bind_pool(&pool);

	std::vector<float> h_q(TOKENS * QS, 0.0f);
	std::vector<float> h_k(TOKENS * KS, 1.0f);
	std::vector<float> h_v(TOKENS * KS, 2.0f);
	std::vector<float> h_out(TOKENS * QS);

	// 第一步: Q 全 0，所以 attention 是前缀 V 的均值，结果全为 2。
	check_cuda(cudaMemcpy(q->ptr, h_q.data(), h_q.size() * sizeof(float), cudaMemcpyHostToDevice));
	check_cuda(cudaMemcpy(k->ptr, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice));
	check_cuda(cudaMemcpy(v->ptr, h_v.data(), h_v.size() * sizeof(float), cudaMemcpyHostToDevice));
	int slot = engine.alloc_seq();
	int slots[] = {slot}, lens[] = {TOKENS};
	GraphShape shape = engine.prepare(slots, 1, lens, prepare_stream);
	check_cuda(cudaStreamSynchronize(prepare_stream));
	llm.forward(shape);                         // 首次见到 shape，bake + launch
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaMemcpy(h_out.data(), attention.out()->ptr,
	                      h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
	check_output(h_out, {2.0f, 2.0f, 2.0f});

	// 第二步: shape 仍是 {1,3}，但新 V 为 7。应 replay 旧图并读取更新后的元数据。
	std::fill(h_v.begin(), h_v.end(), 7.0f);
	check_cuda(cudaMemcpy(v->ptr, h_v.data(), h_v.size() * sizeof(float), cudaMemcpyHostToDevice));
	GraphShape replay_shape = engine.prepare(slots, 1, lens, prepare_stream);
	assert(replay_shape == shape);
	check_cuda(cudaStreamSynchronize(prepare_stream));
	llm.forward(replay_shape);                  // 同 shape，直接 replay
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaMemcpy(h_out.data(), attention.out()->ptr,
	                      h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
	check_output(h_out, {3.25f, 4.0f, 4.5f});
	assert(pool.seq_len(slot) == 2 * TOKENS);

	check_cuda(cudaFree(k_base));
	check_cuda(cudaFree(v_base));
	check_cuda(cudaStreamDestroy(prepare_stream));
	std::printf("test_paged_attention_module PASS\n");
	return 0;
}
