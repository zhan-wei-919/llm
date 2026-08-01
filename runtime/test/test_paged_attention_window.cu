#include "../../model/modules/PagedAttention.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int NH = 32, NKV = 4, HS = 64;
constexpr int QS = NH * HS, KS = NKV * HS;
constexpr int TOKENS = 3, MAX_SEQS = 1, MAX_SEQ_LEN = KV_BLOCK_SIZE;

// 检查每个 query 头和维度都得到对应窗口内 V 的均值。
static void check_output(Tensor *tensor, const std::vector<float> &expected) {
	std::vector<__nv_bfloat16> out((size_t)TOKENS * QS);
	cudaMemcpy(out.data(), tensor->ptr, out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
	for (int t = 0; t < TOKENS; ++t)
		for (int i = 0; i < QS; ++i)
			assert(std::fabs(__bfloat162float(out[(size_t)t * QS + i]) - expected[t]) < 2e-2f);
}

int main() {
	using T = __nv_bfloat16;
	size_t kv_bytes = (size_t)KV_BLOCK_SIZE * KS * sizeof(T);
	void *k_base, *v_base;
	cudaMalloc(&k_base, kv_bytes);
	cudaMalloc(&v_base, kv_bytes);
	LLM llm(6);
	Engine engine(llm.arena(), NH, NKV, HS, MAX_SEQS, MAX_SEQ_LEN);
	Tensor *q = llm.arena().alloc({TOKENS, QS}, Dtype::BF16);
	PagedAttention<T> full(llm, engine, q, 0, -1);
	PagedAttention<T> sliding(llm, engine, q, 0, 3);
	llm.finalize();
	KV_Pool pool(k_base, v_base, Dtype::BF16, KS, kv_bytes, 1, MAX_SEQS, MAX_SEQ_LEN);
	engine.bind_pool(&pool);
	cudaMemset(q->ptr, 0, q->bytes());
	std::vector<T> k((size_t)KV_BLOCK_SIZE * KS, T(0.0f));
	std::vector<T> v((size_t)KV_BLOCK_SIZE * KS, T(0.0f));
	int slot = engine.alloc_seq(), slots[] = {slot}, lens[] = {TOKENS};

	GraphShape shape = engine.prepare(slots, 1, lens, llm.stream());
	for (int p = 0; p < TOKENS; ++p) {
		int dst = pool.physical_token(slot, p) * KS;
		for (int i = 0; i < KS; ++i) v[dst + i] = T((float)p + 1.0f);
	}
	cudaMemcpyAsync(k_base, k.data(), kv_bytes, cudaMemcpyHostToDevice, llm.stream());
	cudaMemcpyAsync(v_base, v.data(), kv_bytes, cudaMemcpyHostToDevice, llm.stream());
	llm.forward(ExecutionPhase::PREFILL, shape);
	cudaStreamSynchronize(llm.stream());
	check_output(full.out(), {1.0f, 1.5f, 2.0f});
	check_output(sliding.out(), {1.0f, 1.5f, 2.0f});

	GraphShape replay = engine.prepare(slots, 1, lens, llm.stream());
	for (int p = TOKENS; p < 2 * TOKENS; ++p) {
		int dst = pool.physical_token(slot, p) * KS;
		for (int i = 0; i < KS; ++i) v[dst + i] = T((float)p + 1.0f);
	}
	cudaMemcpyAsync(v_base, v.data(), kv_bytes, cudaMemcpyHostToDevice, llm.stream());
	llm.forward(ExecutionPhase::PREFILL, replay);
	cudaStreamSynchronize(llm.stream());
	check_output(full.out(), {2.5f, 3.0f, 3.5f});
	check_output(sliding.out(), {3.0f, 4.0f, 5.0f});
	assert(shape == replay && llm.num_graphs(ExecutionPhase::PREFILL) == 1);

	cudaFree(k_base); cudaFree(v_base);
	std::printf("test_paged_attention_window PASS\n");
}
