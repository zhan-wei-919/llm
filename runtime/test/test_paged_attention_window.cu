#include "../../model/modules/PagedAttention.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int NH = 32, NKV = 4, HS = 64;
constexpr int QS = NH * HS, KS = NKV * HS;
constexpr int TOTAL = 66, MAX_SEQS = 3, MAX_SEQ_LEN = 64;

// 检查每个 query 头和维度都得到对应窗口内 V 的均值。
static void check_output(Tensor *tensor, const std::vector<float> &expected) {
	std::vector<__nv_bfloat16> out((size_t)TOTAL * QS);
	cudaMemcpy(out.data(), tensor->ptr, out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
	for (int t = 0; t < TOTAL; ++t)
		for (int i = 0; i < QS; ++i)
			assert(std::fabs(__bfloat162float(out[(size_t)t * QS + i]) - expected[t]) < 2e-2f);
}

static float value_at(int batch, int pos) {
	return batch * 2.0f + (pos + 1) / 32.0f;
}

// 把本轮各序列的新 V 写入真实 KV Pool 物理位置。
static void write_values(KV_Pool &pool, const int *slots, const int *starts, const int *lens, std::vector<__nv_bfloat16> &v) {
	for (int b = 0; b < MAX_SEQS; ++b)
		for (int p = starts[b]; p < starts[b] + lens[b]; ++p) {
			int dst = pool.physical_token(slots[b], p) * KS;
			for (int i = 0; i < KS; ++i) v[dst + i] = __float2bfloat16_rn(value_at(b, p));
		}
}

// 线性 V 在 causal 窗口内的均值可直接由首尾项得到。
static std::vector<float> expected_values(const int *starts, const int *lens, int window_size) {
	std::vector<float> expected;
	expected.reserve(TOTAL);
	for (int b = 0; b < MAX_SEQS; ++b)
		for (int t = 0; t < lens[b]; ++t) {
			int pos = starts[b] + t;
			int begin = window_size < 0 ? 0 : std::max(0, pos + 1 - window_size);
			expected.push_back((value_at(b, begin) + value_at(b, pos)) * 0.5f);
		}
	return expected;
}

int main() {
	using T = __nv_bfloat16;
	size_t kv_bytes = (size_t)MAX_SEQS * MAX_SEQ_LEN * KS * sizeof(T);
	void *k_base, *v_base;
	cudaMalloc(&k_base, kv_bytes);
	cudaMalloc(&v_base, kv_bytes);
	LLM llm(6);
	Engine engine(llm.arena(), NH, NKV, HS, MAX_SEQS, MAX_SEQ_LEN, TOTAL);
	Tensor *q = llm.arena().alloc({TOTAL, QS}, Dtype::BF16);
	PagedAttention<T> full(llm, engine, q, 0, -1);
	PagedAttention<T> sliding(llm, engine, q, 0, 3);
	llm.finalize();
	KV_Pool pool(k_base, v_base, Dtype::BF16, KS, kv_bytes, 1, MAX_SEQS, MAX_SEQ_LEN);
	engine.bind_pool(&pool);
	cudaMemset(q->ptr, 0, q->bytes());
	cudaMemset(k_base, 0, kv_bytes);
	std::vector<T> v(kv_bytes / sizeof(T), T(0.0f));
	int slots[] = {engine.alloc_seq(), engine.alloc_seq(), engine.alloc_seq()};
	int starts0[] = {0,0,0}, lens0[] = {1,32,33};

	GraphShape shape = engine.prepare(slots, MAX_SEQS, lens0, llm.stream());
	write_values(pool, slots, starts0, lens0, v);
	cudaMemcpyAsync(v_base, v.data(), kv_bytes, cudaMemcpyHostToDevice, llm.stream());
	llm.forward(ExecutionPhase::PREFILL, shape);
	cudaStreamSynchronize(llm.stream());
	check_output(full.out(), expected_values(starts0, lens0, -1));
	check_output(sliding.out(), expected_values(starts0, lens0, 3));

	int starts1[] = {1,32,33}, lens1[] = {31,17,18};
	GraphShape replay = engine.prepare(slots, MAX_SEQS, lens1, llm.stream());
	write_values(pool, slots, starts1, lens1, v);
	cudaMemcpyAsync(v_base, v.data(), kv_bytes, cudaMemcpyHostToDevice, llm.stream());
	llm.forward(ExecutionPhase::PREFILL, replay);
	cudaStreamSynchronize(llm.stream());
	check_output(full.out(), expected_values(starts1, lens1, -1));
	check_output(sliding.out(), expected_values(starts1, lens1, 3));
	assert(shape == replay && llm.num_graphs(ExecutionPhase::PREFILL) == 1);

	cudaFree(k_base); cudaFree(v_base);
	std::printf("test_paged_attention_window PASS\n");
}
