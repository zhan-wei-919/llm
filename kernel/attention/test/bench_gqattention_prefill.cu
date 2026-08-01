#include "../GQAttention_prefill.h"
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>

__global__ void init_bf16(__nv_bfloat16 *x, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		unsigned v = unsigned(i) * 747796405u + 2891336453u;
		v = ((v >> ((v >> 28) + 4)) ^ v) * 277803737u;
		v = (v >> 22) ^ v;
		x[i] = __float2bfloat16_rn(float(int(v & 0xffff) - 32768) / 65536.0f);
	}
}

template<typename F>
float time_us(F fn, int warmup, int iters) {
	for (int i = 0; i < warmup; ++i) fn();
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	cudaEventRecord(begin);
	for (int i = 0; i < iters; ++i) fn();
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float ms;
	cudaEventElapsedTime(&ms, begin, end);
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
	return ms * 1000.0f / iters;
}

int main() {
	using T = __nv_bfloat16;
	constexpr int B = 1, NH = 32, NKV = 4, HS = 64;
	for (int total : {128, 256, 512}) {
		int blocks = total / KV_BLOCK_SIZE;
		size_t q_elems = (size_t)total * NH * HS, kv_elems = (size_t)total * NKV * HS;
		T *q, *k, *v, *out;
		int *cu_seqlens, *seq_ids, *pos_offset, *block_table;
		cudaMalloc(&q, q_elems * sizeof(T));
		cudaMalloc(&k, kv_elems * sizeof(T));
		cudaMalloc(&v, kv_elems * sizeof(T));
		cudaMalloc(&out, q_elems * sizeof(T));
		cudaMalloc(&cu_seqlens, 2 * sizeof(int));
		cudaMalloc(&seq_ids, total * sizeof(int));
		cudaMalloc(&pos_offset, sizeof(int));
		cudaMalloc(&block_table, blocks * sizeof(int));
		init_bf16<<<(q_elems + 255) / 256, 256>>>(q, q_elems);
		init_bf16<<<(kv_elems + 255) / 256, 256>>>(k, kv_elems);
		init_bf16<<<(kv_elems + 255) / 256, 256>>>(v, kv_elems);
		int h_cu[2] = {0, total}, zero = 0;
		std::vector<int> h_seq(total, 0), h_table(blocks);
		for (int i = 0; i < blocks; ++i) h_table[i] = i;
		cudaMemcpy(cu_seqlens, h_cu, 2 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(seq_ids, h_seq.data(), total * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(pos_offset, &zero, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(block_table, h_table.data(), blocks * sizeof(int), cudaMemcpyHostToDevice);
		auto run = [&] {launch_gq_attention_prefill(out, q, k, v, cu_seqlens, seq_ids, pos_offset, block_table, B, NH, NKV, HS, blocks, total, -1, 0);};
		float us = time_us(run, 20, 100);
		printf("T=%d custom=%.3f us\n", total, us);
		cudaFree(q);
		cudaFree(k);
		cudaFree(v);
		cudaFree(out);
		cudaFree(cu_seqlens);
		cudaFree(seq_ids);
		cudaFree(pos_offset);
		cudaFree(block_table);
	}
}
