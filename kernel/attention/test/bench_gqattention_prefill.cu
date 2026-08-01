#include "../Attention.h"
#include <cuda_bf16.h>
#include <algorithm>
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

// 测量 packed ragged batch，包含各序列独立的历史长度和页表行。
static void bench_case(const std::vector<int> &lens, const std::vector<int> &starts, int window_size) {
	using T = __nv_bfloat16;
	constexpr int NH = 32, NKV = 4, HS = 64;
	int B = (int)lens.size(), total = 0, blocks_per_seq = 0, max_history = 0;
	std::vector<int> h_cu(B + 1);
	for (int b = 0; b < B; ++b) {
		total += lens[b];
		h_cu[b + 1] = total;
		max_history = std::max(max_history, starts[b] + lens[b]);
		blocks_per_seq = std::max(blocks_per_seq, (starts[b] + lens[b] + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE);
	}
	int blocks = B * blocks_per_seq;
	size_t q_elems = (size_t)total * NH * HS;
	size_t kv_elems = (size_t)blocks * KV_BLOCK_SIZE * NKV * HS;
	T *q, *k, *v, *out;
	int *cu_seqlens, *pos_offset, *block_table;
	cudaMalloc(&q, q_elems * sizeof(T)); cudaMalloc(&k, kv_elems * sizeof(T));
	cudaMalloc(&v, kv_elems * sizeof(T)); cudaMalloc(&out, q_elems * sizeof(T));
	cudaMalloc(&cu_seqlens, h_cu.size() * sizeof(int));
	cudaMalloc(&pos_offset, starts.size() * sizeof(int)); cudaMalloc(&block_table, blocks * sizeof(int));
	init_bf16<<<(q_elems + 255) / 256, 256>>>(q, q_elems);
	init_bf16<<<(kv_elems + 255) / 256, 256>>>(k, kv_elems);
	init_bf16<<<(kv_elems + 255) / 256, 256>>>(v, kv_elems);
	std::vector<int> h_table(blocks);
	for (int i = 0; i < blocks; ++i) h_table[i] = i;
	cudaMemcpy(cu_seqlens, h_cu.data(), h_cu.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pos_offset, starts.data(), starts.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(block_table, h_table.data(), h_table.size() * sizeof(int), cudaMemcpyHostToDevice);
	auto run = [&] {launch_attention(out, q, k, v, cu_seqlens, pos_offset, block_table, B, NH, NKV, HS, blocks_per_seq, total, window_size, 0);};
	float us = time_us(run, 20, 100);
	printf("B=%d T=%d H=%d window=%d custom=%.3f us\n", B, total, max_history, window_size, us);
	cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(out);
	cudaFree(cu_seqlens); cudaFree(pos_offset); cudaFree(block_table);
}

int main() {
	bench_case({128}, {0}, -1);
	bench_case({256}, {0}, -1);
	bench_case({512}, {0}, -1);
	bench_case(std::vector<int>(8, 1), std::vector<int>(8, 511), -1);
	bench_case(std::vector<int>(32, 1), std::vector<int>(32, 511), -1);
	bench_case({7,31,32,33,65}, {31,47,127,511,1023}, 33);
}
