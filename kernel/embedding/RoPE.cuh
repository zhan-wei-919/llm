#pragma once
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template<typename T>
__global__ void rope(
	T		*__restrict__	x,
	const float	*__restrict__	cos_table,
	const float	*__restrict__	sin_table,
	int NH, int HS, int pos0
) {
	int t = blockIdx.x / NH;
	int h = blockIdx.x % NH;
	int i = threadIdx.x;
	int half = HS / 2;

	float c = cos_table[(size_t)(pos0 + t) * half + i];
	float s = sin_table[(size_t)(pos0 + t) * half + i];

	T *head = x + (size_t)t * NH * HS + (size_t)h * HS;
	float x0 = static_cast<float>(head[i]);
	float x1 = static_cast<float>(head[i + half]);
	head[i] = static_cast<T>(x0 * c - x1 * s);
	head[i + half] = static_cast<T>(x0 * s + x1 * c);
}

template<typename T>
void launch_rope(
    T *x, const float *cos_table, const float *sin_table,
    int seq_len, int n_heads, int HS, int pos0, int max_seq, cudaStream_t t
) {
    assert(HS % 2 == 0);
    assert(pos0 + seq_len <= max_seq);      // cos 表行数上限
    int grid  = seq_len * n_heads;          // 一个 block = 一个 token 的一个头
    int block = HS / 2;                     // 64 线程 = 64 个小平面
    rope<T><<<grid, block, 0, t>>>(x, cos_table, sin_table, n_heads, HS, pos0);
}
