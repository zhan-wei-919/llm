#pragma once
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template<typename T>
__global__ void rope (
	T *x,
	const float *cos_table,
	const float *sin_table,
	const int *cu_seqlens,
	const int *seq_ids,
	const int *pos_offset,
	int NH, int HS
) {
	int t = blockIdx.x / NH;
	int h = blockIdx.x % NH;
	int i = threadIdx.x;

	int b = seq_ids[t], local_pos = t - cu_seqlens[b], pos = pos_offset[b] + local_pos;

	int half = HS / 2;
	float c = cos_table[(size_t)pos * half + i];
	float s = sin_table[(size_t)pos * half + i];

	T *head = x + (size_t)t * NH * HS + (size_t)h * HS;

	float x0 = static_cast<float>(head[i]);
	float x1 = static_cast<float>(head[i + half]);

	head[i] = static_cast<T>(x0 * c - x1 * s);
	head[i + half] = static_cast<T>(x0 * s + x1 * c);
}

template<typename T>
void launch_rope(
	T *x,
	const float *cos_table,
	const float *sin_table,
	const int *cu_seqlens,
	const int *seq_ids,
	const int *pos_offset,
	int total, int NH, int HS, cudaStream_t stream
) {
	int grid = total * NH;
	int block = HS / 2;
	rope<T><<<grid, block, 0, stream>>>(x, cos_table, sin_table, cu_seqlens, seq_ids, pos_offset, NH, HS);
}
