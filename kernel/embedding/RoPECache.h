#pragma once
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__global__ void init_rope_table(
	float *cos_table,
	float *sin_table,
	int max_seq_len, int HS, float base
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int half = HS / 2;
	int total = max_seq_len * half;
	if (index >= total) return;
	int pos = index / half;
	int i = index % half;
	float inv_freq = powf(base, -2.0f * i / HS);
	float angle = pos * inv_freq;
	cos_table[index] = cosf(angle);
	sin_table[index] = sinf(angle);
}
