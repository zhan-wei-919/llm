#pragma once
#include <cassert>
#include "../reduce/Reduce.cuh"
#include <cmath>
#include <cstddef>

#define MAX_SEQ_LEN 1024

template <typename T>
__global__ void attention(T *__restrict__ out,       // [B, T, C]
                          T *__restrict__ att,       // [B, NH, T, T]
                          const T *__restrict__ qkv, // [B, T, 3C]
                          int B, int seq_len, int C, int NH) {
	int i = blockIdx.x % seq_len;
	int h = blockIdx.x / seq_len % NH;
	int b = blockIdx.x / seq_len / NH;
	int HS = C / NH;
	float scale = rsqrtf((float)HS);
	__shared__ float scores[MAX_SEQ_LEN];

	const T *q_i = qkv + (size_t)(b * seq_len + i) * 3 * C + 0 * C + h * HS;
	for (int j = threadIdx.x; j <= i; j += blockDim.x) {
		float acc = 0.0f;
		const T *k_j =
		    qkv + (size_t)(b * seq_len + j) * 3 * C + 1 * C + h * HS;
		for (int d = 0; d < HS; ++d) {
			acc += static_cast<float>(q_i[d]) *
			       static_cast<float>(k_j[d]);
		}
		scores[j] = acc * scale;
	}

	float local_max = -INFINITY;
	for (int j = threadIdx.x; j <= i; j += blockDim.x) {
		local_max = device_max(local_max, scores[j]);
	}
	float row_max = block_max(local_max);
	float local_sum = 0.0f;
	for (int j = threadIdx.x; j <= i; j += blockDim.x) {
		float e = expf(scores[j] - row_max);
		local_sum += e;
		scores[j] = e;
	}
	float Z = block_sum(local_sum);
	float inv_z = 1.0f / Z;

	T *out_i = out + (size_t)(b * seq_len + i) * C + h * HS;
	for (int d = threadIdx.x; d < HS; d += blockDim.x) {
		float acc = 0;
		for (int j = 0; j <= i; ++j) {
			const T *v_j = qkv + (size_t)(b * seq_len + j) * 3 * C + 2 * C + h * HS;
			float tmp = scores[j] * inv_z;
			acc += tmp * static_cast<float>(v_j[d]);
		}
		out_i[d] = static_cast<T>(acc);
	}

	T *att_i = att + ((size_t)(b * NH + h) * seq_len + i) * seq_len;
	for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
		float tmp = (j <= i) ? scores[j] * inv_z : 0.0f;
		att_i[j] = static_cast<T>(tmp);
	}
}

template <typename T>
void launch_attention_forward(T *__restrict__ out,       // [B, T, C]
                              T *__restrict__ att,       // [B, NH, T, T]
                              const T *__restrict__ qkv, // [B, T, 3C]
                              int B, int seq_len, int C, int NH, cudaStream_t t) {
	int grid = B * seq_len * NH;
	int block = 256;
	attention<T><<<grid, block, 0, t>>>(out, att, qkv, B, seq_len, C, NH);
}
