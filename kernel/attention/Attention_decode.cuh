#pragma once
#include <cassert>
#include "../reduce/Reduce.cuh"
#include <cmath>
#include <cstddef>
#define MAX_SEQ_LEN 1024

template <typename T>
__global__ void attention_decode(T *__restrict__ out,           // [1, C]
                                 const T *__restrict__ qkv,     // [1, 3C]
                                 const T *__restrict__ k_cache, // [t_max, C]
                                 const T *__restrict__ v_cache, // [t_max, C]
                                 int cur_len, int C, int NH) {
	int h = blockIdx.x;
	int HS = C / NH;
	float scale = rsqrtf((float)HS);

	__shared__ float scores[MAX_SEQ_LEN];

	const T *q = qkv + (size_t)h * HS;
	for (int j = threadIdx.x; j < cur_len; j += blockDim.x) {
		float acc = 0.0f;
		const T *k_j = k_cache + j * C + h * HS;
		for (int d = 0; d < HS; ++d) {
			acc += static_cast<float>(q[d]) *
			       static_cast<float>(k_j[d]);
		}
		scores[j] = acc * scale;
	}

	float local_max = -INFINITY;
	for (int j = threadIdx.x; j < cur_len; j += blockDim.x) {
		local_max = device_max(local_max, scores[j]);
	}
	float row_max = block_max(local_max);
	float local_sum = 0.0f;
	for (int j = threadIdx.x; j < cur_len; j += blockDim.x) {
		float e = expf(scores[j] - row_max);
		local_sum += e;
		scores[j] = e;
	}
	float Z = block_sum(local_sum);
	float inv_z = 1.0f / Z;

	T *out_i = out + (size_t)h * HS;
	for (int d = threadIdx.x; d < HS; d += blockDim.x) {
		float acc = 0;
		for (int j = 0; j < cur_len; ++j) {
			const T *v_j = v_cache + j * C + h * HS;
			float tmp = scores[j] * inv_z;
			acc += tmp * static_cast<float>(v_j[d]);
		}
		out_i[d] = static_cast<T>(acc);
	}
}

template <typename T>
void launch_attention_decode_forward(
    T *__restrict__ out,           // [1, C]
    const T *__restrict__ qkv,     // [1, 3C]
    const T *__restrict__ k_cache, // [t_max, C]
    const T *__restrict__ v_cache, // [t_max, C]
    int cur_len, int C, int NH) {
	int grid = NH;
	int block = 256;
	attention_decode<T><<<grid, block>>>(out, qkv, k_cache, v_cache, cur_len, C, NH);
}
