#pragma once
#include <cassert>
#include "../reduce/Reduce.cuh"
#define GQA_MAX_SEQ_LEN 4096

template<typename T>
__global__ void gq_attention_decode(
	T	*__restrict__	out,			// [1, NH * HS]
	const T *__restrict__	q,			// [1, NH * HS]
	const T *__restrict__	k_cache,		// [t_max, NKV * HS]
	const T *__restrict__	v_cache,		// [t_max, NKV * HS]
	int cur_len, int NH, int NKV, int HS
) {
	int h = blockIdx.x;
	int GROUP = NH / NKV;
	int g = h / GROUP;
	int kv_stride = NKV * HS;
	float scale = rsqrtf((float)HS);
	__shared__ float scores[GQA_MAX_SEQ_LEN];

	const T *q_h = q + (size_t)h * HS;
	for (int j = threadIdx.x; j < cur_len; j += blockDim.x) {
		float acc = 0;
		const T *k_j = k_cache + (size_t)j * kv_stride + g * HS;
		for (int d = 0; d < HS; ++d) {
			acc += static_cast<float>(q_h[d]) * static_cast<float>(k_j[d]);
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
	float Z = block_sum(local_sum); float inv_z = 1.0f / Z;

	T *out_h = out + (size_t)h * HS;
	for (int d = threadIdx.x; d < HS; d += blockDim.x) {
		float acc = 0.0f;
		for (int j = 0; j < cur_len; ++j) {
			const T *v_j = v_cache + (size_t)j * kv_stride + g * HS;
			acc += scores[j] * inv_z * static_cast<float>(v_j[d]);
		}
		out_h[d] = static_cast<T>(acc);
	}
}

template<typename T>
void launch_gq_attention_decode(
	T	*__restrict__	out,
	const T *__restrict__	q,
	const T *__restrict__	k_cache,
	const T *__restrict__	v_cache,
	int cur_len, int NH, int NKV, int HS
) {
	int grid = NH;
	int block = 256;
	gq_attention_decode<T><<<grid, block>>>(out, q, k_cache, v_cache, cur_len, NH, NKV, HS);
}
