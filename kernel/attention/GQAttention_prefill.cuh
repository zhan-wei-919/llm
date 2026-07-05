#pragma once
#include <cassert>
#include "../reduce/Reduce.cuh"
#define GQA_MAX_SEQ_LEN 4096

template <typename T>
__global__ void
gq_attention_prefill(
	T 		*__restrict__ out, 		// [B, T, C]
	const T 	*__restrict__ q,		// [B, T, NH * HS]
	const T 	*__restrict__ k, 		// [B, T, NKV * HS]
	const T 	*__restrict__ v,		// [B, T, NKV * HS]
	int seq_len, int NH, int NKV, int HS
) {
	// 每个 block 算位置 i 的第 h 个 q 头
	int q_stride = NH * HS;
	int kv_stride = NKV * HS;
	int i = blockIdx.x % seq_len;
	int h = blockIdx.x / seq_len % NH;
	int b = blockIdx.x / seq_len / NH;
	int GROPU = NH / NKV;
	int g = h / GROPU;
	float scale = rsqrtf((float)HS);
	__shared__ float scores[GQA_MAX_SEQ_LEN];

	const T *q_i = q + (size_t)(b * seq_len + i) * q_stride + h * HS;
	for (int j = threadIdx.x; j <= i; j += blockDim.x) {
		float acc = 0.0f;
		const T *k_j = k + (size_t)(b * seq_len + j) * kv_stride + g * HS;
		for (int d = 0; d < HS; ++d) {
			acc += static_cast<float>(q_i[d]) * static_cast<float>(k_j[d]);
		}
		scores[j] = acc * scale;
	}

	float local_max = -INFINITY;
	for (int j = threadIdx.x; j <= i; j += blockDim.x) {
		local_max = device_max(local_max, scores[j]);
	}
	float row_max = block_max(local_max);
	float local_sum = 0;
	for (int j = threadIdx.x; j <= i; j += blockDim.x) {
		float e = expf(scores[j] - row_max);
		local_sum += e;
		scores[j] = e;
	}
	float Z = block_sum(local_sum); float inv_z = 1.0f / Z;

	T *out_i = out + (size_t)(b * seq_len + i) * q_stride + h * HS;
	for (int d = threadIdx.x; d < HS; d += blockDim.x) {
		float acc = 0.0f;
		for (int j = 0; j <= i; ++j) {
			const T *v_j = v + (size_t)(b * seq_len + j) * kv_stride + g * HS;
			acc += (scores[j] * inv_z) * static_cast<float>(v_j[d]);
		}
		out_i[d] = static_cast<T>(acc);
	}
}

template <typename T>
void launch_gq_attention_prefill(
	T 		*out, 		// [B, T, C]
	const T 	*q,		// [B, T, NH * HS]
	const T 	*k, 		// [B, T, NKV * HS]
	const T 	*v,		// [B, T, NKV * HS]
	int B, int seq_len, int NH, int NKV, int HS
) {
	int grid = B * seq_len * NH;
	int block = 256;
	gq_attention_prefill<T><<<grid, block>>>(out, q, k, v, seq_len, NH, NKV, HS);
}
