#pragma once
#include <cassert>
#include "../reduce/Reduce.cuh"
#define GQA_MAX_SEQ_LEN 4096

//
// 对于attention的prefill的时候, 要实现paged attention, 有两种方案
// 这里采用的是稠密输入, 另一条线路写进cache
//
// k/v 投影 → [稠密激活] ──→ prefill attention (旧连续 kernel)
// 	└─────→ paged Gather_kv 散射进池子 (为 decode 备货)
//
// Prefill 的多 batch，核心矛盾只有一个：B 个 prompt 长短不齐（比如 12,300,1500）
// 解决方法便是把 B 条序列首尾相接摊平
// prompt长度:  12, 300, 1500
// 打包后:       [t_0...t_11 | t_12...t_311 | t_312...t_1811]   共 total = 1812 行
// 但是打包后, 怎么确定当前这个token属于第几个batch呢?
// 通过一个seq_ids来实现,
// t		0 ... 11 | 12 ... 311 | 312 ... 1811
// seq_ids:	0 ... 0  | 1  ...   1 | 2   ...    2
// 比如t = 198, 只要查 seq_ids[t] 就可以知道当前这个token是b=1
// 但是怎么通过当前序列号知道当前这个序列有多长呢?
// 需要 cu_seqlens = [0, 12, 312, 1812]  直接查 cu_seqlens[seq_ids[t] + 1] - cu_seqlens[seq_ids[t]] = 300
//
// 当然, 也可以不用这个seq_ids, 而是直接通过二分,
// 因为 cu_seqlens 是一个单调递增的序列, 二分天然可以找到对应的位置
//

template <typename T>
__global__ void
gq_attention_prefill(
	T 		*__restrict__ out, 		// [T, NH * HS]
	const T 	*__restrict__ q,		// [T, NH * HS]
	const T 	*__restrict__ k, 		// [T, NKV * HS]
	const T 	*__restrict__ v,		// [T, NKV * HS]
	const int	*__restrict__ cu_seqlens,	// [B]
	const int	*__restrict__ seq_ids,		// [T]
	int B, int total_len, int NH, int NKV, int HS
) {
	int t = blockIdx.x, b = seq_ids[t], len = cu_seqlens[b + 1] - cu_seqlens[b];
	int i = t - cu_seqlens[b];
	int h = blockIdx.y;
	int GROUP = NH / NKV, g = h / GROUP;
	float scale = rsqrtf((float)HS);
	__shared__ float scores[GQA_MAX_SEQ_LEN];

	const int q_stride = NH * HS, kv_stride = NKV * HS;
	const T *q_b_h = q + (size_t)t * q_stride + h * HS;
	for (int j = threadIdx.x; j <= i; j += blockDim.x) {
		float acc = 0.0f;
		const T *k_b_j = k + (size_t)(cu_seqlens[b] + j) * kv_stride + g * HS;
		for (int d = 0; d < HS; ++d) {
			acc += static_cast<float>(q_b_h[d]) * static_cast<float>(k_b_j[d]);
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
	float Z = block_sum(local_sum), inv_z = 1.0f / Z;

	T *out_b_i = out + (size_t)t * q_stride + h * HS;
	for (int d = threadIdx.x; d < HS; d += blockDim.x) {
		float acc = 0.0f;
		for (int j = 0; j <= i; ++j) {
			const T *v_b_j = v + (size_t)(cu_seqlens[b] + j) * kv_stride + g * HS;
			acc += static_cast<float>(v_b_j[d]) * scores[j] * inv_z;
		}
		out_b_i[d] = static_cast<T>(acc);
	}
}

template <typename T>
void launch_gq_attention_prefill(
	T 		*__restrict__ out, 		// [T, NH * HS]
	const T 	*__restrict__ q,		// [T, NH * HS]
	const T 	*__restrict__ k, 		// [T, NKV * HS]
	const T 	*__restrict__ v,		// [T, NKV * HS]
	const int	*__restrict__ cu_seqlens,	// [B]
	const int	*__restrict__ seq_ids,		// [T]
	int B, int total_len, int NH, int NKV, int HS
) {
	dim3 grid (total_len, NH);
	int block = 256;
	gq_attention_prefill<T><<<grid, block>>>(out, q, k, v, cu_seqlens, seq_ids,B, total_len, NH, NKV, HS);
}
