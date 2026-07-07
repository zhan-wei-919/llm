#pragma once
#include <cassert>
#include "../reduce/Reduce.cuh"
#include "../../paged_attention/KV_Layout.h"
#define GQA_MAX_SEQ_LEN 4096

//
// 物理块0: [ 16格 ]    物理块1: [ 16格 ]    物理块2: [ 16格 ]    物理块3: [ 16格 ] ...
//  ↑序列A的第2个逻辑块      ↑空闲              ↑序列B的第0个逻辑块
// 传入的是kv_pool, 那么也就是 [num_blocks, block_size, NKV * HS]
// 之前有找k_j, 是k_j = k + (size_t)(b * seq_len + j) * kv_stride + g * HS;
// 但是现在是直接查表, 找到当前的k_j距离k_pool的起始位置的差距, 单位为blocks
// 我们设定pool为[num_blocks, block_size, NKV * HS]
// 其中, num_blocks为总共的块大小, block_size一般都取16
// 我们还需要两个基址, k_pool和v_pool, 所以计算就是这样的
//
// 	假设某个k_line, 他的属于 j = 47 = 32 + 8 + 4 + 2 + 1 也就是 0010 1111
// 	在 block_size = 16时, 低四位是块内偏移, 高位就是块号, 所以这个 k_47 属于第二块的第15行
// 	但是这里的第二块是逻辑块的第二块, 要对应到真实的物理块需要经过 block_table 的映射,
// 	一块的大小就是block_size * NKV * HS
//
// int phys = block_table[j >> 4]	// 这里是设定的block_size = 16
// T *k_j = k_pool + (size_t)phys * block_size * NKV * HS + NKV * HS * (j & 15)
//
//
template<typename T>
__global__ void gq_attention_decode(
	T		*__restrict__	out,			// [B, NH * HS]
	const T		*__restrict__	q,			// [B, NH * HS]
	const T		*__restrict__	k_pool,			// [num_blocks, KV_BLOCK_SIZE, NKV*HS]
	const T 	*__restrict__	v_pool,			// [num_blocks, KV_BLOCK_SIZE, NKV*HS]
	const int	*__restrict__	block_table,		// [B, max_blocks_per_seq]
	const int 	*__restrict__	cur_len,		// [B]
	int B, int NH, int NKV, int HS, int max_blocks_per_seq
) {
	int h = blockIdx.x, b = blockIdx.y;
	int len = cur_len[b];
	int GROPU = NH / NKV, g = h / GROPU;
	float scale = rsqrtf((float)HS);
	__shared__ float scores[GQA_MAX_SEQ_LEN];

	const int * table = block_table + (size_t)b * max_blocks_per_seq;
	const T *q_b = q + b * NH * HS + h * HS;

	for (int j = threadIdx.x; j < len; j += blockDim.x) {
		float acc = 0.0f;
		int phys = table[j / KV_BLOCK_SIZE], row = j % KV_BLOCK_SIZE;
		const T *k_j = k_pool + ((size_t)phys * KV_BLOCK_SIZE * NKV * HS + row * NKV * HS + g * HS);
		for (int d = 0; d < HS; ++d) {
			acc += static_cast<float>(k_j[d]) * static_cast<float>(q_b[d]);
		}
		scores[j] = acc * scale;
	}

	float local_max = -INFINITY;
	for (int j = threadIdx.x; j < len; j += blockDim.x) {
		local_max = device_max(local_max, scores[j]);
	}
	float row_max = block_max(local_max);
	float local_sum = 0;
	for (int j = threadIdx.x; j < len; j += blockDim.x) {
		float e = expf(scores[j] - row_max);
		local_sum += e;
		scores[j] = e;
	}
	float Z = block_sum(local_sum); float inv_z = 1.0f / Z;

	T *out_b = out + b * NH * HS + h * HS;
	for (int d = threadIdx.x; d < HS; d += blockDim.x) {
		float acc = 0.0f;
		for (int j = 0; j < len; ++j) {
			int phys = table[j / KV_BLOCK_SIZE], row = j % KV_BLOCK_SIZE;
			const T *v_j = v_pool + ((size_t)phys * KV_BLOCK_SIZE * NKV * HS + row * NKV * HS + g * HS);
			acc += static_cast<float>(v_j[d]) * scores[j] * inv_z;
		}
		out_b[d] = static_cast<T>(acc);
	}
}

template<typename T>
void launch_gq_attention_decode(
	T		*__restrict__	out,			// [B, NH * HS]
	const T		*__restrict__	q,			// [B, NH * HS]
	const T		*__restrict__	k_pool,			// [num_blocks, KV_BLOCK_SIZE, NKV*HS]
	const T 	*__restrict__	v_pool,			// [num_blocks, KV_BLOCK_SIZE, NKV*HS]
	const int	*__restrict__	block_table,		// [B, max_blocks_per_seq]
	const int 	*__restrict__	cur_len,		// [B]
	int B, int NH, int NKV, int HS, int max_blocks_per_seq
) {
	dim3 grid (NH, B);
	int block = 256;
	gq_attention_decode<T><<<grid, block>>>(out, q, k_pool, v_pool, block_table, cur_len, B, NH, NKV, HS, max_blocks_per_seq);
}
