#pragma once
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "../../paged_attention/KV_Layout.h"

// scatter_kv的作用是把decode的计算结果和prefill的计算结果都写进kv pool
// 为此, 我们需要一个pos_offset的辅助数组来表明当前的这个B下一个位置应该往哪儿写
//
// 	prefill
// 	k_src/v_src:	[10, kv_stride]
// 	cu_seqlens:	[0, 3, 8, 10]	// 一共三个batch
// 	seq_ids:	[0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
// 	pos_offset:	[0, 0, 0]	// prefill的时候大家都是从头开始写
//
// 	decode的时候有不同, 因为每个batch的长度都是1
// 	k_src/v_src:	[3, kv_stride]
// 	cu_seqlens:	[0, 1, 2, 3]
// 	seq_ids:	[0, 1, 2]
// 	pos_offset:	[8, 59, 32]
//
// 那么实际上还是一样的顺序找到自己的batch,
// 设定发射参数grid为total, 总的行数, 那么找到当前的行数就是 s = blockIdx.x,
// 在通过当前行数, 用seq_ids找到当前的batch, b = seq_idx[s];
// 找到自己的batch后, 还需要确认当前是batch里的第几行, 则 i = s - cu_seqlens[b] -> 在decode下cu_seqlens是一个自然数序列, 所以 s - cu_seqlens[b]一定为0
// 除此之外, 还需要通过pos_offset找到自己的历史序列位置p = pos_offset[s] + i, 再对block_table查表(注意要对p>>4查表).
// 查表时就回到了之前的 p / KV_BLOCK_SIZE, p % KV_BLOCK_SIZE 来确定位置
// 但是scatter是不需要在一当前是第几个head, 直接一行整体来搬运就行

template <typename T>
__global__ void scatter_kv(
	T		*__restrict__	k_pool,		// [num_blocks, KV_BLOCK_SIZE, NKV*HS]
	T		*__restrict__	v_pool,		// [num_blocks, KV_BLOCK_SIZE, NKV*HS]
	const	T	*__restrict__	k_src,		// [total, NKV * HS]
	const	T	*__restrict__	v_src,		// [total, NKV * HS]
	const	int	*__restrict__	block_table,	// [B, max_blocks_per_seq]
	const	int	*__restrict__	cu_seqlens,	// [B + 1]
	const	int	*__restrict__	seq_ids,	// [total]
	const	int	*__restrict__	pos_offset,	// [B]
	int total, int NKV, int HS, int max_blocks_per_seq
) {
	int s = blockIdx.x, b = seq_ids[s];
	int i = s - cu_seqlens[b], p = pos_offset[b] + i;
	const int *table = block_table + (size_t)b * max_blocks_per_seq;
	int phys = table[p / KV_BLOCK_SIZE], offset_in_block = p % KV_BLOCK_SIZE;
	T *k_target = k_pool + ((size_t)phys * KV_BLOCK_SIZE + offset_in_block)* (size_t)(NKV * HS);
	T *v_target = v_pool + ((size_t)phys * KV_BLOCK_SIZE + offset_in_block)* (size_t)(NKV * HS);
	const T *k_from = k_src + (size_t)s * NKV * HS;
	const T *v_from = v_src + (size_t)s * NKV * HS;
	for (int j = threadIdx.x; j < NKV * HS; j += blockDim.x) {
		k_target[j] = k_from[j];
		v_target[j] = v_from[j];
	}
}

template <typename T>
void launch_scatter_kv(
	T		*__restrict__	k_pool,		// [num_blocks, KV_BLOCK_SIZE, NKV*HS]
	T		*__restrict__	v_pool,		// [num_blocks, KV_BLOCK_SIZE, NKV*HS]
	const	T	*__restrict__	k_src,		// [total, NKV * HS]
	const	T	*__restrict__	v_src,		// [total, NKV * HS]
	const	int	*__restrict__	block_table,	// [B, max_blocks_per_seq]
	const	int	*__restrict__	cu_seqlens,	// [B + 1]
	const	int	*__restrict__	seq_ids,	// [total]
	const	int	*__restrict__	pos_offset,	// [B]
	int total, int NKV, int HS, int max_blocks_per_seq
) {
	int grid = total;
	int block = 256;
	scatter_kv<T><<<grid, block>>>(k_pool, v_pool, k_src, v_src, block_table, cu_seqlens, seq_ids, pos_offset, total, NKV, HS, max_blocks_per_seq);
}
