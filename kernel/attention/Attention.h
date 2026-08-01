#pragma once
#include <cstdint>
#include "GQAttention_prefill.h"

template<typename T>
// Q=[total,NH,HS]，K/V=[blocks,KV_BLOCK_SIZE,NKV,HS]；B、total、window_size 由 launcher 处理。
void launch_attention(
	T *out, const T *q, const T *k_pool, const T *v_pool,
	const int *cu_seqlens, const int *pos_offset, const int *block_table,
	int B, int NH, int NKV, int HS, int max_blocks_per_seq, int total, int window_size, cudaStream_t stream
) {
	uint64_t shape = ((uint64_t)NH << 32) | ((uint64_t)NKV << 16) | (uint64_t)HS;
	switch (shape) {
			case ((uint64_t)32 << 32) | ((uint64_t)4 << 16) | 64:
				launch_gq_attention_prefill(
					out, q, k_pool, v_pool, cu_seqlens, pos_offset, block_table,
					B, max_blocks_per_seq, total, window_size, stream);
				break;
	}
}
