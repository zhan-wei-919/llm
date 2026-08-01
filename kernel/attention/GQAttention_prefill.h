#pragma once
#include "../Gemm/MmaPtx.h"
#include "../../kv/KV_Layout.h"

constexpr int GQA_Q_TILE = 16, GQA_GROUP = 8, GQA_HEAD_WARPS = 4, GQA_PAD = 8;
constexpr int GQA_NH = 32, GQA_NKV = 4, GQA_HEAD_SIZE = 64;
constexpr int GQA_SMEM_Q1_K32 = 15 * 1024, GQA_SMEM_Q1_K64 = 27 * 1024, GQA_SMEM_Q2_K32 = 20 * 1024;

__device__ __forceinline__ float gqa_group_max(float x) {
	x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
	x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
	return x;
}

__device__ __forceinline__ float gqa_group_sum(float x) {
	x += __shfl_xor_sync(0xffffffff, x, 1);
	x += __shfl_xor_sync(0xffffffff, x, 2);
	return x;
}

template<int K_TILE, int Q_TILES, bool SWAP_GRID>
// BF16 GQA 专用核：固定 NH=32、NKV=4、HS=64，直接从分页 KV Pool 完成 causal attention。
__global__ __launch_bounds__(GQA_HEAD_WARPS * Q_TILES * 32, (Q_TILES == 1 ? (K_TILE == 32 ? 4 : 3) : 2)) void gq_attention_prefill_bf16_h64(
	__nv_bfloat16 *__restrict__ out, const __nv_bfloat16 *__restrict__ q,
	const __nv_bfloat16 *__restrict__ k_pool, const __nv_bfloat16 *__restrict__ v_pool,
	const int *__restrict__ cu_seqlens, const int *__restrict__ pos_offset,
	const int *__restrict__ block_table, int B, int max_blocks_per_seq,
	int total, int window_size
) {
#if __CUDA_ARCH__ >= 800
	constexpr int QLD = 64 + GQA_PAD, KLD = K_TILE + GQA_PAD, VLD = 64 + GQA_PAD;
	constexpr int WARPS = GQA_HEAD_WARPS * Q_TILES, Q_BLOCK = GQA_Q_TILE * Q_TILES;
	int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
	constexpr int HEAD_SPLITS = GQA_GROUP / GQA_HEAD_WARPS;
	int task = SWAP_GRID ? blockIdx.y : blockIdx.x;
	int head_block = SWAP_GRID ? blockIdx.x : blockIdx.y;
	__shared__ int batch_meta[4];
	extern __shared__ char smem[];
	int b = 0, seq_start = 0, seq_len = total, local_q_base = task * Q_BLOCK;
	if (B > 1) {
		// 将 packed tile 序号还原到单个序列，保证一个 CTA 不跨越序列边界。
		if (tid == 0) {
			int tile_base = 0, found = -1, found_start = 0, found_len = 0, found_local = 0;
			for (int s = 0; s < B; ++s) {
				int start = cu_seqlens[s], len = cu_seqlens[s + 1] - start;
				int next = tile_base + (len + Q_BLOCK - 1) / Q_BLOCK;
				if (task < next) {
					found = s; found_start = start; found_len = len;
					found_local = (task - tile_base) * Q_BLOCK;
					break;
				}
				tile_base = next;
			}
			batch_meta[0] = found; batch_meta[1] = found_start;
			batch_meta[2] = found_len; batch_meta[3] = found_local;
		}
		__syncthreads();
		b = batch_meta[0]; seq_start = batch_meta[1];
		seq_len = batch_meta[2]; local_q_base = batch_meta[3];
		if (b < 0) return;
	}
	int local_q0 = local_q_base + (warp % Q_TILES) * GQA_Q_TILE;
	int g = head_block / HEAD_SPLITS;
	int h = g * GQA_GROUP + (head_block % HEAD_SPLITS) * GQA_HEAD_WARPS + warp / Q_TILES;
	int pos = pos_offset[b], q_last = min(local_q_base + Q_BLOCK - 1, seq_len - 1);
	int key_begin = window_size < 0 ? 0 : max(0, pos + local_q_base + 1 - window_size);
	int key_limit = pos + q_last + 1;
	auto *sq = reinterpret_cast<__nv_bfloat16*>(smem);

	for (int e = tid; e < WARPS * GQA_Q_TILE * 8; e += blockDim.x) {
		int w = e >> 7, x = e & 127, r = x >> 3, d = (x & 7) * 8;
		int local_qt = local_q_base + (w % Q_TILES) * GQA_Q_TILE + r;
		int qt = seq_start + local_qt;
		int qh = g * GQA_GROUP + (head_block % HEAD_SPLITS) * GQA_HEAD_WARPS + w / Q_TILES;
		uint4 qv = {};
		if (local_qt < seq_len) qv = *reinterpret_cast<const uint4*>(q + ((size_t)qt * GQA_NH + qh) * GQA_HEAD_SIZE + d);
		*reinterpret_cast<uint4*>(&sq[(w * GQA_Q_TILE + r) * QLD + d]) = qv;
	}
	__syncthreads();

	int row16 = lane & 15, colh = (lane >> 4) * 8;
	uint32_t q_frag[4][4];
	#pragma unroll
	for (int kk = 0; kk < 64; kk += 16)
		ldmatrix_x4(q_frag[kk >> 4], smem_addr(&sq[(warp * GQA_Q_TILE + row16) * QLD + kk + colh]));
	__syncthreads();

	auto *sk = reinterpret_cast<__nv_bfloat16*>(smem);
	auto *sv = sk + 64 * KLD;
	auto *prob = sv + K_TILE * VLD;

	float out_acc[8][4] = {};
	float row_m0 = -INFINITY, row_l0 = 0.0f, row_m1 = -INFINITY, row_l1 = 0.0f;
	float scale = rsqrtf((float)GQA_HEAD_SIZE);
	const int *table = block_table + (size_t)b * max_blocks_per_seq;
	constexpr int kv_stride = GQA_NKV * GQA_HEAD_SIZE;

	for (int key0 = key_begin; key0 < key_limit; key0 += K_TILE) {
		for (int e = tid; e < K_TILE * 8; e += blockDim.x) {
			int r = e >> 3, d = (e & 7) * 8, key = key0 + r;
			uint4 kval = {}, vval = {};
			int phys = 0;
			if constexpr (K_TILE == 64) {
				int lookup_key = min(key, key_limit - 1);
				if ((lane & 7) == 0) phys = table[lookup_key / KV_BLOCK_SIZE];
				phys = __shfl_sync(0xffffffff, phys, lane & ~7);
			} else if (key < key_limit) {
				phys = table[key / KV_BLOCK_SIZE];
			}
			if (key < key_limit) {
				int row = key % KV_BLOCK_SIZE;
				size_t off = ((size_t)phys * KV_BLOCK_SIZE + row) * kv_stride + g * GQA_HEAD_SIZE + d;
				kval = *reinterpret_cast<const uint4*>(k_pool + off);
				vval = *reinterpret_cast<const uint4*>(v_pool + off);
			}
			*reinterpret_cast<uint4*>(&sk[r * VLD + d]) = kval;
			*reinterpret_cast<uint4*>(&sv[r * VLD + d]) = vval;
		}
		__syncthreads();

		float score_acc[K_TILE / 8][4] = {};
		#pragma unroll
		for (int kk = 0; kk < 64; kk += 16) {
			#pragma unroll
			for (int nb = 0; nb < K_TILE / 16; ++nb) {
				uint32_t rb[4], b0[2], b1[2];
				ldmatrix_x4(rb, smem_addr(&sk[(nb * 16 + row16) * VLD + kk + colh]));
				b0[0] = rb[0]; b0[1] = rb[2];
				b1[0] = rb[1]; b1[1] = rb[3];
				mma_m16n8k16<true>(score_acc[nb * 2], q_frag[kk >> 4], b0);
				mma_m16n8k16<true>(score_acc[nb * 2 + 1], q_frag[kk >> 4], b1);
			}
		}

		int rg = lane >> 2, cg = (lane & 3) * 2;
		int local_row0 = local_q0 + rg, local_row1 = local_row0 + 8;
		int valid0 = local_row0 < seq_len, valid1 = local_row1 < seq_len;
		int begin0 = window_size < 0 ? 0 : max(0, pos + local_row0 + 1 - window_size);
		int begin1 = window_size < 0 ? 0 : max(0, pos + local_row1 + 1 - window_size);
		float tile_m0 = -INFINITY, tile_m1 = -INFINITY;
		#pragma unroll
		for (int ni = 0; ni < K_TILE / 8; ++ni) {
			int c = ni * 8 + cg, key = key0 + c;
			score_acc[ni][0] = valid0 && key >= begin0 && key <= pos + local_row0 ? score_acc[ni][0] * scale : -INFINITY;
			score_acc[ni][1] = valid0 && key + 1 >= begin0 && key + 1 <= pos + local_row0 ? score_acc[ni][1] * scale : -INFINITY;
			score_acc[ni][2] = valid1 && key >= begin1 && key <= pos + local_row1 ? score_acc[ni][2] * scale : -INFINITY;
			score_acc[ni][3] = valid1 && key + 1 >= begin1 && key + 1 <= pos + local_row1 ? score_acc[ni][3] * scale : -INFINITY;
			tile_m0 = fmaxf(tile_m0, fmaxf(score_acc[ni][0], score_acc[ni][1]));
			tile_m1 = fmaxf(tile_m1, fmaxf(score_acc[ni][2], score_acc[ni][3]));
		}
		tile_m0 = gqa_group_max(tile_m0);
		tile_m1 = gqa_group_max(tile_m1);
		float new_m0 = valid0 ? fmaxf(row_m0, tile_m0) : 0.0f;
		float new_m1 = valid1 ? fmaxf(row_m1, tile_m1) : 0.0f;
		float alpha0 = valid0 && row_m0 != -INFINITY ? __expf(row_m0 - new_m0) : 0.0f;
		float alpha1 = valid1 && row_m1 != -INFINITY ? __expf(row_m1 - new_m1) : 0.0f;
		float tile_l0 = 0.0f, tile_l1 = 0.0f;
		#pragma unroll
		for (int ni = 0; ni < K_TILE / 8; ++ni) {
			score_acc[ni][0] = valid0 ? __expf(score_acc[ni][0] - new_m0) : 0.0f;
			score_acc[ni][1] = valid0 ? __expf(score_acc[ni][1] - new_m0) : 0.0f;
			score_acc[ni][2] = valid1 ? __expf(score_acc[ni][2] - new_m1) : 0.0f;
			score_acc[ni][3] = valid1 ? __expf(score_acc[ni][3] - new_m1) : 0.0f;
			tile_l0 += score_acc[ni][0] + score_acc[ni][1];
			tile_l1 += score_acc[ni][2] + score_acc[ni][3];
			int c = ni * 8 + cg;
			prob[(warp * GQA_Q_TILE + rg) * KLD + c] = __float2bfloat16_rn(score_acc[ni][0]);
			prob[(warp * GQA_Q_TILE + rg) * KLD + c + 1] = __float2bfloat16_rn(score_acc[ni][1]);
			prob[(warp * GQA_Q_TILE + rg + 8) * KLD + c] = __float2bfloat16_rn(score_acc[ni][2]);
			prob[(warp * GQA_Q_TILE + rg + 8) * KLD + c + 1] = __float2bfloat16_rn(score_acc[ni][3]);
		}
		tile_l0 = gqa_group_sum(tile_l0);
		tile_l1 = gqa_group_sum(tile_l1);
		row_m0 = new_m0; row_l0 = alpha0 * row_l0 + tile_l0;
		row_m1 = new_m1; row_l1 = alpha1 * row_l1 + tile_l1;
		__syncwarp();

		#pragma unroll
		for (int ni = 0; ni < 8; ++ni) {
			out_acc[ni][0] *= alpha0;
			out_acc[ni][1] *= alpha0;
			out_acc[ni][2] *= alpha1;
			out_acc[ni][3] *= alpha1;
		}

		#pragma unroll
		for (int kk = 0; kk < K_TILE; kk += 16) {
			uint32_t pa[4];
			ldmatrix_x4(pa, smem_addr(&prob[(warp * GQA_Q_TILE + row16) * KLD + kk + colh]));
			#pragma unroll
			for (int nb = 0; nb < 4; ++nb) {
				uint32_t rb[4], b0[2], b1[2];
				ldmatrix_x4_trans(rb, smem_addr(&sv[(kk + row16) * VLD + nb * 16 + colh]));
				b0[0] = rb[0]; b0[1] = rb[1];
				b1[0] = rb[2]; b1[1] = rb[3];
				mma_m16n8k16<true>(out_acc[nb * 2], pa, b0);
				mma_m16n8k16<true>(out_acc[nb * 2 + 1], pa, b1);
			}
		}
		__syncthreads();
	}

	int rg = lane >> 2, cg = (lane & 3) * 2;
	float inv_l0 = 1.0f / row_l0, inv_l1 = 1.0f / row_l1;
	#pragma unroll
	for (int ni = 0; ni < 8; ++ni) {
		int d = ni * 8 + cg;
		*reinterpret_cast<__nv_bfloat162*>(&sq[(warp * GQA_Q_TILE + rg) * QLD + d]) =
			__floats2bfloat162_rn(out_acc[ni][0] * inv_l0, out_acc[ni][1] * inv_l0);
		*reinterpret_cast<__nv_bfloat162*>(&sq[(warp * GQA_Q_TILE + rg + 8) * QLD + d]) =
			__floats2bfloat162_rn(out_acc[ni][2] * inv_l1, out_acc[ni][3] * inv_l1);
	}
	__syncwarp();
	#pragma unroll
	for (int r = 0; r < GQA_Q_TILE; ++r) {
		int local_qr = local_q0 + r, qr = seq_start + local_qr;
		if (local_qr < seq_len) {
			size_t off = ((size_t)qr * GQA_NH + h) * GQA_HEAD_SIZE + lane * 2;
			*reinterpret_cast<uint32_t*>(out + off) =
				*reinterpret_cast<uint32_t*>(&sq[(warp * GQA_Q_TILE + r) * QLD + lane * 2]);
		}
	}
#endif
}

// grid.x 覆盖序列内的 query tile，grid.y 覆盖每个 KV head 对应的 query-head 子组。
inline void launch_gq_attention_prefill(
	__nv_bfloat16 *out, const __nv_bfloat16 *q, const __nv_bfloat16 *k_pool,
	const __nv_bfloat16 *v_pool, const int *cu_seqlens, const int *pos_offset,
	const int *block_table, int B, int max_blocks_per_seq, int total,
	int window_size, cudaStream_t stream
) {
	static bool configured = [] {
		cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<32, 2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, GQA_SMEM_Q2_K32);
		cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<32, 2, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
		cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<32, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, GQA_SMEM_Q1_K32);
		cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<32, 1, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
		cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<64, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, GQA_SMEM_Q1_K64);
		cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<64, 1, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
		return true;
	}();
	(void)configured;
	constexpr int head_blocks = GQA_NKV * (GQA_GROUP / GQA_HEAD_WARPS);
	if (total <= 128) {
		int tasks = (total + GQA_Q_TILE * 2 - 1) / (GQA_Q_TILE * 2) + B - 1;
		dim3 grid(tasks, head_blocks), block(GQA_HEAD_WARPS * 2 * 32);
		gq_attention_prefill_bf16_h64<32, 2, false><<<grid, block, GQA_SMEM_Q2_K32, stream>>>(
			out, q, k_pool, v_pool, cu_seqlens, pos_offset, block_table,
			B, max_blocks_per_seq, total, window_size);
	} else if (total <= 256) {
		int tasks = (total + GQA_Q_TILE - 1) / GQA_Q_TILE + B - 1;
		dim3 grid(tasks, head_blocks), block(GQA_HEAD_WARPS * 32);
		gq_attention_prefill_bf16_h64<32, 1, false><<<grid, block, GQA_SMEM_Q1_K32, stream>>>(
			out, q, k_pool, v_pool, cu_seqlens, pos_offset, block_table,
			B, max_blocks_per_seq, total, window_size);
	} else {
		int tasks = (total + GQA_Q_TILE - 1) / GQA_Q_TILE + B - 1;
		dim3 grid(head_blocks, tasks), block(GQA_HEAD_WARPS * 32);
		gq_attention_prefill_bf16_h64<64, 1, true><<<grid, block, GQA_SMEM_Q1_K64, stream>>>(
			out, q, k_pool, v_pool, cu_seqlens, pos_offset, block_table,
			B, max_blocks_per_seq, total, window_size);
	}
}
