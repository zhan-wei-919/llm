#pragma once
#include <type_traits>
#include "../Gemm/MmaPtx.h"
#include "../../kv/KV_Layout.h"

constexpr int GQA_Q_TILE = 16, GQA_GROUP = 8, GQA_HEAD_WARPS = 4, GQA_PAD = 8;
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
	const int *__restrict__ pos_offset, const int *__restrict__ block_table, int total
) {
#if __CUDA_ARCH__ >= 800
	constexpr int QLD = 64 + GQA_PAD, KLD = K_TILE + GQA_PAD, VLD = 64 + GQA_PAD;
	constexpr int WARPS = GQA_HEAD_WARPS * Q_TILES, Q_BLOCK = GQA_Q_TILE * Q_TILES;
	int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
	constexpr int HEAD_SPLITS = GQA_GROUP / GQA_HEAD_WARPS;
	int q_block = SWAP_GRID ? blockIdx.y : blockIdx.x;
	int head_block = SWAP_GRID ? blockIdx.x : blockIdx.y;
	int q_base = q_block * Q_BLOCK, q0 = q_base + (warp % Q_TILES) * GQA_Q_TILE;
	int g = head_block / HEAD_SPLITS;
	int h = g * GQA_GROUP + (head_block % HEAD_SPLITS) * GQA_HEAD_WARPS + (warp / Q_TILES);
	int pos = pos_offset[0], q_last = min(q_base + Q_BLOCK - 1, total - 1);
	int key_limit = pos + q_last + 1;
	extern __shared__ char smem[];
	auto *sq = reinterpret_cast<__nv_bfloat16*>(smem);

	for (int e = tid; e < WARPS * GQA_Q_TILE * 8; e += blockDim.x) {
		int w = e >> 7, x = e & 127, r = x >> 3, d = (x & 7) * 8;
		int qt = q_base + (w % Q_TILES) * GQA_Q_TILE + r;
		int qh = g * GQA_GROUP + (head_block % HEAD_SPLITS) * GQA_HEAD_WARPS + (w / Q_TILES);
		uint4 qv = {};
		if (qt < total) qv = *reinterpret_cast<const uint4*>(q + ((size_t)qt * 32 + qh) * 64 + d);
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
	float scale = 0.125f;
	const int *table = block_table;
	constexpr int kv_stride = 4 * 64;

	for (int key0 = 0; key0 < key_limit; key0 += K_TILE) {
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
				size_t off = ((size_t)phys * KV_BLOCK_SIZE + row) * kv_stride + g * 64 + d;
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
		int qrow0 = q0 + rg, qrow1 = qrow0 + 8;
		float tile_m0 = -INFINITY, tile_m1 = -INFINITY;
		#pragma unroll
		for (int ni = 0; ni < K_TILE / 8; ++ni) {
			int c = ni * 8 + cg, key = key0 + c;
			score_acc[ni][0] = qrow0 < total && key <= pos + qrow0 ? score_acc[ni][0] * scale : -INFINITY;
			score_acc[ni][1] = qrow0 < total && key + 1 <= pos + qrow0 ? score_acc[ni][1] * scale : -INFINITY;
			score_acc[ni][2] = qrow1 < total && key <= pos + qrow1 ? score_acc[ni][2] * scale : -INFINITY;
			score_acc[ni][3] = qrow1 < total && key + 1 <= pos + qrow1 ? score_acc[ni][3] * scale : -INFINITY;
			tile_m0 = fmaxf(tile_m0, fmaxf(score_acc[ni][0], score_acc[ni][1]));
			tile_m1 = fmaxf(tile_m1, fmaxf(score_acc[ni][2], score_acc[ni][3]));
		}
		tile_m0 = gqa_group_max(tile_m0);
		tile_m1 = gqa_group_max(tile_m1);
		float new_m0 = qrow0 < total ? fmaxf(row_m0, tile_m0) : 0.0f;
		float new_m1 = qrow1 < total ? fmaxf(row_m1, tile_m1) : 0.0f;
		float alpha0 = qrow0 < total && row_m0 != -INFINITY ? __expf(row_m0 - new_m0) : 0.0f;
		float alpha1 = qrow1 < total && row_m1 != -INFINITY ? __expf(row_m1 - new_m1) : 0.0f;
		float tile_l0 = 0.0f, tile_l1 = 0.0f;
		#pragma unroll
		for (int ni = 0; ni < K_TILE / 8; ++ni) {
			score_acc[ni][0] = qrow0 < total ? __expf(score_acc[ni][0] - new_m0) : 0.0f;
			score_acc[ni][1] = qrow0 < total ? __expf(score_acc[ni][1] - new_m0) : 0.0f;
			score_acc[ni][2] = qrow1 < total ? __expf(score_acc[ni][2] - new_m1) : 0.0f;
			score_acc[ni][3] = qrow1 < total ? __expf(score_acc[ni][3] - new_m1) : 0.0f;
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
		int qr = q0 + r;
		if (qr < total) {
			size_t off = ((size_t)qr * 32 + h) * 64 + lane * 2;
			*reinterpret_cast<uint32_t*>(out + off) =
				*reinterpret_cast<uint32_t*>(&sq[(warp * GQA_Q_TILE + r) * QLD + lane * 2]);
		}
	}
#endif
}

template <typename T>
// 根据 total 选择 Q_TILE/K_TILE 配置并发射 BF16 GQA 专用核。
void launch_gq_attention_prefill(
	T 		*__restrict__ out, 		// [total, NH * HS]
	const T 	*__restrict__ q,		// [total, NH * HS]
	const T 	*__restrict__ k_pool, 		// [num_blocks, KV_BLOCK_SIZE, NKV*HS]
	const T 	*__restrict__ v_pool,		// [num_blocks, KV_BLOCK_SIZE, NKV*HS]
	const int	*__restrict__ cu_seqlens,	// [B + 1]
	const int	*__restrict__ seq_ids,		// [total]
	const int	*__restrict__ pos_offset,	// [B]		// [64, 128, 512]每个位置表明这是prefill里的第几个chunk
	const int 	*__restrict__ block_table,	// [B, max_blocks_per_seq]
	int B, int NH, int NKV, int HS, int max_blocks_per_seq, int total, cudaStream_t t
) {
	if constexpr (std::is_same<T, __nv_bfloat16>::value) {
		if (B == 1 && NH == 32 && NKV == 4 && HS == 64) {
			static bool configured = [] {
				cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<32, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, GQA_SMEM_Q1_K32);
				cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<32, 1, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
				cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<64, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, GQA_SMEM_Q1_K64);
				cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<64, 1, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
				cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<32, 2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, GQA_SMEM_Q2_K32);
				cudaFuncSetAttribute(gq_attention_prefill_bf16_h64<32, 2, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
				return true;
			}();
			(void)configured;
			int grid_y = NKV * (GQA_GROUP / GQA_HEAD_WARPS);
			if (total <= 128) {
				dim3 grid((total + GQA_Q_TILE * 2 - 1) / (GQA_Q_TILE * 2), grid_y);
				gq_attention_prefill_bf16_h64<32, 2, false><<<grid, GQA_HEAD_WARPS * 2 * 32, GQA_SMEM_Q2_K32, t>>>(
					out, q, k_pool, v_pool, pos_offset, block_table, total);
			} else if (total <= 256) {
				dim3 grid((total + GQA_Q_TILE - 1) / GQA_Q_TILE, grid_y);
				gq_attention_prefill_bf16_h64<32, 1, false><<<grid, GQA_HEAD_WARPS * 32, GQA_SMEM_Q1_K32, t>>>(
					out, q, k_pool, v_pool, pos_offset, block_table, total);
			} else {
				dim3 grid(grid_y, (total + GQA_Q_TILE - 1) / GQA_Q_TILE);
				gq_attention_prefill_bf16_h64<64, 1, true><<<grid, GQA_HEAD_WARPS * 32, GQA_SMEM_Q1_K64, t>>>(
					out, q, k_pool, v_pool, pos_offset, block_table, total);
			}
			return;
		}
	}
}
