#pragma once
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include "Config.h"
#include "GemmLoader.h"
#include "MmaPtx.h"

template<int BM_, int BN_, int BK_, int WM_, int WN_>
struct QKVGemmConfig {
	static constexpr int BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_;
	static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
	static constexpr int THREADS = 32 * WARPS_M * WARPS_N;
	static constexpr int VEC = sizeof(float4) / sizeof(__nv_bfloat16);
	static constexpr int A_F4 = BM * BK / VEC;
	static constexpr int B_F4 = BK * BN / VEC;
	static constexpr int A_LPT = (A_F4 + THREADS - 1) / THREADS;
	static constexpr int B_LPT = (B_F4 + THREADS - 1) / THREADS;
	static constexpr int PAD = 8;
	static constexpr int SMEM_BYTES = sizeof(__nv_bfloat16) * 2 * BM * (BK + PAD) +
		sizeof(__nv_bfloat16) * 2 * BK * (BN + PAD);
	using In = __nv_bfloat16;
};

// Q/K 在融合权重中按 [前半维, 后半维] 成对排列，因此相邻两个 accumulator 可以直接完成 RoPE。
__device__ __forceinline__ void store_qkv_pair_bf16(
	float x0, float x1, int row, int packed_col,
	__nv_bfloat16 *q, __nv_bfloat16 *k_pool, __nv_bfloat16 *v_pool,
	const __nv_bfloat16 *bias, const int *positions, const int *kv_dst,
	const float *cos_table, const float *sin_table,
	int QN, int KN, int HS
) {
	if (bias) {
		x0 += __bfloat162float(bias[packed_col]);
		x1 += __bfloat162float(bias[packed_col + 1]);
	}
	__nv_bfloat16 raw0 = __float2bfloat16_rn(x0);
	__nv_bfloat16 raw1 = __float2bfloat16_rn(x1);
	if (packed_col >= QN + KN) {
		int d = packed_col - QN - KN;
		__nv_bfloat16 *target = v_pool + (size_t)kv_dst[row] * KN;
		target[d] = raw0;
		target[d + 1] = raw1;
		return;
	}

	int local = packed_col < QN ? packed_col : packed_col - QN;
	int head = local / HS, pair = (local % HS) / 2, half = HS / 2;
	float c = cos_table[(size_t)positions[row] * half + pair];
	float s = sin_table[(size_t)positions[row] * half + pair];
	float a = __bfloat162float(raw0), b = __bfloat162float(raw1);
	__nv_bfloat16 y0 = __float2bfloat16_rn(a * c - b * s);
	__nv_bfloat16 y1 = __float2bfloat16_rn(a * s + b * c);
	int logical0 = head * HS + pair, logical1 = logical0 + half;
	if (packed_col < QN) {
		__nv_bfloat16 *target = q + (size_t)row * QN;
		target[logical0] = y0;
		target[logical1] = y1;
	} else {
		__nv_bfloat16 *target = k_pool + (size_t)kv_dst[row] * KN;
		target[logical0] = y0;
		target[logical1] = y1;
	}
}

// Tensor Core 主核：一次读取 X 和融合权重，完成投影、RoPE 及分页 KV 写入。
template<typename Config>
__global__ void qkv_gemm_mma_bf16(
	const __nv_bfloat16 *__restrict__ A,
	const __nv_bfloat16 *__restrict__ B,
	const __nv_bfloat16 *__restrict__ bias,
	__nv_bfloat16 *__restrict__ q,
	__nv_bfloat16 *__restrict__ k_pool,
	__nv_bfloat16 *__restrict__ v_pool,
	const int *__restrict__ positions,
	const int *__restrict__ kv_dst,
	const float *__restrict__ cos_table,
	const float *__restrict__ sin_table,
	int M, int N, int K, int QN, int KN, int HS
) {
#if __CUDA_ARCH__ >= 800
	constexpr int BM = Config::BM, BN = Config::BN, BK = Config::BK;
	constexpr int WM = Config::WM, WN = Config::WN, PAD = Config::PAD;
	constexpr int MI = WM / 16, NI = WN / 8, NB = WN / 16;
	int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;
	int warp_row = warp_id / Config::WARPS_N, warp_col = warp_id % Config::WARPS_N;
	int wm0 = warp_row * WM, wn0 = warp_col * WN;
	int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;

	extern __shared__ char smem_buf[];
	auto &sa = *reinterpret_cast<__nv_bfloat16(*)[2][BM][BK + PAD]>(smem_buf);
	auto &sb = *reinterpret_cast<__nv_bfloat16(*)[2][BK][BN + PAD]>(
		smem_buf + sizeof(__nv_bfloat16) * 2 * BM * (BK + PAD));

	float acc[MI][NI][4];
	#pragma unroll
	for (int mi = 0; mi < MI; ++mi)
		#pragma unroll
		for (int ni = 0; ni < NI; ++ni)
			#pragma unroll
			for (int t = 0; t < 4; ++t) acc[mi][ni][t] = 0.0f;

	TileLoader<Config> loader{A, B, tid, block_row, block_col, M, N, K};
	loader.gmem_to_smem(0, 0, sa, sb);
	__pipeline_wait_prior(0);
	__syncthreads();

	int row16 = lane % 16, colh = (lane / 16) * 8, cur = 0;
	for (int k0 = 0; k0 < K; k0 += BK) {
		int next_k = k0 + BK;
		if (next_k < K) loader.gmem_to_smem(next_k, cur ^ 1, sa, sb);
		#pragma unroll
		for (int kk = 0; kk < BK; kk += 16) {
			uint32_t ar[MI][4];
			#pragma unroll
			for (int mi = 0; mi < MI; ++mi)
				ldmatrix_x4(ar[mi], smem_addr(&sa[cur][wm0 + mi * 16 + row16][kk + colh]));
			uint32_t br[NI][2];
			#pragma unroll
			for (int nb = 0; nb < NB; ++nb) {
				uint32_t r[4];
				ldmatrix_x4_trans(r, smem_addr(&sb[cur][kk + row16][wn0 + nb * 16 + colh]));
				br[nb * 2][0] = r[0]; br[nb * 2][1] = r[1];
				br[nb * 2 + 1][0] = r[2]; br[nb * 2 + 1][1] = r[3];
			}
			#pragma unroll
			for (int mi = 0; mi < MI; ++mi)
				#pragma unroll
				for (int ni = 0; ni < NI; ++ni) mma_m16n8k16<true>(acc[mi][ni], ar[mi], br[ni]);
		}
		if (next_k < K) {
			__pipeline_wait_prior(0);
			__syncthreads();
			cur ^= 1;
		}
	}

	int group = lane / 4, thread_in_group = lane % 4;
	#pragma unroll
	for (int mi = 0; mi < MI; ++mi) {
		#pragma unroll
		for (int ni = 0; ni < NI; ++ni) {
			int base_row = block_row + wm0 + mi * 16;
			int base_col = block_col + wn0 + ni * 8;
			int rows[2] = {base_row + group, base_row + group + 8};
			int col = base_col + thread_in_group * 2;
			#pragma unroll
			for (int p = 0; p < 2; ++p)
				if (rows[p] < M && col + 1 < N)
					store_qkv_pair_bf16(acc[mi][ni][p * 2], acc[mi][ni][p * 2 + 1],
						rows[p], col, q, k_pool, v_pool, bias, positions, kv_dst,
						cos_table, sin_table, QN, KN, HS);
		}
	}
#else
	__trap();
#endif
}

template<typename Config>
inline void launch_qkv_gemm_bf16_config(
	const __nv_bfloat16 *A, const __nv_bfloat16 *B, const __nv_bfloat16 *bias,
	__nv_bfloat16 *q, __nv_bfloat16 *k_pool, __nv_bfloat16 *v_pool,
	const int *positions, const int *kv_dst, const float *cos_table, const float *sin_table,
	int M, int K, int NH, int NKV, int HS, cudaStream_t stream
) {
	int QN = NH * HS, KN = NKV * HS, N = QN + 2 * KN;
	dim3 block(Config::THREADS);
	dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
	cudaFuncSetAttribute(qkv_gemm_mma_bf16<Config>, cudaFuncAttributeMaxDynamicSharedMemorySize, Config::SMEM_BYTES);
	qkv_gemm_mma_bf16<Config><<<grid, block, Config::SMEM_BYTES, stream>>>(
		A, B, bias, q, k_pool, v_pool, positions, kv_dst, cos_table, sin_table,
		M, N, K, QN, KN, HS);
}

// 一次完成 QKV 投影、Q/K RoPE 和 K/V 分页落池。
inline void launch_qkv_gemm_bf16(
	const __nv_bfloat16 *A, const __nv_bfloat16 *B, const __nv_bfloat16 *bias,
	__nv_bfloat16 *q, __nv_bfloat16 *k_pool, __nv_bfloat16 *v_pool,
	const int *positions, const int *kv_dst, const float *cos_table, const float *sin_table,
	int M, int K, int NH, int NKV, int HS, cudaStream_t stream
) {
	int N = (NH + 2 * NKV) * HS;
	if (M <= 256 && N <= 4096) {
		using Config = QKVGemmConfig<64, 64, 32, 32, 32>;
		launch_qkv_gemm_bf16_config<Config>(A, B, bias, q, k_pool, v_pool, positions, kv_dst,
			cos_table, sin_table, M, K, NH, NKV, HS, stream);
	} else if (M <= 256) {
		using Config = QKVGemmConfig<64, 128, 32, 32, 32>;
		launch_qkv_gemm_bf16_config<Config>(A, B, bias, q, k_pool, v_pool, positions, kv_dst,
			cos_table, sin_table, M, K, NH, NKV, HS, stream);
	} else if (M < 416) {
		using Config = QKVGemmConfig<128, 128, 32, 64, 64>;
		launch_qkv_gemm_bf16_config<Config>(A, B, bias, q, k_pool, v_pool, positions, kv_dst,
			cos_table, sin_table, M, K, NH, NKV, HS, stream);
	} else {
		using Config = QKVGemmConfig<64, 128, 16, 64, 32>;
		launch_qkv_gemm_bf16_config<Config>(A, B, bias, q, k_pool, v_pool, positions, kv_dst,
			cos_table, sin_table, M, K, NH, NKV, HS, stream);
	}
}
