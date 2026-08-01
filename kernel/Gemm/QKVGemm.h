#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <type_traits>
#include "Config.h"
#include "GemmLoader.h"
#include "MmaPtx.h"

template<typename T_, int BM_, int BN_, int BK_, int WM_, int WN_>
struct QKVGemmConfig {
	static constexpr int BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_;
	static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
	static constexpr int THREADS = 32 * WARPS_M * WARPS_N;
	static constexpr int VEC = sizeof(float4) / sizeof(T_);
	static constexpr int A_F4 = BM * BK / VEC;
	static constexpr int B_F4 = BK * BN / VEC;
	static constexpr int A_LPT = (A_F4 + THREADS - 1) / THREADS;
	static constexpr int B_LPT = (B_F4 + THREADS - 1) / THREADS;
	static constexpr int PAD = 8;
	static constexpr int SMEM_BYTES = sizeof(T_) * 2 * BM * (BK + PAD) +
		sizeof(T_) * 2 * BK * (BN + PAD);
	using In = T_;
};

// Q/K 在融合权重中按 [前半维, 后半维] 成对排列，因此相邻两个 accumulator 可以直接完成 RoPE。
template<typename T>
__device__ __forceinline__ void store_qkv_pair(
	float x0, float x1, int row, int packed_col,
	T *q, T *k_pool, T *v_pool, const T *bias, const int *positions, const int *kv_dst,
	const float *cos_table, const float *sin_table,
	int QN, int KN, int HS
) {
	if (bias) {
		x0 += static_cast<float>(bias[packed_col]);
		x1 += static_cast<float>(bias[packed_col + 1]);
	}
	T raw0 = static_cast<T>(x0), raw1 = static_cast<T>(x1);
	if (packed_col >= QN + KN) {
		int d = packed_col - QN - KN;
		T *target = v_pool + (size_t)kv_dst[row] * KN;
		target[d] = raw0;
		target[d + 1] = raw1;
		return;
	}

	int local = packed_col < QN ? packed_col : packed_col - QN;
	int head = local / HS, pair = (local % HS) / 2, half = HS / 2;
	float c = cos_table[(size_t)positions[row] * half + pair];
	float s = sin_table[(size_t)positions[row] * half + pair];
	float a = static_cast<float>(raw0), b = static_cast<float>(raw1);
	T y0 = static_cast<T>(a * c - b * s), y1 = static_cast<T>(a * s + b * c);
	int logical0 = head * HS + pair, logical1 = logical0 + half;
	if (packed_col < QN) {
		T *target = q + (size_t)row * QN;
		target[logical0] = y0;
		target[logical1] = y1;
	} else {
		T *target = k_pool + (size_t)kv_dst[row] * KN;
		target[logical0] = y0;
		target[logical1] = y1;
	}
}

// Tensor Core 主核：一次读取 X 和融合权重，完成投影、RoPE 及分页 KV 写入。
template<typename Config>
__global__ void qkv_gemm_mma(
	const typename Config::In *__restrict__ A,
	const typename Config::In *__restrict__ B,
	const typename Config::In *__restrict__ bias,
	typename Config::In *__restrict__ q,
	typename Config::In *__restrict__ k_pool,
	typename Config::In *__restrict__ v_pool,
	const int *__restrict__ positions,
	const int *__restrict__ kv_dst,
	const float *__restrict__ cos_table,
	const float *__restrict__ sin_table,
	int M, int N, int K, int QN, int KN, int HS
) {
#if __CUDA_ARCH__ >= 800
	using T = typename Config::In;
	constexpr int BM = Config::BM, BN = Config::BN, BK = Config::BK;
	constexpr int WM = Config::WM, WN = Config::WN, PAD = Config::PAD;
	constexpr int MI = WM / 16, NI = WN / 8, NB = WN / 16;
	int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;
	int warp_row = warp_id / Config::WARPS_N, warp_col = warp_id % Config::WARPS_N;
	int wm0 = warp_row * WM, wn0 = warp_col * WN;
	int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;

	extern __shared__ char smem_buf[];
	auto &sa = *reinterpret_cast<T(*)[2][BM][BK + PAD]>(smem_buf);
	auto &sb = *reinterpret_cast<T(*)[2][BK][BN + PAD]>(
		smem_buf + sizeof(T) * 2 * BM * (BK + PAD));

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
				for (int ni = 0; ni < NI; ++ni)
					mma_m16n8k16<std::is_same<T, __nv_bfloat16>::value>(acc[mi][ni], ar[mi], br[ni]);
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
					store_qkv_pair<T>(acc[mi][ni][p * 2], acc[mi][ni][p * 2 + 1],
						rows[p], col, q, k_pool, v_pool, bias, positions, kv_dst,
						cos_table, sin_table, QN, KN, HS);
		}
	}
#else
	__trap();
#endif
}

template<typename Config>
inline void launch_qkv_gemm_config(
	const typename Config::In *A, const typename Config::In *B, const typename Config::In *bias,
	typename Config::In *q, typename Config::In *k_pool, typename Config::In *v_pool,
	const int *positions, const int *kv_dst, const float *cos_table, const float *sin_table,
	int M, int K, int NH, int NKV, int HS, cudaStream_t stream
) {
	int QN = NH * HS, KN = NKV * HS, N = QN + 2 * KN;
	dim3 block(Config::THREADS);
	dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
	cudaFuncSetAttribute(qkv_gemm_mma<Config>, cudaFuncAttributeMaxDynamicSharedMemorySize, Config::SMEM_BYTES);
	qkv_gemm_mma<Config><<<grid, block, Config::SMEM_BYTES, stream>>>(
		A, B, bias, q, k_pool, v_pool, positions, kv_dst, cos_table, sin_table,
		M, N, K, QN, KN, HS);
}

// 一次完成 QKV 投影、Q/K RoPE 和 K/V 分页落池。
template<typename T>
inline void launch_qkv_gemm(
	const T *A, const T *B, const T *bias, T *q, T *k_pool, T *v_pool,
	const int *positions, const int *kv_dst, const float *cos_table, const float *sin_table,
	int M, int K, int NH, int NKV, int HS, cudaStream_t stream
) {
	static_assert(std::is_same<T, __nv_bfloat16>::value || std::is_same<T, half>::value,
	              "QKVGemm only supports F16/BF16");
	int N = (NH + 2 * NKV) * HS;
	if (M <= 256 && N <= 4096) {
		using Config = QKVGemmConfig<T, 64, 64, 32, 32, 32>;
		launch_qkv_gemm_config<Config>(A, B, bias, q, k_pool, v_pool, positions, kv_dst,
			cos_table, sin_table, M, K, NH, NKV, HS, stream);
	} else if (M <= 256) {
		using Config = QKVGemmConfig<T, 64, 128, 32, 32, 32>;
		launch_qkv_gemm_config<Config>(A, B, bias, q, k_pool, v_pool, positions, kv_dst,
			cos_table, sin_table, M, K, NH, NKV, HS, stream);
	} else if (M < 416) {
		using Config = QKVGemmConfig<T, 128, 128, 32, 64, 64>;
		launch_qkv_gemm_config<Config>(A, B, bias, q, k_pool, v_pool, positions, kv_dst,
			cos_table, sin_table, M, K, NH, NKV, HS, stream);
	} else {
		using Config = QKVGemmConfig<T, 64, 128, 16, 64, 32>;
		launch_qkv_gemm_config<Config>(A, B, bias, q, k_pool, v_pool, positions, kv_dst,
			cos_table, sin_table, M, K, NH, NKV, HS, stream);
	}
}
