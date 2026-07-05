#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <type_traits>
#include "Config.h"
#include "GemmLoader.cuh"
#include "MmaPtx.cuh"

template<typename Config>
__global__ void Gemm (
		const typename Config::In	*__restrict__ A,
		const typename Config::In	*__restrict__ B,
		typename Config::Out		*__restrict__ C,
		const typename Config::Out	*__restrict__ bias,
		const float	alpha, const float beta,
		int M, int N, int K
) {
		using namespace nvcuda;
		int tid = threadIdx.x;
		int block_row = blockIdx.y * Config::BM, block_col = blockIdx.x * Config::BN;
		int warp_id = threadIdx.x / 32, warp_row = warp_id / Config::WARPS_N, warp_col = warp_id % Config::WARPS_N;
		int wm0 = warp_row * Config::WM, wn0 = warp_col * Config::WN;
		
		extern __shared__ char smem_buf[];
		auto &shared_a = *reinterpret_cast<typename Config::In(*)[2][Config::BM][Config::BK + Config::PAD]>(smem_buf);
		auto &shared_b = *reinterpret_cast<typename Config::In(*)[2][Config::BK][Config::BN + Config::PAD]>(
				smem_buf + sizeof(typename Config::In) * 2 * Config::BM * (Config::BK + Config::PAD));
		
		wmma::fragment<wmma::accumulator, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, typename Config::Acc> c_frag[Config::FM][Config::FN];
		for (int m = 0; m < Config::FM; ++m) {
				for (int n = 0; n < Config::FN; ++n) {
						wmma::fill_fragment(c_frag[m][n], 0);
				}
		}
		
		TileLoader<Config> loader {
				A, B, tid,
				block_row, block_col,
				M, N, K
		};
		
		int cur = 0;

#if __CUDA_ARCH__ >= 800
		loader.gmem_to_smem(0, 0, shared_a, shared_b);
		__pipeline_wait_prior(0);
		__syncthreads();

		for (int i = 0; i < K; i += Config::BK) {
				int next_i = i + Config::BK;
				if (next_i < K) loader.gmem_to_smem(next_i, cur ^ 1, shared_a, shared_b);

				wmma::fragment<wmma::matrix_a, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, typename Config::In, wmma::row_major> a_frag[Config::FM];
				wmma::fragment<wmma::matrix_b, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, typename Config::In, wmma::row_major> b_frag[Config::FN];

				for (int k = 0; k < Config::BK; k += Config::WMMA_K) {
						for (int m = 0; m < Config::FM; ++m) {
								wmma::load_matrix_sync(a_frag[m], &shared_a[cur][wm0 + m * Config::WMMA_M][k], Config::BK + Config::PAD);
						}
						for (int n = 0; n < Config::FN; ++n) {
								wmma::load_matrix_sync(b_frag[n], &shared_b[cur][k][wn0 + n * Config::WMMA_N], Config::BN + Config::PAD);
						}
						for (int m = 0; m < Config::FM; ++m) {
								for (int n = 0; n < Config::FN; ++n) {
										wmma::mma_sync(c_frag[m][n], a_frag[m], b_frag[n], c_frag[m][n]);
								}
						}
				}

				if (next_i < K) {
						__pipeline_wait_prior(0);
						__syncthreads();
						cur ^= 1;
				}
		}
#else
		float4 ra[Config::A_LPT], rb[Config::B_LPT];
		loader.gmem_to_reg(0, ra, rb);
		loader.reg_to_smem(0, ra, rb, shared_a, shared_b);
		__syncthreads();

		for (int i = 0; i < K; i += Config::BK) {
				int next_i = i + Config::BK;
				float4 next_a[Config::A_LPT], next_b[Config::B_LPT];
				if (next_i < K) loader.gmem_to_reg(next_i, next_a, next_b);

				wmma::fragment<wmma::matrix_a, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, typename Config::In, wmma::row_major> a_frag[Config::FM];
				wmma::fragment<wmma::matrix_b, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, typename Config::In, wmma::row_major> b_frag[Config::FN];

				for (int k = 0; k < Config::BK; k += Config::WMMA_K) {
						for (int m = 0; m < Config::FM; ++m) {
								wmma::load_matrix_sync(a_frag[m], &shared_a[cur][wm0 + m * Config::WMMA_M][k], Config::BK + Config::PAD);
						}
						for (int n = 0; n < Config::FN; ++n) {
								wmma::load_matrix_sync(b_frag[n], &shared_b[cur][k][wn0 + n * Config::WMMA_N], Config::BN + Config::PAD);
						}
						for (int m = 0; m < Config::FM; ++m) {
								for (int n = 0; n < Config::FN; ++n) {
										wmma::mma_sync(c_frag[m][n], a_frag[m], b_frag[n], c_frag[m][n]);
								}
						}
				}

				if (next_i < K) {
						__syncthreads();
						loader.reg_to_smem(cur ^ 1, next_a, next_b, shared_a, shared_b);
						__syncthreads();
						cur ^= 1;
				}
		}
#endif
		__syncthreads();
		
		auto *smem_base = reinterpret_cast<typename Config::Acc*>(&shared_a[0][0][0]);
		typename Config::Acc *warp_c = smem_base + warp_id * (Config::WMMA_M * Config::WMMA_N);
		
		int lane = tid &31;
		
		for (int m = 0; m < Config::FM; ++m) {
				for (int n = 0; n < Config::FN; ++n) {
						wmma::store_matrix_sync(warp_c, c_frag[m][n], Config::WMMA_N, wmma::mem_row_major);
						__syncwarp();
						int tile_row = block_row + wm0 + m * Config::WMMA_M;
						int tile_col = block_col + wn0 + n * Config::WMMA_N;
						
						for (int e = lane; e < Config::WMMA_M * Config::WMMA_N; e += 32) {
								int r = e / Config::WMMA_N, c = e % Config::WMMA_N;
								int gr = tile_row + r, gc = tile_col + c;
								if (gr < M && gc < N) {
										int idx = gr * N + gc;
										typename  Config::Acc v = alpha * warp_c[r * Config::WMMA_N + c];
										if (beta != 0) v += beta * static_cast<typename Config::Acc>(C[idx]);
										if (bias)	   v += static_cast<typename Config::Acc>(bias[gc]);
										C[idx] = static_cast<typename Config::Out>(v); 
								}
						}
						__syncwarp();
				}
		}	
}

template<typename Config>
__global__ void GemmMma (
		const typename Config::In	*__restrict__ A,
		const typename Config::In	*__restrict__ B,
		typename Config::Out		*__restrict__ C,
		const typename Config::Out	*__restrict__ bias,
		const float	alpha, const float beta,
		int M, int N, int K
) {
#if __CUDA_ARCH__ >= 800
		using In = typename Config::In;
		constexpr bool IsBF16 = std::is_same<In, __nv_bfloat16>::value;
		constexpr int BM = Config::BM, BN = Config::BN, BK = Config::BK;
		constexpr int WM = Config::WM, WN = Config::WN, PAD = Config::PAD;
		constexpr int MI = WM / 16;   // 每个 warp M 方向的 mma tile 数（每个 16 行）
		constexpr int NI = WN / 8;    // 每个 warp N 方向的 mma tile 数（每个 8 列）
		constexpr int NB = WN / 16;   // ldmatrix.x4.trans 每次取 16 列 = 2 个 n-tile

		int tid = threadIdx.x;
		int warp_id = tid / 32, lane = tid % 32;
		int warp_row = warp_id / Config::WARPS_N, warp_col = warp_id % Config::WARPS_N;
		int wm0 = warp_row * WM, wn0 = warp_col * WN;
		int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;

		extern __shared__ char smem_buf[];
		auto &sa = *reinterpret_cast<In(*)[2][BM][BK + PAD]>(smem_buf);
		auto &sb = *reinterpret_cast<In(*)[2][BK][BN + PAD]>(
				smem_buf + sizeof(In) * 2 * BM * (BK + PAD));

		float acc[MI][NI][4];
		#pragma unroll
		for (int mi = 0; mi < MI; ++mi)
				#pragma unroll
				for (int ni = 0; ni < NI; ++ni)
						#pragma unroll
						for (int t = 0; t < 4; ++t) acc[mi][ni][t] = 0.0f;

		TileLoader<Config> loader { A, B, tid, block_row, block_col, M, N, K };
		loader.gmem_to_smem(0, 0, sa, sb);
		__pipeline_wait_prior(0);
		__syncthreads();

		int row16 = lane % 16, colh = (lane / 16) * 8;
		int cur = 0;

		for (int i = 0; i < K; i += BK) {
				int next_i = i + BK;
				if (next_i < K) loader.gmem_to_smem(next_i, cur ^ 1, sa, sb);

				#pragma unroll
				for (int kk = 0; kk < BK; kk += 16) {
						uint32_t a[MI][4];
						#pragma unroll
						for (int mi = 0; mi < MI; ++mi)
								ldmatrix_x4(a[mi], smem_addr(&sa[cur][wm0 + mi * 16 + row16][kk + colh]));

						uint32_t b[NI][2];
						#pragma unroll
						for (int nb = 0; nb < NB; ++nb) {
								uint32_t r[4];
								ldmatrix_x4_trans(r, smem_addr(&sb[cur][kk + row16][wn0 + nb * 16 + colh]));
								b[nb * 2 + 0][0] = r[0]; b[nb * 2 + 0][1] = r[1];
								b[nb * 2 + 1][0] = r[2]; b[nb * 2 + 1][1] = r[3];
						}

						#pragma unroll
						for (int mi = 0; mi < MI; ++mi)
								#pragma unroll
								for (int ni = 0; ni < NI; ++ni)
										mma_m16n8k16<IsBF16>(acc[mi][ni], a[mi], b[ni]);
				}

				if (next_i < K) {
						__pipeline_wait_prior(0);
						__syncthreads();
						cur ^= 1;
				}
		}

		int groupID = lane / 4, tig = lane % 4;
		#pragma unroll
		for (int mi = 0; mi < MI; ++mi) {
				#pragma unroll
				for (int ni = 0; ni < NI; ++ni) {
						int base_row = block_row + wm0 + mi * 16;
						int base_col = block_col + wn0 + ni * 8;
						int rr[4] = { groupID, groupID, groupID + 8, groupID + 8 };
						int cc[4] = { tig * 2, tig * 2 + 1, tig * 2, tig * 2 + 1 };
						#pragma unroll
						for (int t = 0; t < 4; ++t) {
								int gr = base_row + rr[t], gc = base_col + cc[t];
								if (gr < M && gc < N) {
										int idx = gr * N + gc;
										float v = alpha * acc[mi][ni][t];
										if (beta != 0) v += beta * static_cast<float>(C[idx]);
										if (bias)     v += static_cast<float>(bias[gc]);
										C[idx] = static_cast<typename Config::Out>(v);
								}
						}
				}
		}
#else
		__trap();
#endif
}

template<typename Config>
void launch_Gemm_forward(
		const typename Config::In *A, 
		const typename Config::In *B,
		typename Config::Out *C,
		const typename Config::Out *bias,
		float alpha, float beta, int M, int N, int K, cudaStream_t s
) {
		dim3 block(Config::THREADS);
		dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
		if constexpr (sizeof(typename Config::In) == 2) {
				cudaFuncSetAttribute(
						GemmMma<Config>,
						cudaFuncAttributeMaxDynamicSharedMemorySize,
						Config::SMEM_BYTES
				);
				GemmMma<Config><<<grid, block, Config::SMEM_BYTES, s>>>(A, B, C, bias, alpha, beta, M, N, K);
		} else {
				cudaFuncSetAttribute(
						Gemm<Config>,
						cudaFuncAttributeMaxDynamicSharedMemorySize,
						Config::SMEM_BYTES
				);
				Gemm<Config><<<grid, block, Config::SMEM_BYTES, s>>>(A, B, C, bias, alpha, beta, M, N, K);
		}
}

// gemm 的 extern 分发符号统一放在 kernels.cu:
// 需要同时看到 f32 独立 launcher 与本文件的 tensor-core launcher,单个 TU 才凑得齐