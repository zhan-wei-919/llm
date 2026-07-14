#pragma once
#include "Config.h"
#include <cuda_pipeline.h>

template<typename Config>
struct TileLoader {
		using T = typename Config::In;

		const T *__restrict__ A;
		const T *__restrict__ B;
		int tid;
		int block_row, block_col;
		int M, N, K;
		static constexpr int THREADS = Config::THREADS;
		static constexpr int VEC = Config::VEC;
		static constexpr int A_LPT = Config::A_LPT;
		static constexpr int A_F4 = Config::A_F4;
		static constexpr int B_LPT = Config::B_LPT;
		static constexpr int B_F4 = Config::B_F4;
		static constexpr int BM = Config::BM;
		static constexpr int BN = Config::BN;
		static constexpr int BK = Config::BK;
		static constexpr int PAD = Config::PAD;

		__device__ inline void gmem_to_reg(int k0, float4 ra[A_LPT], float4 rb[B_LPT]) const {
				#pragma unroll
				for (int p = 0; p < A_LPT; ++p) {
						int f4_idx = tid + p * THREADS;
						if (f4_idx < A_F4) {
								int e_idx = f4_idx * VEC;
								int row = e_idx / BK, col = e_idx % BK;
								int idx = (block_row + row) * K + k0 + col;
								bool op_in = (block_row + row < M) && (k0 + col + VEC <= K);
								ra[p] = op_in ? *reinterpret_cast<const float4*>(&A[idx]) : float4{0,0,0,0};
						}
				}
				#pragma unroll
				for (int p = 0; p < B_LPT; ++p) {
						int f4_idx = tid + p * THREADS;
						if (f4_idx < B_F4) {
								int e_idx = f4_idx * VEC;
								int row = e_idx / BN, col = e_idx % BN;
								int idx = (k0 + row) * N + block_col + col;
								bool op_in = (k0 + row < K) && (block_col + col + VEC <= N);
								rb[p] = op_in ? *reinterpret_cast<const float4*>(&B[idx]) : float4{0,0,0,0};
						}
				}
		}

		__device__ inline void reg_to_smem(int stage, float4 ra[A_LPT], float4 rb[B_LPT],
											T (&shared_a)[2][BM][BK + PAD], T (&shared_b)[2][BK][BN + PAD]) const {
				#pragma unroll
				for (int p = 0; p < A_LPT; ++p) {
						int f4_idx = tid + p * THREADS;
						if (f4_idx < A_F4) {
								int e_idx = f4_idx * VEC;
								int row = e_idx / BK, col = e_idx % BK;
								*reinterpret_cast<float4*>(&shared_a[stage][row][col]) = ra[p];
						}
				}
				#pragma unroll
				for (int p = 0; p < B_LPT; ++p) {
						int f4_idx = tid + p * THREADS;
						if (f4_idx < B_F4) {
								int e_idx = f4_idx * VEC;
								int row = e_idx / BN, col = e_idx % BN;
								*reinterpret_cast<float4*>(&shared_b[stage][row][col]) = rb[p];
						}
				}
		}
		
		__device__ inline void gmem_to_smem(int k0, int stage, T (&shared_a)[2][BM][BK + PAD], T (&shared_b)[2][BK][BN + PAD]) {
				#pragma unroll
				for (int p = 0; p < A_LPT; ++p) {
						int f4_idx = tid + p * THREADS;
						if (f4_idx < A_F4) {
								int e_idx = f4_idx * VEC;
								int row = e_idx / BK, col = e_idx % BK;
								int idx = (block_row + row) * K + k0 + col;
								bool op_in = (block_row + row < M) && (k0 + col + VEC <= K);
								if (op_in) {
										__pipeline_memcpy_async(
												&shared_a[stage][row][col],
												&A[idx],
												sizeof(float4));
								} else {
										*reinterpret_cast<float4*>(&shared_a[stage][row][col]) = float4{0,0,0,0};
								}
						}
				}
				#pragma unroll
				for (int p = 0; p < B_LPT; ++p) {
						int f4_idx = tid + p * THREADS;
						if (f4_idx < B_F4) {
								int e_idx = f4_idx * VEC;
								int row = e_idx / BN, col = e_idx % BN;
								int idx = (k0 + row) * N + block_col + col;
								bool op_in = (k0 + row < K) && (block_col + col + VEC <= N);
								if (op_in) {
										__pipeline_memcpy_async(
												&shared_b[stage][row][col],
												&B[idx],
												sizeof(float4));
								} else {
										*reinterpret_cast<float4*>(&shared_b[stage][row][col]) = float4{0,0,0,0};
								}
						}
				}
				__pipeline_commit();
		}
};