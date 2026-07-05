#pragma once
#include <cstdio>

struct GemmF32Config {
	static constexpr int BM = 128, BN = 128, BK = 16;
	static constexpr int TM = 8, TN = 8;
	static constexpr int VEC = 4;  // float4 = 4 floats
	static constexpr int T = (BM * BN) / (TM * TN);           // 256 threads
	static constexpr int C_COLS = BN / TN;                     // 16
	static constexpr int A_TPR = BK / VEC;                     // float4 per row of A tile
	static constexpr int A_VEC_ITEMS = BM * BK / VEC;
	static constexpr int A_LPT = (A_VEC_ITEMS + T - 1) / T;
	static constexpr int B_TPR = BN / VEC;
	static constexpr int B_VEC_ITEMS = BK * BN / VEC;
	static constexpr int B_LPT = (B_VEC_ITEMS + T - 1) / T;
};

template<typename Config>
__global__ void Gemm_f32(
		const float	*__restrict__ A,
		const float	*__restrict__ B,
		float		*__restrict__ C,
		const float	*__restrict__ bias,
		const float	alpha, const float beta,
		int M, int N, int K
) {
		constexpr int BM = Config::BM, BN = Config::BN, BK = Config::BK;
		constexpr int TM = Config::TM, TN = Config::TN, VEC = Config::VEC;
		constexpr int T = Config::T, C_COLS = Config::C_COLS;
		constexpr int A_TPR = Config::A_TPR, A_VEC_ITEMS = Config::A_VEC_ITEMS, A_LPT = Config::A_LPT;
		constexpr int B_TPR = Config::B_TPR, B_VEC_ITEMS = Config::B_VEC_ITEMS, B_LPT = Config::B_LPT;

		const int tid = threadIdx.x;
		const int tile_row = blockIdx.y * BM, tile_col = blockIdx.x * BN;
		const int t_row = tid / C_COLS * TM, t_col = tid % C_COLS * TN;
		__shared__ float shared_a[2][BK][BM], shared_b[2][BK][BN];

		auto gmem_to_reg = [&](int k0, float4 ra[A_LPT], float4 rb[B_LPT]) {
				for (int p = 0; p < A_LPT; ++p) {
						int f4_idx = tid + p * T;
						if (f4_idx < A_VEC_ITEMS) {
								int row = f4_idx / A_TPR, col = f4_idx % A_TPR * VEC;
								int idx = (tile_row + row) * K + k0 + col;
								ra[p] = *reinterpret_cast<const float4*>(&A[idx]);
						}
				}
				for (int p = 0; p < B_LPT; ++p) {
						int f4_idx = tid + p * T;
						if (f4_idx < B_VEC_ITEMS) {
								int row = f4_idx / B_TPR, col = f4_idx % B_TPR * VEC;
								int idx = (k0 + row) * N + tile_col + col;
								rb[p] = *reinterpret_cast<const float4*>(&B[idx]);
						}
				}
		};

		// shared_a 转置存储: shared_a[k][m]，方便后续按列读取
		auto reg_to_smem = [&](int stage, float4 ra[A_LPT], float4 rb[B_LPT]) {
				for (int p = 0; p < A_LPT; ++p) {
						int f4_idx = tid + p * T;
						if (f4_idx < A_VEC_ITEMS) {
								int row = f4_idx / A_TPR, col = f4_idx % A_TPR * VEC;
								shared_a[stage][col    ][row] = ra[p].x;
								shared_a[stage][col + 1][row] = ra[p].y;
								shared_a[stage][col + 2][row] = ra[p].z;
								shared_a[stage][col + 3][row] = ra[p].w;
						}
				}
				for (int p = 0; p < B_LPT; ++p) {
						int f4_idx = tid + p * T;
						if (f4_idx < B_VEC_ITEMS) {
								int row = f4_idx / B_TPR, col = f4_idx % B_TPR * VEC;
								*reinterpret_cast<float4*>(&shared_b[stage][row][col]) = rb[p];
						}
				}
		};

		int cur = 0;
		float4 ra[A_LPT], rb[B_LPT];
		float reg_a[TM], reg_b[TN];
		float acc[TM][TN] = {};
		gmem_to_reg(0, ra, rb);
		reg_to_smem(0, ra, rb);
		__syncthreads();

		for (int i = 0; i < K; i += BK) {
				int next_i = i + BK;
				float4 next_a[A_LPT], next_b[B_LPT];
				if (next_i < K) gmem_to_reg(next_i, next_a, next_b);

				for (int k = 0; k < BK; ++k) {
						for (int m = 0; m < TM; m += 4)
								*reinterpret_cast<float4*>(&reg_a[m]) = *reinterpret_cast<const float4*>(&shared_a[cur][k][t_row + m]);
						for (int n = 0; n < TN; n += 4)
								*reinterpret_cast<float4*>(&reg_b[n]) = *reinterpret_cast<const float4*>(&shared_b[cur][k][t_col + n]);
						for (int m = 0; m < TM; ++m)
								for (int n = 0; n < TN; ++n)
										acc[m][n] += reg_a[m] * reg_b[n];
				}

				if (next_i < K) {
						reg_to_smem(cur ^ 1, next_a, next_b);
						__syncthreads();
						cur ^= 1;
				}
		}

		for (int m = 0; m < TM; ++m) {
				int row = tile_row + t_row + m;
				if (row >= M) continue;
				for (int n = 0; n < TN; n += 4) {
						int col = tile_col + t_col + n;
						if (col + 3 >= N) continue;
						int idx = row * N + col;

						float4 result;
						result.x = alpha * acc[m][n    ];
						result.y = alpha * acc[m][n + 1];
						result.z = alpha * acc[m][n + 2];
						result.w = alpha * acc[m][n + 3];

						if (beta != 0.0f) {
								float4 old_c = *reinterpret_cast<const float4*>(&C[idx]);
								result.x += beta * old_c.x;
								result.y += beta * old_c.y;
								result.z += beta * old_c.z;
								result.w += beta * old_c.w;
						}

						if (bias) {
								float4 b4 = *reinterpret_cast<const float4*>(&bias[col]);
								result.x += b4.x;
								result.y += b4.y;
								result.z += b4.z;
								result.w += b4.w;
						}

						*reinterpret_cast<float4*>(&C[idx]) = result;
				}
		}
}

void launch_gemm_f32_forward(
		const float	*__restrict__ A,
		const float	*__restrict__ B,
		float		*__restrict__ C,
		const float	*__restrict__ bias,
		float alpha, float beta,
		int M, int N, int K,
		cudaStream_t stream = nullptr
) {
		using Config = GemmF32Config;
		dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
		Gemm_f32<Config><<<grid, Config::T, 0, stream>>>(A, B, C, bias, alpha, beta, M, N, K);
}

// gemm 的 extern 分发符号统一放在 kernels.cu(见该文件尾部 gemm_forward)
