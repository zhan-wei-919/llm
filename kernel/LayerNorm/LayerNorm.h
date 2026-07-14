#pragma once
#include <cassert>
#include "../reduce/Reduce.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void layernorm(T *__restrict__ out,          // [B, T, C]
                          float *__restrict__ mean_out, // [B, T]
                          float *__restrict__ rstd_out, // [B, T]
                          const T *__restrict__ x,      // [B, T, C]
                          const T *__restrict__ gamma,  // [C]
                          const T *__restrict__ beta,   // [C]
                          int C, float eps) {
	int bt = blockIdx.x;
	float local_sum = 0.0f;
	float local_sum2 = 0.0f;

	constexpr int VEC = sizeof(float4) / sizeof(T);
	for (int i = threadIdx.x * VEC; i + VEC <= C; i += blockDim.x * VEC) {
		float4 xc_v = *reinterpret_cast<const float4 *>(&x[bt * C + i]);
		T *elem = reinterpret_cast<T *>(&xc_v);
		for (int j = 0; j < VEC; ++j) {
			local_sum += static_cast<float>(elem[j]);
			local_sum2 += static_cast<float>(elem[j]) *
			              static_cast<float>(elem[j]);
		}
	}
	for (int c = C / VEC * VEC + threadIdx.x; c < C; c += blockDim.x) {
		float xc = static_cast<float>(x[bt * C + c]);
		local_sum += xc;
		local_sum2 += xc * xc;
	}

	float sum_x = block_sum(local_sum);
	float sum_x2 = block_sum(local_sum2);

	float mean = sum_x / C;
	float var = sum_x2 / C - mean * mean;
	float rstd = rsqrtf(var + eps);

	if (threadIdx.x == 0) {
		mean_out[bt] = mean;
		rstd_out[bt] = rstd;
	}

	for (int c = threadIdx.x * VEC; c + VEC <= C; c += blockDim.x * VEC) {
		float4 xc_v = *reinterpret_cast<const float4 *>(&x[bt * C + c]);
		float4 gamma_v = *reinterpret_cast<const float4 *>(&gamma[c]);
		float4 beta_v = *reinterpret_cast<const float4 *>(&beta[c]);
		T *x_elems = reinterpret_cast<T *>(&xc_v);
		T *g_elems = reinterpret_cast<T *>(&gamma_v);
		T *b_elems = reinterpret_cast<T *>(&beta_v);
		float4 out_v;
		T *o_elems = reinterpret_cast<T *>(&out_v);
		for (int j = 0; j < VEC; ++j) {
			float norm =
			    (static_cast<float>(x_elems[j]) - mean) * rstd;
			float g = static_cast<float>(g_elems[j]);
			float b = static_cast<float>(b_elems[j]);
			o_elems[j] = static_cast<T>(norm * g + b);
		}
		*reinterpret_cast<float4 *>(&out[bt * C + c]) = out_v;
	}

	for (int c = C / VEC * VEC + threadIdx.x; c < C; c += blockDim.x) {
		float xc = static_cast<float>(x[bt * C + c]);
		float norm = (xc - mean) * rstd;
		float g = static_cast<float>(gamma[c]);
		float b = static_cast<float>(beta[c]);
		out[bt * C + c] = static_cast<T>(norm * g + b);
	}
}

template <typename T>
void launch_LayerNorm_forward(T *__restrict__ out,          // [B, T, C]
                              float *__restrict__ mean_out, // [B, T]
                              float *__restrict__ rstd_out, // [B, T]
                              const T *__restrict__ x,      // [B, T, C]
                              const T *__restrict__ gamma,  // [C]
                              const T *__restrict__ beta,   // [C]
                              int B, int seq_len, int C, float eps, cudaStream_t t) {
	int block = 256;
	int grid = B * seq_len;
	layernorm<<<grid, block, 0, t>>>(out, mean_out, rstd_out, x, gamma, beta, C,
	                           eps);
}
