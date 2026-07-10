#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cuda_bf16.h>
#include "../reduce/Reduce.cuh"

template<typename T>
__global__ void cross_entropy(
		float			*__restrict__		losses,		// [B, T]
		T			*__restrict__		probs,		// [B, T, V]
		const T			*__restrict__		logits,		// [B, T, V]
		const int 		*__restrict__		targets,	// [B, T]
		int V
) {
		int bt = blockIdx.x;
		const T *logits_bt = logits + (size_t)bt * V;
		float local_max = -INFINITY;
		for (int v = threadIdx.x; v < V; v += blockDim.x) {
				local_max = device_max(local_max, logits_bt[v]);
		}
		float row_max = block_max(local_max);
		float local_sum = 0.0f;
		for (int v = threadIdx.x; v < V; v += blockDim.x) {
				float tmp = expf(static_cast<float>(logits_bt[v]) - row_max);
				local_sum += tmp;
		}
		float Z = block_sum(local_sum); float inv_z = 1.0f / Z;

		T *probs_bt = probs + (size_t)bt * V;
		for (int v = threadIdx.x; v < V; v += blockDim.x) {
				float tmp = expf(static_cast<float>(logits_bt[v]) - row_max) * inv_z;
				probs_bt[v] = static_cast<T>(tmp);
		}

		if (threadIdx.x == 0) {
				int tgt = targets[bt];
				float logit_tgt = static_cast<float>(logits_bt[tgt]);
				losses[bt] = row_max + logf(Z) - logit_tgt;
		}
}

template<typename T>
void launch_crossentropy_forward(
		float			*__restrict__		losses,		// [B, T]
		T			*__restrict__		probs,		// [B, T, V]
		const T			*__restrict__		logits,		// [B, T, V]
		const int		*__restrict__		targets,	// [B, T]
		int B, int seq_len, int V, cudaStream_t t
) {
		int grid = B * seq_len;
		int block = 256;
		cross_entropy<T><<<grid, block, 0, t>>>(losses, probs, logits, targets, V);
}
