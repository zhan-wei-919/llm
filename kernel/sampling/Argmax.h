#pragma once
#include <cuda_runtime.h>
#include <climits>
#include "../../core/Dtype.h"

template<typename T>
__global__ void argmax_last_token(
	int *tokens,
	const T *logits,
	const int *cu_seqlens,
	int vocab_size) {
	constexpr int BLOCK = 256;
	__shared__ float values[BLOCK];
	__shared__ int indices[BLOCK];

	int row = cu_seqlens[blockIdx.x + 1] - 1;
	const T *row_logits = logits + static_cast<size_t>(row) * vocab_size;
	float best_value = -__int_as_float(0x7f800000);
	int best_index = INT_MAX;
	for (int token = threadIdx.x; token < vocab_size; token += BLOCK) {
		float value = static_cast<float>(row_logits[token]);
		if (value > best_value || (value == best_value && token < best_index)) {
			best_value = value;
			best_index = token;
		}
	}
	values[threadIdx.x] = best_value;
	indices[threadIdx.x] = best_index;
	__syncthreads();

	for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			float other_value = values[threadIdx.x + stride];
			int other_index = indices[threadIdx.x + stride];
			if (other_value > values[threadIdx.x]
			 || (other_value == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
				values[threadIdx.x] = other_value;
				indices[threadIdx.x] = other_index;
			}
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) tokens[blockIdx.x] = indices[0];
}

inline void launch_argmax_last_token(
	int *tokens,
	const void *logits,
	Dtype dtype,
	const int *cu_seqlens,
	int batch,
	int vocab_size,
	cudaStream_t stream) {
	dtype_dispatch(dtype, [&](auto tag) {
		using T = typename decltype(tag)::type;
		argmax_last_token<<<batch, 256, 0, stream>>>(
			tokens,
			static_cast<const T *>(logits),
			cu_seqlens,
			vocab_size);
	});
}
