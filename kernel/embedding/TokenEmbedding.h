#pragma once
#include <cuda_runtime.h>

// 一个 block 负责一个 token，把 embedding table 的对应行搬到输出。
template<typename T>
__global__ void token_embedding(
	T *__restrict__ out,                 // [total, hidden_dim]
	const int *__restrict__ token_ids,   // [total]
	const T *__restrict__ weight,        // [vocab_size, hidden_dim]
	int hidden_dim
) {
	int t = blockIdx.x;
	int token_id = token_ids[t];
	const T *src = weight + (size_t)token_id * hidden_dim;
	T *dst = out + (size_t)t * hidden_dim;
	for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x)
		dst[d] = src[d];
}

template<typename T>
void launch_token_embedding(
	T *out,
	const int *token_ids,
	const T *weight,
	int total,
	int hidden_dim,
	cudaStream_t stream
) {
	token_embedding<T><<<total, 256, 0, stream>>>(
		out, token_ids, weight, hidden_dim
	);
}
