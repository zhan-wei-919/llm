#pragma once
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// 一个block负责一个token
template<typename T>
__global__ void add_token_and_pos_embedding(
		T			*__restrict__		out,				// [B, T, C]
		const int	*__restrict__		token_ids,			// [B, T]
		const T		*__restrict__		token_table,		// [V, C]
		const T		*__restrict__		pos_table,			// [T, C]
		int B, int seq_len, int C
) {
		int bt = blockIdx.x;								// 找到自己负责的那个token
		int seq_pos = bt % seq_len;							// 找到这个token在句子里的位置
		int token_id = token_ids[bt];						// 找到那个token的id, 这个id对应的是token_table里的行号
		
		constexpr int VEC = sizeof(float4) / sizeof(T);
		for (int c = threadIdx.x * VEC; c + VEC <= C; c += blockDim.x * VEC) {
				float4 t4 = *reinterpret_cast<const float4*>(&token_table[token_id * C + c]);
				float4 p4 = *reinterpret_cast<const float4*>(&pos_table[seq_pos * C + c]);
				T *tv = reinterpret_cast<T*>(&t4);
				T *pv = reinterpret_cast<T*>(&p4);
				float4 o4; T *ov = reinterpret_cast<T*>(&o4);
				for (int i = 0; i < VEC; ++i) {
						ov[i] = static_cast<T>(static_cast<float>(tv[i]) + static_cast<float>(pv[i]));
				}
				*reinterpret_cast<float4*>(&out[bt * C + c]) = o4;
		}
		for (int c = C / VEC * VEC + threadIdx.x; c < C; c += blockDim.x) {
				float tok = static_cast<float>(token_table[token_id * C + c]);
				float pos = static_cast<float>(pos_table[seq_pos * C + c]);
				out[bt * C + c] = static_cast<T>(tok + pos);
		}
}

template<typename T>
void launch_embedding_forward(
		T			*__restrict__		out,				// [B, T, C]
		const int	*__restrict__		token_ids,			// [B, T]
		const T		*__restrict__		token_table,		// [V, C]
		const T		*__restrict__		pos_table,			// [T, C]
		int B, int seq_len, int C, cudaStream_t t
) {
		int grid = seq_len * B;
		int block = 256;
		add_token_and_pos_embedding<<<grid, block, 0, t>>>(out, token_ids, token_table, pos_table, B, seq_len, C);
}
