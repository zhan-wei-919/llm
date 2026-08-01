#pragma once
#include <cuda_runtime.h>
#include "../kv/KV_pool.h"
#include "../tensor/Tensor.h"
#include "../tensor/Arena.h"
#include "../kernel/attention/Attention.h"
#include "../kernel/embedding/RoPECache.h"
#include "../kernel/Gemm/QKVGemm.h"
#include "GraphShape.h"

// Engine: 把"一步推理"翻译成固定的操作序列
//   记账 (调 pool) → 收集 (gather_tables) → 搬运 (memcpy 上传) → 发射 (kernel)
// 只借不占: pool 和 device 工作数组由装配层构造好递进来, 这里不做任何显存分配.
// QKV 落池和 attention 在同一个 stream 上按发射顺序执行, 先写后读天然成立,
// 所以全程不需要显式同步.
class Engine {
public:
	// S = max_seqs
	// W = max_blocks_per_seq
	// T = max_batched_tokens
	// table:      S * W       个 int
	// pos:        S           个 int
	// positions:  T           个 int
	// kv_dst:     T           个 int
	// cu_seqlens: S + 1       个 int
	Engine(Arena &arena, int NH, int NKV, int HS, int max_seqs, int max_seq_len, int max_batched_tokens)
	: arena_(arena), NH_(NH), NKV_(NKV), HS_(HS), max_tokens_(max_batched_tokens) {
		int S = max_seqs, W = (max_seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
		total_ints_ = S * W + S + 2 * max_tokens_ + S + 1;
		meta_ = arena_.alloc({total_ints_}, Dtype::I32);
		int L = W * KV_BLOCK_SIZE;
		cos_ = arena_.alloc({L, HS / 2}, Dtype::F32);
		sin_ = arena_.alloc({L, HS / 2}, Dtype::F32);
	}

	void bind_pool(KV_Pool *pool) {
		pool_ = pool; int *d_base = (int*)meta_->ptr;
		int S = pool_->max_seqs(), W = pool_->max_blocks_per_seq();
		d_table_	= d_base;	d_base += S * W;
		d_pos_		= d_base;	d_base += S;
		d_positions_	= d_base;	d_base += max_tokens_;
		d_kv_dst_	= d_base;	d_base += max_tokens_;
		d_cu_seqlens_	= d_base;
		cudaHostAlloc(&h_base_, 2 * total_ints_ * sizeof(int), 0);
		h_half_[0] = h_base_; h_half_[1] = h_base_ + total_ints_;
		select_half(0);
	}

	int alloc_seq() { return pool_->alloc_seq(); }
	void release(int slot) { pool_->release(slot); }

	GraphShape prepare(const int *slots, int B, const int *lens, cudaStream_t stream) {
		current_B_ = B;
		select_half(step_++ & 1);
		h_cu_seqlens_[0] = 0;
		for (int b = 0; b < B; ++b) {
			h_pos_[b] = pool_->append(slots[b], lens[b]);
			h_cu_seqlens_[b + 1] = h_cu_seqlens_[b] + lens[b];
		}
		current_total_ = h_cu_seqlens_[B];
		pool_->gather_tables(slots, B, h_table_);
		for (int b = 0, t = 0; b < B; ++b) {
			for (int i = 0; i < lens[b]; ++i, ++t) {
				int p = h_pos_[b] + i;
				h_positions_[t] = p;
				h_kv_dst_[t] = pool_->physical_token(slots[b], p);
			}
		}
		upload(stream);
		return {current_B_, current_total_};
	}

	// 融合完成 QKV 投影、Q/K RoPE，并把当前层 K/V 写入 prepare 预先计算的物理位置。
	template<typename T>
	void forward_qkv(int layer, const T *input, const T *weight, const T *bias, T *q, int K, cudaStream_t stream) {
		launch_qkv_gemm(input, weight, bias, q,
			static_cast<T *>(pool_->k_base(layer)),
			static_cast<T *>(pool_->v_base(layer)),
			d_positions_, d_kv_dst_, static_cast<const float *>(cos_->ptr),
			static_cast<const float *>(sin_->ptr),
			current_total_, K, NH_, NKV_, HS_, stream);
	}

	// 当前层 K/V 已经落池；T 在模型装配期确定，shape family 在 Graph 构建期选择。
	template<typename T>
	void forward_attention(int layer, const T *q, T *out, int window_size, cudaStream_t stream) {
		const T *k_base = static_cast<const T *>(pool_->k_base(layer));
		const T *v_base = static_cast<const T *>(pool_->v_base(layer));
		int W = pool_->max_blocks_per_seq();
		launch_attention(
			out, q, k_base, v_base, d_cu_seqlens_, d_pos_, d_table_,
			current_B_, NH_, NKV_, HS_, W, current_total_, window_size, stream);
	}

	void init_rope(cudaStream_t stream, float base = 10000.0f) {
		int total = pool_->max_blocks_per_seq() * KV_BLOCK_SIZE * (HS_ / 2);
		int block = 256;
		int grid = (total + block - 1) / block;
		float *cos_table = static_cast<float*>(cos_->ptr);
		float *sin_table = static_cast<float*>(sin_->ptr);
		init_rope_table<<<grid, block, 0, stream>>>(cos_table, sin_table, total, HS_, base);
	}

	const int *cu_seqlens() const {return d_cu_seqlens_;}
	std::vector<int> qkv_size() {return {NH_, NKV_, HS_};}

private:
	// 仅上传当前批次的有效元数据，避免按最大容量搬运未使用区域。
	void upload(cudaStream_t t) {
		int W = pool_->max_blocks_per_seq();
		cudaMemcpyAsync(d_table_, h_table_, (size_t)current_B_ * W * sizeof(int), cudaMemcpyHostToDevice, t);
		cudaMemcpyAsync(d_pos_, h_pos_, (size_t)current_B_ * sizeof(int), cudaMemcpyHostToDevice, t);
		cudaMemcpyAsync(d_positions_, h_positions_, (size_t)current_total_ * sizeof(int), cudaMemcpyHostToDevice, t);
		cudaMemcpyAsync(d_kv_dst_, h_kv_dst_, (size_t)current_total_ * sizeof(int), cudaMemcpyHostToDevice, t);
		cudaMemcpyAsync(d_cu_seqlens_, h_cu_seqlens_, (size_t)(current_B_ + 1) * sizeof(int), cudaMemcpyHostToDevice, t);
	}

	void select_half(int i) {
		int S = pool_->max_seqs(), W = pool_->max_blocks_per_seq();
		int *p = h_half_[i];
		h_base_ = p;
		h_table_      = p;   p += S * W;
		h_pos_        = p;   p += S;
		h_positions_  = p;   p += max_tokens_;
		h_kv_dst_     = p;   p += max_tokens_;
		h_cu_seqlens_ = p;
	}

	KV_Pool	*pool_;
	Arena &arena_;
	const int NH_, NKV_, HS_;
	const int max_tokens_;
	int total_ints_ = 0;
	int *h_base_ = nullptr;
	int *d_table_ = nullptr, *d_pos_ = nullptr, *d_cu_seqlens_ = nullptr;
	int *d_positions_ = nullptr, *d_kv_dst_ = nullptr;
	int *h_table_ = nullptr, *h_pos_ = nullptr, *h_cu_seqlens_ = nullptr;
	int *h_positions_ = nullptr, *h_kv_dst_ = nullptr;

	Tensor *meta_ = nullptr, *cos_ = nullptr, *sin_ = nullptr;

	int *h_half_[2] = {nullptr, nullptr};
	int step_ = 0;
	int current_B_ = 0;
	int current_total_ = 0;
};
