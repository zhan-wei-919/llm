#pragma once
#include <cuda_runtime.h>
#include "../kv/KV_pool.h"
#include "../tensor/Tensor.h"
#include "../tensor/Arena.h"
#include "../kernel/attention/Scatter_kv.h"
#include "../kernel/attention/GQAttention_prefill.h"
#include "../kernel/embedding/RoPECache.h"
#include "../kernel/embedding/RoPE.h"
#include "GraphShape.h"

// Engine: 把"一步推理"翻译成固定的操作序列
//   记账 (调 pool) → 收集 (gather_tables) → 搬运 (memcpy 上传) → 发射 (kernel)
// 只借不占: pool 和 device 工作数组由装配层构造好递进来, 这里不做任何显存分配.
// scatter 和 attention 在同一个 stream 上按发射顺序执行, 先写后读天然成立,
// 所以全程不需要显式同步.
class Engine {
public:
	// S = max_seqs
	// W = max_blocks_per_seq
	// L = max_seq_len  (= S * W * KV_BLOCK_SIZE 的上界，但实际是 pool 的 max_seq_len)
	// table:      S * W       个 int
	// len:        S           个 int
	// pos:        S           个 int
	// seq_ids:    S * L       个 int   (最坏情况)
	// cu_seqlens: S + 1       个 int
	Engine(Arena &arena, int NH, int NKV, int HS, int max_seqs, int max_seq_len)
	: arena_(arena), NH_(NH), NKV_(NKV), HS_(HS) {
		int S = max_seqs, W = (max_seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE, L = W * KV_BLOCK_SIZE;
		total_ints_ = S * W + S + S + S * L + S + 1;
		meta_ = arena_.alloc({total_ints_}, Dtype::I32);
		cos_ = arena_.alloc({L, HS / 2}, Dtype::F32);
		sin_ = arena_.alloc({L, HS / 2}, Dtype::F32);
	}

	void bind_pool(KV_Pool *pool) {
		pool_ = pool; int *d_base = (int*)meta_->ptr; d_base_ = d_base;
		int S = pool_->max_seqs(), W = pool_->max_blocks_per_seq(), L = pool_->max_blocks_per_seq() * KV_BLOCK_SIZE;
		d_table_	= d_base;	d_base += S * W;
		d_len_		= d_base;	d_base += S;
		d_pos_		= d_base;	d_base += S;
		d_seq_ids_	= d_base;	d_base += S * L;
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
		pool_->gather_tables(slots, B, h_table_, h_len_);
		for (int b = 0, t = 0; b < B; ++b)			// 展开: 第 b 段的 lens[b] 行都属于 b
			for (int i = 0; i < lens[b]; ++i) h_seq_ids_[t++] = b;
		upload(stream);
		return {current_B_, current_total_};
	}

	// forward: 混合批的统一一步. 本批 B 条序列首尾相接打包, lens[b] 是第 b 条本步
	// 要算的 token 数 —— decode 条目为 1, 追赶 chunk 为一段, 同一次发射混装.
	// q: [total, NH*HS], k/v: [total, NKV*HS] 稠密激活, out: [total, NH*HS]
	// scatter 先把本步 k/v 写进 pool, attention 经页表读 [0, pos+i] 的全部前缀
	void forward_layer(int layer, const void *q, const void *k, const void *v, void *out, cudaStream_t stream){
		auto k_base = pool_->k_base(layer);
		auto v_base = pool_->v_base(layer);
		int W = pool_->max_blocks_per_seq();
		dtype_dispatch(pool_->dtype(), [&](auto tag) {
			using T_ = typename decltype(tag)::type;
			launch_scatter_kv<T_>(
				(T_ *)k_base, (T_ *)v_base,
				(const T_ *)k, (const T_ *)v,
				d_table_, d_cu_seqlens_, d_seq_ids_, d_pos_,
				current_total_, NKV_, HS_, W, stream);
			launch_gq_attention_prefill<T_>(
				(T_ *)out, (const T_ *)q, (const T_ *)k_base, (const T_ *)v_base,
				d_cu_seqlens_, d_seq_ids_, d_pos_, d_table_,
				current_B_, NH_, NKV_, HS_, W, current_total_, stream);
		});
	}

	void apple_rope(void *q, void *k, const float *cos_table, const float *sin_table, cudaStream_t stream){
		dtype_dispatch(pool_->dtype(), [&](auto tag){
			using T = typename decltype(tag)::type;
			launch_rope(static_cast<T*>(q), cos_table, sin_table, d_cu_seqlens_, d_seq_ids_, d_pos_, current_total_, NH_, HS_, stream);
			launch_rope(static_cast<T*>(k), cos_table, sin_table, d_cu_seqlens_, d_seq_ids_, d_pos_, current_total_, NKV_, HS_, stream);
		});
	}

	void init_rope(cudaStream_t stream, float base = 10000.0f) {
		int total = pool_->max_blocks_per_seq() * KV_BLOCK_SIZE * (HS_ / 2);
		int block = 256;
		int grid = (total + block - 1) / block;
		float *cos_table = static_cast<float*>(cos_->ptr);
		float *sin_table = static_cast<float*>(sin_->ptr);
		init_rope_table<<<grid, block, 0, stream>>>(cos_table, sin_table, total, HS_, base);
	}

	Tensor *cos_table() {return cos_;}
	Tensor *sin_table() {return sin_;}
	const int *cu_seqlens() const {return d_cu_seqlens_;}
	std::vector<int> qkv_size() {return {NH_, NKV_, HS_};}

private:
	void upload(cudaStream_t t) {
		size_t bytes = total_ints_ * sizeof(int);
		cudaMemcpyAsync(d_base_, h_base_, bytes, cudaMemcpyHostToDevice, t);
	}

	void select_half(int i) {
		int S = pool_->max_seqs(), W = pool_->max_blocks_per_seq(), L = W * KV_BLOCK_SIZE;
		int *p = h_half_[i];
		h_base_ = p;
		h_table_      = p;   p += S * W;
		h_len_        = p;   p += S;
		h_pos_        = p;   p += S;
		h_seq_ids_    = p;   p += S * L;
		h_cu_seqlens_ = p;
	}

	KV_Pool	*pool_;
	Arena &arena_;
	const int NH_, NKV_, HS_;
	int total_ints_ = 0;
	int *d_base_ = nullptr, *h_base_ = nullptr;
	int *d_table_ = nullptr, *d_len_ = nullptr, *d_pos_ = nullptr, *d_seq_ids_ = nullptr, *d_cu_seqlens_ = nullptr;
	int *h_table_ = nullptr, *h_len_ = nullptr, *h_pos_ = nullptr, *h_seq_ids_ = nullptr, *h_cu_seqlens_ = nullptr;

	Tensor *meta_ = nullptr, *cos_ = nullptr, *sin_ = nullptr;

	int *h_half_[2] = {nullptr, nullptr};
	int step_ = 0;
	int current_B_ = 0;
	int current_total_ = 0;
};
