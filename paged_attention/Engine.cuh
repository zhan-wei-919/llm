#pragma once
#include <cuda_runtime.h>
#include "KV_pool.h"
#include "../kernel/attention/Scatter_kv.cuh"
#include "../kernel/attention/GQAttention_decode.cuh"
#include "../kernel/attention/GQAttention_prefill.cuh"

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
	Engine(KV_Pool &pool, int NH, int NKV, int HS, int *d_base, int *h_base)
	: pool_(pool), NH_(NH), NKV_(NKV), HS_(HS), d_base_(d_base), h_base_(h_base) {
		int S = pool.max_seqs();
		int W = pool.max_blocks_per_seq();
		int L = pool.max_blocks_per_seq() * KV_BLOCK_SIZE;
		d_table_	= d_base;	d_base += S * W;	total_ints_ += S * W;
		d_len_		= d_base;	d_base += S;		total_ints_ += S;
		d_pos_		= d_base;	d_base += S;		total_ints_ += S;
		d_seq_ids_	= d_base;	d_base += S * L;	total_ints_ += S * L;
		d_cu_seqlens_	= d_base;				total_ints_ += S + 1;
		h_half_[0] = h_base;
		h_half_[1] = h_base + total_ints_;
		select_half(0);
	}

	int alloc_seq() { return pool_.alloc_seq(); }
	void release(int slot) { pool_.release(slot); }

	// prefill: 本批 B 条 prompt 首尾相接打包, lens[b] 是 slots[b] 的 prompt 长度
	// q: [total, NH*HS], k/v: [total, NKV*HS] 稠密激活, out: [total, NH*HS]
	// attention 读稠密 k/v (不查表), scatter 把同一份 k/v 写进 pool 为 decode 备货
	void prefill(const int *slots, int B, const int *lens, const void *q, const void *k, const void *v, void *out) {
		select_half(0);
		int W = pool_.max_blocks_per_seq();
		h_cu_seqlens_[0] = 0;
		for (int b = 0; b < B; ++b) {
			h_pos_[b] = pool_.append(slots[b], lens[b]);
			h_cu_seqlens_[b + 1] = h_cu_seqlens_[b] + lens[b];
		}
		int total = h_cu_seqlens_[B];
		pool_.gather_tables(slots, B, h_table_, h_len_);
		for (int b = 0, t = 0; b < B; ++b)			// 展开: 第 b 段的 lens[b] 行都属于 b
			for (int i = 0; i < lens[b]; ++i) h_seq_ids_[t++] = b;
		upload();
		dtype_dispatch(pool_.dtype(), [&](auto tag) {
			using T_ = typename decltype(tag)::type;
			launch_scatter_kv<T_>(
				(T_ *)pool_.k_base(), (T_ *)pool_.v_base(),
				(const T_ *)k, (const T_ *)v,
				d_table_, d_cu_seqlens_, d_seq_ids_, d_pos_,
				total, NKV_, HS_, W);
			launch_gq_attention_prefill<T_>(
				(T_ *)out, (const T_ *)q, (const T_ *)pool_.k_base(), (const T_ *)pool_.v_base(),
				d_cu_seqlens_, d_seq_ids_, d_pos_, d_table_,
				B, NH_, NKV_, HS_, W, total);
		});
	}

	// decode: 本批 B 条序列各产出 1 个新 token
	// q: [B, NH*HS], k/v: [B, NKV*HS], out: [B, NH*HS] —— 第 b 行都属于 slots[b],
	// 批内顺序是本步所有数组的共同坐标系
	void decode(const int *slots, int B, const void *q, const void *k, const void *v, void *out) {
		select_half(step_++ & 1);
		int W = pool_.max_blocks_per_seq();
		for (int b = 0; b < B; ++b)
			h_pos_[b] = pool_.append(slots[b], 1);		// 写入起点用旧 len
		pool_.gather_tables(slots, B, h_table_, h_len_);	// 读范围用新 len
		for (int b = 0; b <= B; ++b) h_cu_seqlens_[b] = b;	// decode 打包形态固定: 每序列一行
		for (int b = 0; b < B; ++b) h_seq_ids_[b] = b;
		upload();
		dtype_dispatch(pool_.dtype(), [&](auto tag) {
			using T_ = typename decltype(tag)::type;
			launch_scatter_kv<T_>(
				(T_ *)pool_.k_base(), (T_ *)pool_.v_base(),
				(const T_ *)k, (const T_ *)v,
				d_table_, d_cu_seqlens_, d_seq_ids_, d_pos_,
				B, NKV_, HS_, W);
			launch_gq_attention_decode<T_>(
				(T_ *)out, (const T_ *)q,
				(const T_ *)pool_.k_base(), (const T_ *)pool_.v_base(),
				d_table_, d_len_,
				B, NH_, NKV_, HS_, W);
		});
	}

private:
	void upload() {
		size_t bytes = total_ints_ * sizeof(int);
		cudaMemcpyAsync(d_base_, h_base_, bytes, cudaMemcpyHostToDevice);
	}

	void select_half(int i) {
		int S = pool_.max_seqs(), W = pool_.max_blocks_per_seq(), L = W * KV_BLOCK_SIZE;
		int *p = h_half_[i];
		h_base_ = p;
		h_table_      = p;   p += S * W;
		h_len_        = p;   p += S;
		h_pos_        = p;   p += S;
		h_seq_ids_    = p;   p += S * L;
		h_cu_seqlens_ = p;
	}

	KV_Pool	&pool_;
	const int NH_, NKV_, HS_;
	size_t total_ints_ = 0;
	int *d_base_, *h_base_;
	int *d_table_, *d_len_, *d_pos_, *d_seq_ids_, *d_cu_seqlens_;
	int *h_table_, *h_len_, *h_pos_, *h_seq_ids_, *h_cu_seqlens_;

	int *h_half_[2];
	int step_ = 0;
};
