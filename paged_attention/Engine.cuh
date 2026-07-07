#pragma once
#include <vector>
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
	// device 工作数组的尺寸要求 (装配层从 Arena 切好):
	//   d_table:      [max_seqs, max_blocks_per_seq]
	//   d_len/d_pos:  [max_seqs]
	//   d_seq_ids:    [max_seqs * max_seq_len]   (prefill 打包行数的最坏情况)
	//   d_cu_seqlens: [max_seqs + 1]
	Engine(KV_Pool &pool, int NH, int NKV, int HS,
	       int *d_table, int *d_len, int *d_pos, int *d_seq_ids, int *d_cu_seqlens)
	: pool_(pool), NH_(NH), NKV_(NKV), HS_(HS)
	, d_table_(d_table), d_len_(d_len), d_pos_(d_pos)
	, d_seq_ids_(d_seq_ids), d_cu_seqlens_(d_cu_seqlens)
	, h_table_((size_t)pool.max_seqs() * pool.max_blocks_per_seq())
	, h_len_(pool.max_seqs()), h_pos_(pool.max_seqs())
	, h_seq_ids_((size_t)pool.max_seqs() * pool.max_blocks_per_seq() * KV_BLOCK_SIZE)
	, h_cu_seqlens_(pool.max_seqs() + 1) {}

	int alloc_seq() { return pool_.alloc_seq(); }
	void release(int slot) { pool_.release(slot); }

	// prefill: 本批 B 条 prompt 首尾相接打包, lens[b] 是 slots[b] 的 prompt 长度
	// q: [total, NH*HS], k/v: [total, NKV*HS] 稠密激活, out: [total, NH*HS]
	// attention 读稠密 k/v (不查表), scatter 把同一份 k/v 写进 pool 为 decode 备货
	void prefill(const int *slots, int B, const int *lens, const void *q, const void *k, const void *v, void *out) {
		int W = pool_.max_blocks_per_seq();
		h_cu_seqlens_[0] = 0;
		for (int b = 0; b < B; ++b) {
			h_pos_[b] = pool_.append(slots[b], lens[b]);
			h_cu_seqlens_[b + 1] = h_cu_seqlens_[b] + lens[b];
		}
		int total = h_cu_seqlens_[B];
		pool_.gather_tables(slots, B, h_table_.data(), h_len_.data());
		for (int b = 0, t = 0; b < B; ++b)			// 展开: 第 b 段的 lens[b] 行都属于 b
			for (int i = 0; i < lens[b]; ++i) h_seq_ids_[t++] = b;
		upload(B, W, total);
		dtype_dispatch(pool_.dtype(), [&](auto tag) {
			using T_ = typename decltype(tag)::type;
			launch_scatter_kv<T_>(
				(T_ *)pool_.k_base(), (T_ *)pool_.v_base(),
				(const T_ *)k, (const T_ *)v,
				d_table_, d_cu_seqlens_, d_seq_ids_, d_pos_,
				total, NKV_, HS_, W);
			launch_gq_attention_prefill<T_>(
				(T_ *)out, (const T_ *)q, (const T_ *)k, (const T_ *)v,
				d_cu_seqlens_, d_seq_ids_,
				B, total, NH_, NKV_, HS_);
		});
	}

	// decode: 本批 B 条序列各产出 1 个新 token
	// q: [B, NH*HS], k/v: [B, NKV*HS], out: [B, NH*HS] —— 第 b 行都属于 slots[b],
	// 批内顺序是本步所有数组的共同坐标系
	void decode(const int *slots, int B, const void *q, const void *k, const void *v, void *out) {
		int W = pool_.max_blocks_per_seq();
		for (int b = 0; b < B; ++b)
			h_pos_[b] = pool_.append(slots[b], 1);		// 写入起点用旧 len
		pool_.gather_tables(slots, B, h_table_.data(), h_len_.data());	// 读范围用新 len
		for (int b = 0; b <= B; ++b) h_cu_seqlens_[b] = b;	// decode 打包形态固定: 每序列一行
		for (int b = 0; b < B; ++b) h_seq_ids_[b] = b;
		upload(B, W, /*total=*/B);
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
	void upload(int B, int W, int total) {
		cudaMemcpy(d_table_, h_table_.data(), sizeof(int) * B * W, cudaMemcpyHostToDevice);
		cudaMemcpy(d_len_, h_len_.data(), sizeof(int) * B, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pos_, h_pos_.data(), sizeof(int) * B, cudaMemcpyHostToDevice);
		cudaMemcpy(d_seq_ids_, h_seq_ids_.data(), sizeof(int) * total, cudaMemcpyHostToDevice);
		cudaMemcpy(d_cu_seqlens_, h_cu_seqlens_.data(), sizeof(int) * (B + 1), cudaMemcpyHostToDevice);
	}

	KV_Pool		&pool_;
	const int	NH_, NKV_, HS_;
	int *const	d_table_;
	int *const	d_len_;
	int *const	d_pos_;
	int *const	d_seq_ids_;
	int *const	d_cu_seqlens_;
	std::vector<int> h_table_, h_len_, h_pos_, h_seq_ids_, h_cu_seqlens_;
};
