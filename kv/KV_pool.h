#pragma once
#include "../core/Dtype.h"
#include "KV_Layout.h"
#include <vector>
#include <cstring>
#include <cassert>

// KV_Pool: 物理块分配器 + 每个序列的页表
//
// 页表是一块预分配的稠密矩阵 block_table_[max_seqs, max_blocks_per_seq]:
// 行号(slot)就是序列 id, 行内下标是逻辑块号, 值是物理块号.
// 每行的已用块数不单独存, 由不变量 块数 == ceil(len/16) 从 len_ 反推.
class KV_Pool {
public:
	KV_Pool(void *k_base, void *v_base, Dtype d, size_t kv_stride, size_t capacity, int num_layers, int max_seqs, int max_seq_len)
	: k_base_(k_base), v_base_(v_base), dtype_(d), kv_stride_(kv_stride), num_layers_(num_layers)
	, num_blocks_(capacity / ((size_t)num_layers * KV_BLOCK_SIZE * kv_stride * dtype_size(d)))
	, layer_bytes_(num_blocks_ * KV_BLOCK_SIZE * kv_stride_ * dtype_size(d))
	, max_seqs_(max_seqs)
	, max_blocks_per_seq_((max_seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE)
	, block_table_((size_t)max_seqs * max_blocks_per_seq_)
	, len_(max_seqs, 0) {
		free_list_.reserve(num_blocks_);
		for (int i = num_blocks_ - 1; i >= 0; --i) free_list_.push_back(i);
		free_slots_.reserve(max_seqs);
		for (int s = max_seqs - 1; s >= 0; --s) free_slots_.push_back(s);
	}

	// 新序列入场: 领一个空行
	// 契约: 资源够不够由调度层保证, 这里只断言契约没被打破
	int alloc_seq() {
		assert(!free_slots_.empty());
		int slot = free_slots_.back();
		free_slots_.pop_back();
		len_[slot] = 0;
		return slot;
	}

	// 序列 slot 即将写入 n 个 token: 补齐页表, 返回写入起点(旧 len, 即 pos_offset)
	int append(int slot, int n) {
		int *row = &block_table_[(size_t)slot * max_blocks_per_seq_];
		int have = (len_[slot] + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
		int need = (len_[slot] + n + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
		for (int q = have; q < need; ++q) {
			assert(!free_list_.empty());
			row[q] = free_list_.back();
			free_list_.pop_back();
		}
		int pos = len_[slot];
		len_[slot] += n;
		return pos;
	}

	// 序列结束: 物理块和槽位都归还
	void release(int slot) {
		const int *row = &block_table_[(size_t)slot * max_blocks_per_seq_];
		int have = (len_[slot] + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
		for (int q = 0; q < have; ++q) free_list_.push_back(row[q]);
		free_slots_.push_back(slot);
	}

	// 把本批序列的页表拍成 [B, max_blocks_per_seq] 供上传, 顺手带出 cur_len
	// 行宽固定, 每行一次 memcpy; 行尾 padding 不清理, kernel 读不到那里
	void gather_tables(const int *slots, int B, int *table_out, int *cur_len_out) const {
		for (int b = 0; b < B; ++b) {
			memcpy(table_out + (size_t)b * max_blocks_per_seq_,
			       &block_table_[(size_t)slots[b] * max_blocks_per_seq_],
			       max_blocks_per_seq_ * sizeof(int));
			cur_len_out[b] = len_[slots[b]];
		}
	}

	void *k_base(int layer) const {return  static_cast<void*>(static_cast<char*>(k_base_) +(size_t)(layer) * layer_bytes_);}

	void *v_base(int layer) const {return static_cast<void*>(static_cast<char*>(v_base_) + (size_t)(layer) * layer_bytes_);}

	int seq_len(int slot) const { return len_[slot]; }
	int max_seqs() const { return max_seqs_; }
	int max_blocks_per_seq() const { return max_blocks_per_seq_; }
	Dtype dtype() const { return dtype_; }
	size_t num_free_blocks() const { return free_list_.size(); }
	size_t num_free_slots() const { return free_slots_.size(); }

private:
	void *const k_base_;
	void *const v_base_;
	const Dtype dtype_;
	const size_t kv_stride_;
	const int num_layers_;
	const size_t num_blocks_;
	const size_t layer_bytes_;
	const int max_seqs_;
	const int max_blocks_per_seq_;
	std::vector<int> block_table_;	// [max_seqs, max_blocks_per_seq] 稠密页表
	std::vector<int> len_;		// [max_seqs] 每个 slot 已写入的 token 数
	std::vector<int> free_list_;	// 空闲物理块号
	std::vector<int> free_slots_;	// 空闲行号
};
