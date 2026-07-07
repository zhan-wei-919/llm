#pragma once
#include <cstdlib>
#include <initializer_list>
#include <cuda_runtime.h>
#include "../../core/Dtype.h"
#include "Tensor.h"

struct KVAlloc {
	void	*k_base;	// 未定型的原始显存, 由消费方 cast 赋予解释
	void	*v_base;
	size_t	bytes_each;	// 字节
};

class Arena {
public:
	explicit Arena(int max_tensors)
	: entries_((Tensor *)malloc(sizeof(Tensor) * max_tensors))
	, max_tensors_(max_tensors) {}

	~Arena() {
		if (kv_base_) cudaFree(kv_base_);
		if (base_) cudaFree(base_);
		free(entries_);
	}

	Tensor *alloc(std::initializer_list<int> shape, Dtype d) {
		if (count_ == max_tensors_) __builtin_trap();
		Tensor *t = &entries_[count_++];
		t->ptr = nullptr;
		t->ndim = 0;
		for (int dim : shape) t->shape[t->ndim++] = dim;
		t->dtype = d;
		return t;
	}

	void finalize() {
		size_t total = 0;
		for (int i = 0; i < count_; ++i) total += align_up(entries_[i].bytes());
		cudaError_t err = cudaMalloc((void **)&base_, total);
		if (err != cudaSuccess) __builtin_trap();
		size_t offset = 0;
		for (int i = 0; i < count_; ++i) {
			entries_[i].ptr = (void*)((char*)base_ + offset);
			offset += align_up(entries_[i].bytes());
		}
	}

	// 必须在 finalize 之后调用
	KVAlloc alloc_kv_pool(double utilization = 0.9) {
		size_t free_bytes, total_bytes;
		cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
		if (err != cudaSuccess) __builtin_trap();
		// 向下对齐到 256, 保证中点切出来的 v_base 也对齐
		size_t half = ((size_t)(free_bytes * utilization) >> 1) & ~(size_t)255;
		err = cudaMalloc((void **)&kv_base_, 2 * half);
		if (err != cudaSuccess) __builtin_trap();
		return KVAlloc {kv_base_, (void*)((char*)kv_base_ + half), half};
	}

private:
	static size_t align_up(size_t n) {return (n + 255) &~(size_t)255;}
	Tensor	*const	entries_;
	const	int	max_tensors_;
	int		count_ = 0;
	void		*base_ = nullptr;
	void		*kv_base_ = nullptr;
};
