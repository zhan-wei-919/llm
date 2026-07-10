#pragma once
#include "../core/Dtype.h"

constexpr int MAX_DIMS = 4;

struct Tensor {
	void	*ptr;
	int	shape[MAX_DIMS];
	int	ndim;
	Dtype	dtype;

	size_t numel() const {
		size_t n = 1;
		for (int i = 0; i < ndim; ++i) n *= shape[i];
		return n;
	}
	size_t bytes() const { return numel() * dtype_size(dtype);}
};
