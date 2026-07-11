#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../kernel/residual/Residual.cuh"

template<typename T>
class Residual : public Module {
public:
	Residual(LLM &llm, Tensor *x1, Tensor *x2, int max_tokens, int hidden_dim)
	:x1_(x1), x2_(x2), hidden_dim_(hidden_dim) {
		out_ = llm.arena().alloc({max_tokens, hidden_dim_}, dtype_of<T>::value);
		attach(llm, *this);
	}

	void forward(const GraphShape &shape, cudaStream_t stream) {
		const T *x1 = static_cast<const T*>(x1_->ptr);
		const T *x2 = static_cast<const T*>(x2_->ptr);
		T *out = static_cast<T*>(out_->ptr);
		launch_residual_forward(out, x1, x2, shape.total_tokens, hidden_dim_, stream);
	}

	Tensor *out() {return out_;}

private:
	Tensor *x1_, *x2_, *out_;
	int hidden_dim_;
};
