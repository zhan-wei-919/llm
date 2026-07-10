#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../kernel/residual/Residual.cuh"

template<typename T>
class Residual : public Module {
public:
	Residual(LLM &llm, Tensor *x1, Tensor *x2, int b, int t, int hidden_dim, std::string prefix)
	:x1_(x1), x2_(x2), b_(b), t_(t), hidden_dim_(hidden_dim), prefix_(prefix) {
		out_ = llm.arena().alloc({b_, t_, hidden_dim_}, dtype_of<T>::value);
		attach(llm, prefix_, *this);
	}

	void forward(cudaStream_t stream) {
		const T *x1 = static_cast<const T*>(x1_->ptr);
		const T *x2 = static_cast<const T*>(x2_->ptr);
		T *out = static_cast<T*>(out_->ptr);
		launch_residual_forward(out, x1, x2, b_, t_, hidden_dim_, stream);
	}

	Tensor *out() {return out_;}

private:
	Tensor *x1_, *x2_, *out_;
	int b_, t_, hidden_dim_;
	std::string prefix_;
};
