#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../kernel/activation/SwiGLU.cuh"

template<typename T>
class SwiMul : public Module {
public:
	SwiMul(LLM &llm, Tensor *gate, Tensor *up, int b, int t, int proj_dim, std::string prefix)
	: gate_(gate), up_(up), b_(b), t_(t), proj_dim_(proj_dim), prefix_(prefix) {
		out_ = llm.arena().alloc({b_, t_, proj_dim_}, dtype_of<T>::value);
		attach(llm, prefix_, *this);
	}

	void forward(cudaStream_t stream) {
		const T *gate = static_cast<const T*>(gate_->ptr);
		const T *up = static_cast<const T*>(up_->ptr);
		T *out = static_cast<T*>(out_->ptr);
		const int N = b_ * t_ * proj_dim_;
		launch_silu_mul(out, gate, up, N, stream);
	}

	Tensor *out() {return out_;}

private:
	Tensor *gate_, *up_, *out_;
	int b_, t_, proj_dim_;
	std::string prefix_;
};
