#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../kernel/activation/SwiGLU.h"

template<typename T>
class SwiMul : public Module {
public:
	SwiMul(LLM &llm, Tensor *gate, Tensor *up, int max_tokens, int proj_dim)
	: gate_(gate), up_(up), proj_dim_(proj_dim) {
		out_ = llm.arena().alloc({max_tokens, proj_dim_}, dtype_of<T>::value);
		attach(llm, *this, {gate_, up_}, {out_});
	}

	void forward(const GraphShape &shape, cudaStream_t stream) {
		const T *gate = static_cast<const T*>(gate_->ptr);
		const T *up = static_cast<const T*>(up_->ptr);
		T *out = static_cast<T*>(out_->ptr);
		const int N = shape.total_tokens * proj_dim_;
		launch_silu_mul(out, gate, up, N, stream);
	}

	Tensor *out() {return out_;}

private:
	Tensor *gate_, *up_, *out_;
	int proj_dim_;
};
