#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../kernel/LayerNorm/RMSNorm.cuh"

template<typename T>
class RMSNorm : public Module {
public:
	RMSNorm(LLM &llm, Tensor *input, int max_tokens, int hiddem_dim, float eps, std::string prefix)
	: input_(input), hidden_dim_(hiddem_dim), eps_(eps) {
		weight_ = llm.parameter(prefix + ".weight", {hidden_dim_}, dtype_of<T>::value);
		out_ = llm.arena().alloc({max_tokens, hidden_dim_}, dtype_of<T>::value);
		attach(llm, *this);
	}

	void forward(const GraphShape &shape, cudaStream_t stream) {
		const T* input = static_cast<const T*>(input_->ptr);
		const T* weight = static_cast<const T*>(weight_->ptr);
		T* out = static_cast<T*>(out_->ptr);
		launch_RMSNorm_forward(out, input, weight, shape.total_tokens, hidden_dim_, eps_, stream);
	}

	Tensor *out() {return out_;}

private:
	Tensor *input_, *weight_, *out_;
	int hidden_dim_;
	float eps_;
};
