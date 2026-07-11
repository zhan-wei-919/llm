#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../kernel/embedding/TokenEmbedding.cuh"

template<typename T>
class Embedding : public Module {
public:
	Embedding(LLM &llm, Tensor *token_ids, int max_tokens,
	          int vocab_size, int hidden_dim, std::string prefix)
	: token_ids_(token_ids), hidden_dim_(hidden_dim), prefix_(std::move(prefix)) {
		weight_ = llm.arena().alloc({vocab_size, hidden_dim}, dtype_of<T>::value);
		out_ = llm.arena().alloc({max_tokens, hidden_dim}, dtype_of<T>::value);
		attach(llm, prefix_, *this);
	}

	void forward(const GraphShape &shape, cudaStream_t stream) {
		launch_token_embedding(
			static_cast<T *>(out_->ptr),
			static_cast<const int *>(token_ids_->ptr),
			static_cast<const T *>(weight_->ptr),
			shape.total_tokens,
			hidden_dim_,
			stream
		);
	}

	Tensor *out() { return out_; }
	Tensor *weight() { return weight_; }

private:
	Tensor *token_ids_, *weight_, *out_;
	int hidden_dim_;
	std::string prefix_;
};
