#pragma once
#include <string>
#include "Linear.h"

// [total, hidden_dim] -> [total, vocab_size]
// 当前使用独立的 [hidden_dim, vocab_size] 权重，不与 Embedding 共享。
template<typename T>
class LMHead {
public:
	LMHead(LLM &llm, Tensor *input, int max_tokens,
	       int hidden_dim, int vocab_size, std::string prefix)
	: proj_(llm, input, max_tokens, hidden_dim, vocab_size,
	        /*has_bias=*/false, std::move(prefix)) {}

	Tensor *out() { return proj_.out(); }
	Tensor *weight() { return proj_.weight(); }

private:
	Linear<T> proj_;
};
