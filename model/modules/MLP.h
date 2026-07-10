#pragma once
#include <string>
#include "Linear.h"
#include "SwiMul.h"

template<typename T>
class MLP {
public:
	MLP(LLM &llm, Tensor *input, int b, int t, int hidden_dim, int proj_dim, bool has_bias, const std::string &prefix)
	: gate_proj_(llm, input, b, t, hidden_dim, proj_dim, has_bias, prefix + ".gate_proj")
	, up_proj_(llm, input, b, t, hidden_dim, proj_dim, has_bias, prefix + ".up_proj")
	, swi_mul_(llm, gate_proj_.out(), up_proj_.out(), b, t, proj_dim, prefix + ".swi_mul")
	, down_proj_(llm, swi_mul_.out(), b, t, proj_dim, hidden_dim, has_bias, prefix + ".down_proj") {}

	Tensor *out() {return down_proj_.out();}
private:
	Linear<T> gate_proj_;
	Linear<T> up_proj_;
	SwiMul<T> swi_mul_;
	Linear<T> down_proj_;
};
