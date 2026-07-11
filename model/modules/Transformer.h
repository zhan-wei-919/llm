#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../runtime/Engine.h"
#include "Attention.h"
#include "RMSNorm.h"
#include "Residual.h"
#include "MLP.h"

//	x -> RMSNorm -> Attention -> Attention.out() + x -> RMSNorm -> MLP -> MLP.out() + x

template<typename T>
class Transformer : public Module {
public:
	Transformer(
		LLM &llm, Engine &engine, Tensor *input, int max_tokens, int hidden_dim, int layer, int fc_proj_dim,
		bool qkv_has_bias, bool o_has_boas, bool fc_has_bias, float eps, std::string prefix)
	: ln1_(llm, input, max_tokens, hidden_dim, eps, prefix + ".ln1")
	, attn_(llm, engine, ln1_.out(), max_tokens, hidden_dim, layer, qkv_has_bias, o_has_boas, prefix + ".attn")
	, r1_(llm, input, attn_.out(), max_tokens, hidden_dim)
	, ln2_(llm, r1_.out(), max_tokens, hidden_dim, eps, prefix + ".ln2")
	, fc_(llm, ln2_.out(), max_tokens, hidden_dim, fc_proj_dim, fc_has_bias, prefix + ".fc")
	, r2_(llm, r1_.out(), fc_.out(),  max_tokens, hidden_dim) {}

	Tensor *out() {return r2_.out();}
private:
	RMSNorm<T> ln1_;
	Attention<T> attn_;
	Residual<T> r1_;
	RMSNorm<T> ln2_;
	MLP<T> fc_;
	Residual<T> r2_;
};
