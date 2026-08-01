#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../runtime/Engine.h"
#include "PagedAttention.h"
#include "Linear.h"
#include "QKVLinear.h"

// x -> 融合 QKVLinear；Q 连续输出，K/V 直接落池 -> PagedAttention -> o_proj

template<typename T>
class Attention : public Module {
public:
	Attention(LLM &llm, Engine &engine, Tensor *input, int max_tokens, int hidden_dim, int layer, bool qkv_has_bias, bool o_has_bias, std::string prefix)
	: qkv_proj_(llm, engine, input, max_tokens, hidden_dim, layer, qkv_has_bias, prefix)
	, paged_attention_(llm, engine, qkv_proj_.out(), layer)
	, o_proj_(llm, paged_attention_.out(), max_tokens, engine.qkv_size()[0] * engine.qkv_size()[2], hidden_dim, o_has_bias, prefix + ".o_proj"){}

	Tensor *out() {return o_proj_.out();}

private:
	QKVLinear<T> qkv_proj_;
	PagedAttention<T> paged_attention_;
	Linear<T> o_proj_;
};
