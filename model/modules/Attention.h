#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../runtime/Engine.h"
#include "PagedAttention.h"
#include "Linear.h"
#include "RoPE.h"

// x [max_tokes, hidden_dim] -> q_proj -> q [max_tokens, NH * HS] -> attention -> o_proj -> x[max_tokens, hidden_dim]
//			-> k_proj -> k [max_tokens, NKV * HS]
// 			-> v_proj -> v [max_tokens, NKV * HS]

template<typename T>
class Attention : public Module {
public:
	Attention(LLM &llm, Engine &engine, Tensor *input, int max_tokens, int hidden_dim, int layer, bool qkv_has_bias, bool o_has_bias, std::string prefix)
	: q_proj_(llm, input, max_tokens, hidden_dim, engine.qkv_size()[0] * engine.qkv_size()[2], qkv_has_bias, prefix + ".q_proj")
	, k_proj_(llm, input, max_tokens, hidden_dim, engine.qkv_size()[1] * engine.qkv_size()[2], qkv_has_bias, prefix + ".k_proj")
	, v_proj_(llm, input, max_tokens, hidden_dim, engine.qkv_size()[1] * engine.qkv_size()[2], qkv_has_bias, prefix + ".v_proj")
	, rope_(llm, engine, q_proj_.out(), k_proj_.out(), prefix + ".rope")
	, paged_attention_(llm, engine, q_proj_.out(), k_proj_.out(), v_proj_.out(), layer, prefix + ".attn")
	, o_proj_(llm, paged_attention_.out(), max_tokens, engine.qkv_size()[0] * engine.qkv_size()[2], hidden_dim, o_has_bias, prefix + ".o_proj"){}

	Tensor *out() {return o_proj_.out();}

private:
	Linear<T> q_proj_, k_proj_, v_proj_;
	RoPE<T> rope_;
	PagedAttention<T> paged_attention_;
	Linear<T> o_proj_;
};
