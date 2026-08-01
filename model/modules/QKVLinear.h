#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <string>
#include <type_traits>
#include <vector>
#include "../Module.h"
#include "../../runtime/Engine.h"
#include "../../tensor/Tensor.h"

template<typename T>
class QKVLinear : public Module {
public:
	// 注册 checkpoint 中独立的 q/k/v 参数，但只分配一份融合权重和一份 Q 输出。
	QKVLinear(LLM &llm, Engine &engine, Tensor *input, int max_tokens, int in_channel,
	          int layer, bool has_bias, const std::string &prefix)
	: engine_(engine), input_(input), in_channel_(in_channel), layer_(layer), has_bias_(has_bias) {
		static_assert(std::is_same<T, __nv_bfloat16>::value || std::is_same<T, half>::value,
		              "QKVLinear only supports F16/BF16");
		std::vector<int> dims = engine_.qkv_size();
		int NH = dims[0], NKV = dims[1], HS = dims[2];
		q_channel_ = NH * HS;
		kv_channel_ = NKV * HS;
		int total_channel = q_channel_ + 2 * kv_channel_;
		weight_ = llm.parameter_storage({in_channel_, total_channel}, dtype_of<T>::value);
		llm.bind_parameter_slice(prefix + ".q_proj.weight", weight_, 0, q_channel_, HS);
		llm.bind_parameter_slice(prefix + ".k_proj.weight", weight_, q_channel_, kv_channel_, HS);
		llm.bind_parameter_slice(prefix + ".v_proj.weight", weight_, q_channel_ + kv_channel_, kv_channel_);
		if (has_bias_) {
			bias_ = llm.parameter_storage({total_channel}, dtype_of<T>::value);
			llm.bind_parameter_slice(prefix + ".q_proj.bias", bias_, 0, q_channel_, HS);
			llm.bind_parameter_slice(prefix + ".k_proj.bias", bias_, q_channel_, kv_channel_, HS);
			llm.bind_parameter_slice(prefix + ".v_proj.bias", bias_, q_channel_ + kv_channel_, kv_channel_);
		}
		out_ = llm.arena().alloc({max_tokens, q_channel_}, dtype_of<T>::value);
		attach(llm, *this, {input_}, {out_});
	}

	// 执行融合投影；返回旋转后的 Q，并把旋转后的 K 与 V 写入当前层 KV Pool。
	void forward(const GraphShape &, cudaStream_t stream) {
		engine_.forward_qkv<T>(layer_, static_cast<const T *>(input_->ptr),
			static_cast<const T *>(weight_->ptr),
			has_bias_ ? static_cast<const T *>(bias_->ptr) : nullptr,
			static_cast<T *>(out_->ptr), in_channel_, stream);
	}

	Tensor *out() {return out_;}
	Tensor *weight() {return weight_;}
	Tensor *bias() {return bias_;}

private:
	Engine &engine_;
	Tensor *input_, *weight_, *bias_ = nullptr, *out_;
	int in_channel_, q_channel_, kv_channel_, layer_;
	bool has_bias_;
};
