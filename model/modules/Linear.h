#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include <type_traits>
#include "../../kernel/Gemm/Gemm.cuh"
#include "../../kernel/Gemm/Gemm_f32.cuh"

template<typename T>
class Linear : public Module {
public:
	Linear(LLM &llm, Tensor *input, int b, int t, int in_channel, int out_channel, bool has_bias, std::string prefix)
	: input_(input), b_(b), t_(t), in_channel_(in_channel), out_channel_(out_channel), has_bias_(has_bias), prefix_(prefix){
		weight_ = llm.arena().alloc({in_channel_, out_channel_}, dtype_of<T>::value);
		out_ = llm.arena().alloc({b_, t_, out_channel_}, dtype_of<T>::value);
		bias_ = has_bias_? llm.arena().alloc({out_channel_}, dtype_of<T>::value) : nullptr;
		attach(llm, prefix_, *this);
	}

	void forward(cudaStream_t stream) {
		const T *input = static_cast<const T*>(input_->ptr);
		const T *weight = static_cast<const T*>(weight_->ptr);
		T *out = static_cast<T*>(out_->ptr);
		const T *bias = has_bias_? static_cast<const T*>(bias_->ptr) : nullptr;
		int M = b_ * t_;
		int N = out_channel_;
		int K = in_channel_;
		if constexpr (std::is_same_v<T, float>) {
			launch_gemm_f32_forward(input, weight, out, bias, 1.0f, 0.0f, M, N, K, stream);
		} else {
			using Config = GemmConfig<T, T>;
			launch_Gemm_forward<Config>(input, weight, out, bias, 1.0f, 0.0f, M, N, K, stream);
		}
	};

	Tensor *out() {return out_;}

private:
	Tensor *input_, *weight_, *bias_, *out_;
	int b_, t_, in_channel_, out_channel_;
	bool has_bias_;
	std::string prefix_;
};
