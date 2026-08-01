#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../runtime/Engine.h"

template<typename T>
class PagedAttention : public Module {
public:
	// window_size=-1 读取完整历史，正数读取包含当前 token 的最近窗口。
	PagedAttention(LLM &llm, Engine &engine, Tensor *q, int layer, int window_size)
	: engine_(engine), q_(q), layer_(layer), window_size_(window_size) {
		out_ = llm.arena().alloc({q_->shape[0], q_->shape[1]}, dtype_of<T>::value);
		attach(llm, *this, {q_}, {out_});
	}

	void forward(const GraphShape &, cudaStream_t stream) {
		const T *q = static_cast<const T*>(q_->ptr);
		T *out = static_cast<T*>(out_->ptr);
		engine_.forward_attention(layer_, q, out, window_size_, stream);
	}

	Tensor *out() {return out_;}

private:
	Engine &engine_;
	Tensor *q_, *out_;
	int layer_, window_size_;
};
