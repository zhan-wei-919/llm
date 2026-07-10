#pragma once
#include <cuda_runtime.h>
#include <string>
#include <utility>
#include "GraphShape.h"

class OpRecord {
public:
	template<typename Module>
	OpRecord(std::string name, Module &module)
	: name_(std::move(name)), module_(&module), forward_(&call_forward<Module>){}

	void forward(const GraphShape &shape, cudaStream_t stream) const {forward_(module_, shape, stream);}

	const std::string &name() const {return name_;}

private:
	using ForwardFn = void (*)(void*, const GraphShape &, cudaStream_t);

	template<typename Module>
	static void call_forward(void *module,const GraphShape &shape, cudaStream_t stream) {
		static_cast<Module*>(module)->forward(shape, stream);
	}

	std::string name_;
	void *module_;
	ForwardFn forward_;
};
