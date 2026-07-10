#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "../tensor/Arena.cuh"
#include "OpRecord.h"

class LLM {
	friend class Module;
public:
	explicit LLM(int max_tensors)
	: arena_(max_tensors){
		cudaStreamCreate(&stream_);
	}

	~LLM(){
		cudaStreamSynchronize(stream_);
		cudaGraphExecDestroy(graph_exec_);
		cudaGraphDestroy(graph_);
		cudaStreamDestroy(stream_);
	}

	void bake() {
		arena_.finalize();
		cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
		for(const auto &op : program_) op.forward(stream_);
		cudaStreamEndCapture(stream_, &graph_);
		cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
	}

	LLM(const LLM &) = delete;
	LLM &operator=(const LLM &) = delete;

	Arena &arena() {return arena_;}
	void forward() {cudaGraphLaunch(graph_exec_, stream_);}


private:
	template<typename T>
	void append(std::string name, T &module) {
		program_.emplace_back(std::move(name), module);
	}

	Arena arena_;
	std::vector<OpRecord> program_;
	cudaStream_t stream_;
	cudaGraph_t graph_;
	cudaGraphExec_t graph_exec_;

};
