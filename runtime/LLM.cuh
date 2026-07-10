#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
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
		for (auto &[shape, exec] : graphs_) cudaGraphExecDestroy(exec);
		cudaStreamDestroy(stream_);
	}

	cudaGraphExec_t bake(const GraphShape &shape) {
		cudaGraph_t graph;
		cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
		for (const auto &op : program_) op.forward(shape, stream_);
		cudaStreamEndCapture(stream_, &graph);
		cudaGraphExec_t exec;
		cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
		cudaGraphDestroy(graph);
		return exec;
	}

	LLM(const LLM &) = delete;
	LLM &operator=(const LLM &) = delete;

	Arena &arena() {return arena_;}
	void forward(const GraphShape &shape) {
		auto it = graphs_.find(shape);
		if (it == graphs_.end()) {
			auto exec = bake(shape);
			it = graphs_.emplace(shape, exec).first;
		}
		cudaGraphLaunch(it->second, stream_);
	}
	void finalize() {arena_.finalize();}


private:
	template<typename T>
	void append(std::string name, T &module) {
		program_.emplace_back(std::move(name), module);
	}

	Arena arena_;
	std::vector<OpRecord> program_;
	cudaStream_t stream_;
	std::unordered_map<GraphShape, cudaGraphExec_t, GraphShapeHash> graphs_;

};
