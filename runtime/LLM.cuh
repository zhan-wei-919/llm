#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <initializer_list>
#include <string>
#include <utility>
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
	Tensor *parameter(std::string name, std::initializer_list<int> shape, Dtype dtype) {
		Tensor *tensor = arena_.alloc(shape, dtype);
		parameters_.emplace(std::move(name), tensor);
		return tensor;
	}
	const std::unordered_map<std::string, Tensor *> &parameters() const {return parameters_;}
	cudaStream_t stream() const {return stream_;}

	void forward(const GraphShape &shape) {
		auto it = graphs_.find(shape);
		if (it == graphs_.end()) {
			cudaStreamSynchronize(stream_);
			auto exec = bake(shape);
			it = graphs_.emplace(shape, exec).first;
		}
		cudaGraphLaunch(it->second, stream_);
	}
	void finalize() {arena_.finalize();}


private:
	template<typename T>
	void append(T &module) {
		program_.emplace_back(module);
	}

	Arena arena_;
	std::vector<OpRecord> program_;
	std::unordered_map<std::string, Tensor *> parameters_;
	cudaStream_t stream_;
	std::unordered_map<GraphShape, cudaGraphExec_t, GraphShapeHash> graphs_;

};
