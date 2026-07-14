#include "../../modules/Attention.h"
#include "../../modules/Embedding.h"
#include "../../modules/LMHead.h"
#include "../../modules/MLP.h"
#include "../../modules/RMSNorm.h"
#include "../../modules/Residual.h"
#include "../../../runtime/Driver.h"
#include "../../../runtime/Engine.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <memory>
#include <nvml.h>
#include <string>
#include <thread>
#include <vector>

// 编译:
// nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler -pthread \
//   -Xlinker=/lib/x86_64-linux-gnu/libnvidia-ml.so.1 bench_tiny_llama_load.cu -o /tmp/bench_tiny_llama_load

constexpr int NUM_LAYERS = 22, NH = 32, NKV = 4, HS = 64;
constexpr int HIDDEN = 2048, INTERMEDIATE = 5632, VOCAB = 32000;
constexpr int MAX_SEQS = 8, MAX_BATCHED_TOKENS = 512, MAX_SEQ_LEN = 2048;
constexpr int REQUESTS = 32, PROMPT_TOKENS = 128, NEW_TOKENS = 128;
constexpr float RMS_EPS = 1e-5f, ROPE_THETA = 10000.0f;
const std::string WEIGHT_PATH = "weight/weight/model.safetensors";

template<typename T>
class TinyLlamaBenchBlock {
public:
	TinyLlamaBenchBlock(LLM &llm, Engine &engine, Tensor *input, int layer)
	: input_norm_(llm, input, MAX_BATCHED_TOKENS, HIDDEN, RMS_EPS, prefix(layer) + ".input_layernorm")
	, self_attn_(llm, engine, input_norm_.out(), MAX_BATCHED_TOKENS, HIDDEN, layer, false, false, prefix(layer) + ".self_attn")
	, attention_residual_(llm, input, self_attn_.out(), MAX_BATCHED_TOKENS, HIDDEN)
	, post_attention_norm_(llm, attention_residual_.out(), MAX_BATCHED_TOKENS, HIDDEN, RMS_EPS, prefix(layer) + ".post_attention_layernorm")
	, mlp_(llm, post_attention_norm_.out(), MAX_BATCHED_TOKENS, HIDDEN, INTERMEDIATE, false, prefix(layer) + ".mlp")
	, mlp_residual_(llm, attention_residual_.out(), mlp_.out(), MAX_BATCHED_TOKENS, HIDDEN) {}

	Tensor *out() {return mlp_residual_.out();}

private:
	static std::string prefix(int layer) {return "model.layers." + std::to_string(layer);}
	RMSNorm<T> input_norm_;
	Attention<T> self_attn_;
	Residual<T> attention_residual_;
	RMSNorm<T> post_attention_norm_;
	MLP<T> mlp_;
	Residual<T> mlp_residual_;
};

struct NvmlStats {
	double power_sum = 0, gpu_util_sum = 0, memory_util_sum = 0, sm_clock_sum = 0;
	unsigned power_max = 0, gpu_util_max = 0, sm_clock_max = 0, samples = 0, power_limit = 0;
};

class NvmlSampler {
public:
	NvmlSampler() {
		nvmlInit();
		nvmlDeviceGetHandleByIndex(0, &device_);
		nvmlDeviceGetPowerManagementLimit(device_, &stats_.power_limit);
	}

	~NvmlSampler() {nvmlShutdown();}

	void start() {
		running_ = true;
		thread_ = std::thread([&] {
			while (running_) {
				unsigned power, sm_clock;
				nvmlUtilization_t utilization;
				nvmlDeviceGetPowerUsage(device_, &power);
				nvmlDeviceGetUtilizationRates(device_, &utilization);
				nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &sm_clock);
				stats_.power_sum += power;
				stats_.gpu_util_sum += utilization.gpu;
				stats_.memory_util_sum += utilization.memory;
				stats_.sm_clock_sum += sm_clock;
				stats_.power_max = std::max(stats_.power_max, power);
				stats_.gpu_util_max = std::max(stats_.gpu_util_max, utilization.gpu);
				stats_.sm_clock_max = std::max(stats_.sm_clock_max, sm_clock);
				++stats_.samples;
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
			}
		});
	}

	NvmlStats stop() {
		running_ = false;
		thread_.join();
		return stats_;
	}

private:
	nvmlDevice_t device_;
	std::atomic<bool> running_{false};
	std::thread thread_;
	NvmlStats stats_;
};

struct RunStats {
	double wall_ms, busy_ms, gap_ms;
	int finished, generated_tokens, model_rows, batch_entries, steps;
	NvmlStats nvml;
};

RunStats run_load(LLM &llm, Engine &engine, KV_Pool &pool, int requests, int prompt_tokens, int new_tokens, bool measure) {
	Scheduler scheduler(pool, {MAX_SEQS, MAX_BATCHED_TOKENS, /*eos=*/-1});
	LLM::InferenceBuffers buffers;
	buffers.input_stride = MAX_BATCHED_TOKENS;
	buffers.token_stride = MAX_SEQS;
	cudaHostAlloc(&buffers.h_inputs, 2 * buffers.input_stride * sizeof(int), 0);
	cudaMalloc(&buffers.d_tokens, 2 * buffers.token_stride * sizeof(int));
	cudaHostAlloc(&buffers.h_tokens, 2 * buffers.token_stride * sizeof(int), 0);
	for (int request = 0; request < requests; ++request) scheduler.add_request(std::vector<int>(prompt_tokens, 15043), new_tokens);

	int event_capacity = requests * (new_tokens + 2) + 16;
	std::vector<cudaEvent_t> starts(event_capacity), ends(event_capacity);
	for (int i = 0; i < event_capacity; ++i) {cudaEventCreate(&starts[i]); cudaEventCreate(&ends[i]);}
	int finished = 0, model_rows = 0, batch_entries = 0, steps = 0;
	auto launch = [&](const StepPlan &plan, int parity) {
		cudaEventRecord(starts[steps], llm.stream());
		llm.inference(plan, parity, engine, buffers);
		cudaEventRecord(ends[steps], llm.stream());
		model_rows += static_cast<int>(plan.ids.size());
		batch_entries += static_cast<int>(plan.req_ids.size());
		++steps;
	};
	auto on_finished = [&](std::vector<Request> &&done) {finished += static_cast<int>(done.size());};
	PipelineDriver driver(scheduler, buffers.h_tokens, buffers.token_stride, llm.stream(), launch, on_finished);
	NvmlSampler sampler;
	if (measure) {sampler.start(); cudaProfilerStart();}
	auto begin = std::chrono::steady_clock::now();
	driver.run_to_idle();
	auto end = std::chrono::steady_clock::now();
	if (measure) cudaProfilerStop();
	NvmlStats nvml = measure ? sampler.stop() : NvmlStats{};

	double busy_ms = 0, gap_ms = 0;
	for (int step = 0; step < steps; ++step) {
		float elapsed;
		cudaEventElapsedTime(&elapsed, starts[step], ends[step]);
		busy_ms += elapsed;
		if (step > 0) {cudaEventElapsedTime(&elapsed, ends[step - 1], starts[step]); gap_ms += elapsed;}
	}
	for (int i = 0; i < event_capacity; ++i) {cudaEventDestroy(starts[i]); cudaEventDestroy(ends[i]);}
	cudaFreeHost(buffers.h_inputs);
	cudaFree(buffers.d_tokens);
	cudaFreeHost(buffers.h_tokens);
	return {
		std::chrono::duration<double, std::milli>(end - begin).count(), busy_ms, gap_ms,
		finished, requests * new_tokens, model_rows, batch_entries, steps, nvml
	};
}

int main() {
	using T = __nv_bfloat16;
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device, 0);
	LLM llm(512);
	Engine engine(llm.arena(), NH, NKV, HS, MAX_SEQS, MAX_SEQ_LEN);
	Tensor *token_ids = llm.arena().alloc({MAX_BATCHED_TOKENS}, Dtype::I32);
	Embedding<T> embedding(llm, token_ids, MAX_BATCHED_TOKENS, VOCAB, HIDDEN, "model.embed_tokens");
	std::vector<std::unique_ptr<TinyLlamaBenchBlock<T>>> layers;
	layers.reserve(NUM_LAYERS);
	Tensor *hidden = embedding.out();
	for (int layer = 0; layer < NUM_LAYERS; ++layer) {
		layers.emplace_back(std::make_unique<TinyLlamaBenchBlock<T>>(llm, engine, hidden, layer));
		hidden = layers.back()->out();
	}
	RMSNorm<T> final_norm(llm, hidden, MAX_BATCHED_TOKENS, HIDDEN, RMS_EPS, "model.norm");
	LMHead<T> lm_head(llm, final_norm.out(), MAX_BATCHED_TOKENS, HIDDEN, VOCAB, "lm_head");
	llm.finalize();
	KVAlloc kv = llm.arena().alloc_kv_pool();
	KV_Pool pool(kv.k_base, kv.v_base, Dtype::BF16, NKV * HS, kv.bytes_each, NUM_LAYERS, MAX_SEQS, MAX_SEQ_LEN);
	engine.bind_pool(&pool);
	engine.init_rope(llm.stream(), ROPE_THETA);
	llm.load_safetensor(WEIGHT_PATH, [](const std::string &name) {
		return name == "lm_head.weight" || name.find("_proj.weight") != std::string::npos;
	});

	run_load(llm, engine, pool, MAX_SEQS, 16, 4, false);
	RunStats stats = run_load(llm, engine, pool, REQUESTS, PROMPT_TOKENS, NEW_TOKENS, true);
	double seconds = stats.wall_ms / 1000.0;
	double avg_batch = static_cast<double>(stats.batch_entries) / stats.steps;
	double gap_percent = 100.0 * stats.gap_ms / (stats.busy_ms + stats.gap_ms);
	double samples = stats.nvml.samples;
	std::printf("TinyLlama high-load benchmark\n");
	std::printf("  device=%s sms=%d\n", device.name, device.multiProcessorCount);
	std::printf("  requests=%d max_seqs=%d prompt=%d new_tokens=%d steps=%d avg_batch=%.2f\n", REQUESTS, MAX_SEQS, PROMPT_TOKENS, NEW_TOKENS, stats.steps, avg_batch);
	std::printf("  wall=%.3f s generated=%d throughput=%.2f tok/s model_rows=%d row_throughput=%.2f row/s\n", seconds, stats.generated_tokens, stats.generated_tokens / seconds, stats.model_rows, stats.model_rows / seconds);
	std::printf("  gpu_step_busy=%.3f ms gpu_step_gap=%.3f ms gap=%.3f%% finished=%d\n", stats.busy_ms / stats.steps, stats.gap_ms / (stats.steps - 1), gap_percent, stats.finished);
	std::printf("  power=%.2f W avg / %.2f W max / %.2f W limit\n", stats.nvml.power_sum / samples / 1000.0, stats.nvml.power_max / 1000.0, stats.nvml.power_limit / 1000.0);
	std::printf("  gpu_util=%.1f%% avg / %u%% max, memory_util=%.1f%% avg, sm_clock=%.0f MHz avg / %u MHz max\n", stats.nvml.gpu_util_sum / samples, stats.nvml.gpu_util_max, stats.nvml.memory_util_sum / samples, stats.nvml.sm_clock_sum / samples, stats.nvml.sm_clock_max);
	return 0;
}
