#include "../../model/modules/QKVLinear.h"
#include "../GraphShape.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

constexpr int NH = 2, NKV = 1, HS = 8;
constexpr int K = 16, QN = NH * HS, KN = NKV * HS, N = QN + 2 * KN;
constexpr int MAX_TOKENS = 19, MAX_SEQS = 4, MAX_SEQ_LEN = 64, NUM_BLOCKS = 12;

template<typename T>
struct Blob {
	std::string name;
	std::vector<int> shape;
	std::vector<T> data;
};

struct ErrorStats {
	float max_error = 0.0f;
	double error_sum = 0.0;
	int above = 0, count = 0;

	void add(float got, float want) {
		float error = std::fabs(got - want);
		if (error > max_error) max_error = error;
		error_sum += error;
		above += error > 2e-2f;
		count++;
	}
};

template<typename T>
static std::vector<T> make_values(int count, int seed) {
	std::vector<T> values(count);
	for (int i = 0; i < count; ++i) {
		int x = (i * 17 + seed * 11) % 31 - 15;
		values[i] = static_cast<T>(x * 0.0078125f);
	}
	return values;
}

static std::string shape_json(const std::vector<int> &shape) {
	std::string text = "[";
	for (size_t i = 0; i < shape.size(); ++i) {
		if (i) text += ",";
		text += std::to_string(shape[i]);
	}
	return text + "]";
}

// 写入测试专用 safetensors，保持 q/k/v checkpoint 为独立逻辑参数。
template<typename T>
static void write_safetensor(const std::string &path, const std::vector<Blob<T>> &blobs) {
	std::string header = "{";
	uint64_t offset = 0;
	for (size_t i = 0; i < blobs.size(); ++i) {
		uint64_t end = offset + blobs[i].data.size() * sizeof(T);
		if (i) header += ",";
		header += "\"" + blobs[i].name + "\":{\"dtype\":\"" +
			std::string(std::is_same<T, half>::value ? "F16" : "BF16") + "\",\"shape\":" +
			shape_json(blobs[i].shape) + ",\"data_offsets\":[" +
			std::to_string(offset) + "," + std::to_string(end) + "]}";
		offset = end;
	}
	header += "}";
	std::ofstream file(path, std::ios::binary);
	uint64_t header_size = header.size();
	for (int i = 0; i < 8; ++i) file.put(static_cast<char>((header_size >> (8 * i)) & 0xff));
	file.write(header.data(), static_cast<std::streamsize>(header.size()));
	for (const Blob<T> &blob : blobs)
		file.write(reinterpret_cast<const char *>(blob.data.data()),
			static_cast<std::streamsize>(blob.data.size() * sizeof(T)));
}

static int packed_column(int logical, int head_dim) {
	int head = logical / head_dim, d = logical % head_dim, half = head_dim / 2;
	return head * head_dim + (d < half ? 2 * d : 2 * (d - half) + 1);
}

// 验证 loader 已把转置后的 Q/K 按 RoPE 对重排，并写入唯一融合存储。
template<typename T>
static void check_packed_parameters(QKVLinear<T> &qkv,
	const std::vector<T> &qw, const std::vector<T> &kw, const std::vector<T> &vw,
	const std::vector<T> &qb, const std::vector<T> &kb, const std::vector<T> &vb, bool has_bias) {
	std::vector<T> weight(K * N);
	cudaMemcpy(weight.data(), qkv.weight()->ptr, qkv.weight()->bytes(), cudaMemcpyDeviceToHost);
	for (int k = 0; k < K; ++k) {
		for (int d = 0; d < QN; ++d)
			assert(static_cast<float>(weight[k * N + packed_column(d, HS)]) == static_cast<float>(qw[d * K + k]));
		for (int d = 0; d < KN; ++d)
			assert(static_cast<float>(weight[k * N + QN + packed_column(d, HS)]) == static_cast<float>(kw[d * K + k]));
		for (int d = 0; d < KN; ++d)
			assert(static_cast<float>(weight[k * N + QN + KN + d]) == static_cast<float>(vw[d * K + k]));
	}
	if (!has_bias) return;
	std::vector<T> bias(N);
	cudaMemcpy(bias.data(), qkv.bias()->ptr, qkv.bias()->bytes(), cudaMemcpyDeviceToHost);
	for (int d = 0; d < QN; ++d) assert(static_cast<float>(bias[packed_column(d, HS)]) == static_cast<float>(qb[d]));
	for (int d = 0; d < KN; ++d) assert(static_cast<float>(bias[QN + packed_column(d, HS)]) == static_cast<float>(kb[d]));
	for (int d = 0; d < KN; ++d) assert(static_cast<float>(bias[QN + KN + d]) == static_cast<float>(vb[d]));
}

template<typename T>
static float projection(const std::vector<T> &input, int row, const std::vector<T> &weight,
	const std::vector<T> &bias, int out, bool has_bias) {
	float sum = has_bias ? static_cast<float>(bias[out]) : 0.0f;
	for (int k = 0; k < K; ++k)
		sum += static_cast<float>(input[row * K + k]) * static_cast<float>(weight[out * K + k]);
	return static_cast<float>(static_cast<T>(sum));
}

// 用三份独立投影的 CPU 参考结果检查 Q 输出与分页 KV 写入。
template<typename T>
static void run_case(LLM &llm, Engine &engine, KV_Pool &pool, Tensor *input_tensor, QKVLinear<T> &qkv,
	const std::vector<int> &slots, const std::vector<int> &lens, ExecutionPhase phase, int seed,
	const std::vector<T> &qw, const std::vector<T> &kw, const std::vector<T> &vw,
	const std::vector<T> &qb, const std::vector<T> &kb, const std::vector<T> &vb,
	bool has_bias, ErrorStats &stats) {
	int B = static_cast<int>(slots.size()), M = 0;
	std::vector<int> starts(B);
	for (int b = 0; b < B; ++b) {
		starts[b] = pool.seq_len(slots[b]);
		M += lens[b];
	}
	std::vector<T> input = make_values<T>(M * K, seed);
	cudaMemcpyAsync(input_tensor->ptr, input.data(), input.size() * sizeof(T),
		cudaMemcpyHostToDevice, llm.stream());
	GraphShape shape = engine.prepare(slots.data(), B, lens.data(), llm.stream());
	assert(shape.batch == B && shape.total_tokens == M);

	std::vector<int> positions(M), destinations(M);
	for (int b = 0, t = 0; b < B; ++b)
		for (int i = 0; i < lens[b]; ++i, ++t) {
			positions[t] = starts[b] + i;
			destinations[t] = pool.physical_token(slots[b], positions[t]);
		}

	llm.forward(phase, shape);
	cudaStreamSynchronize(llm.stream());
	std::vector<T> got_q(M * QN);
	std::vector<T> got_k(NUM_BLOCKS * KV_BLOCK_SIZE * KN);
	std::vector<T> got_v(NUM_BLOCKS * KV_BLOCK_SIZE * KN);
	cudaMemcpy(got_q.data(), qkv.out()->ptr, got_q.size() * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(got_k.data(), pool.k_base(0), got_k.size() * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(got_v.data(), pool.v_base(0), got_v.size() * sizeof(T), cudaMemcpyDeviceToHost);

	for (int t = 0; t < M; ++t) {
		for (int h = 0; h < NH; ++h) {
			for (int i = 0; i < HS / 2; ++i) {
				int d0 = h * HS + i, d1 = d0 + HS / 2;
				float x0 = projection<T>(input, t, qw, qb, d0, has_bias);
				float x1 = projection<T>(input, t, qw, qb, d1, has_bias);
				float angle = positions[t] * std::pow(10000.0f, -2.0f * i / HS);
				float c = std::cos(angle), s = std::sin(angle);
				float want0 = static_cast<float>(static_cast<T>(x0 * c - x1 * s));
				float want1 = static_cast<float>(static_cast<T>(x0 * s + x1 * c));
				stats.add(static_cast<float>(got_q[t * QN + d0]), want0);
				stats.add(static_cast<float>(got_q[t * QN + d1]), want1);
			}
		}
		int target = destinations[t] * KN;
		for (int h = 0; h < NKV; ++h) {
			for (int i = 0; i < HS / 2; ++i) {
				int d0 = h * HS + i, d1 = d0 + HS / 2;
				float x0 = projection<T>(input, t, kw, kb, d0, has_bias);
				float x1 = projection<T>(input, t, kw, kb, d1, has_bias);
				float angle = positions[t] * std::pow(10000.0f, -2.0f * i / HS);
				float c = std::cos(angle), s = std::sin(angle);
				float want0 = static_cast<float>(static_cast<T>(x0 * c - x1 * s));
				float want1 = static_cast<float>(static_cast<T>(x0 * s + x1 * c));
				stats.add(static_cast<float>(got_k[target + d0]), want0);
				stats.add(static_cast<float>(got_k[target + d1]), want1);
			}
		}
		for (int d = 0; d < KN; ++d)
			stats.add(static_cast<float>(got_v[target + d]), projection<T>(input, t, vw, vb, d, has_bias));
	}
}

// 分别运行 bias/no-bias，覆盖小 M、Tensor Core 大 M 和同形状 Graph replay。
template<typename T>
static void run_suite(bool has_bias, ErrorStats &stats) {
	std::vector<T> qw = make_values<T>(QN * K, 1);
	std::vector<T> kw = make_values<T>(KN * K, 2);
	std::vector<T> vw = make_values<T>(KN * K, 3);
	std::vector<T> qb = make_values<T>(QN, 4);
	std::vector<T> kb = make_values<T>(KN, 5);
	std::vector<T> vb = make_values<T>(KN, 6);
	std::vector<Blob<T>> blobs = {
		{"attn.q_proj.weight", {QN, K}, qw},
		{"attn.k_proj.weight", {KN, K}, kw},
		{"attn.v_proj.weight", {KN, K}, vw}
	};
	if (has_bias) {
		blobs.push_back({"attn.q_proj.bias", {QN}, qb});
		blobs.push_back({"attn.k_proj.bias", {KN}, kb});
		blobs.push_back({"attn.v_proj.bias", {KN}, vb});
	}
	const char *type = std::is_same<T, half>::value ? "f16" : "bf16";
	std::string path = std::string("/tmp/test_qkv_") + type + (has_bias ? "_bias.safetensors" : "_no_bias.safetensors");
	write_safetensor(path, blobs);

	size_t kv_bytes = (size_t)NUM_BLOCKS * KV_BLOCK_SIZE * KN * sizeof(T);
	void *k_base, *v_base;
	cudaMalloc(&k_base, kv_bytes);
	cudaMalloc(&v_base, kv_bytes);
	{
		LLM llm(8);
		Engine engine(llm.arena(), NH, NKV, HS, MAX_SEQS, MAX_SEQ_LEN);
		Tensor *input = llm.arena().alloc({MAX_TOKENS, K}, dtype_of<T>::value);
		QKVLinear<T> qkv(llm, engine, input, MAX_TOKENS, K, 0, has_bias, "attn");
		assert(llm.parameters().size() == (has_bias ? 6 : 3));
		assert(llm.parameter_storages().size() == (has_bias ? 2 : 1));
		assert(llm.parameters().at("attn.q_proj.weight").tensor == qkv.weight());
		assert(llm.parameters().at("attn.k_proj.weight").tensor == qkv.weight());
		assert(llm.parameters().at("attn.v_proj.weight").tensor == qkv.weight());
		llm.finalize();
		KV_Pool pool(k_base, v_base, dtype_of<T>::value, KN, kv_bytes, 1, MAX_SEQS, MAX_SEQ_LEN);
		engine.bind_pool(&pool);
		engine.init_rope(llm.stream());
		llm.load_safetensor(path, [](const std::string &name) {
			return name.find(".weight") != std::string::npos;
		});
		cudaStreamSynchronize(llm.stream());
		check_packed_parameters(qkv, qw, kw, vw, qb, kb, vb, has_bias);
		cudaMemset(k_base, 0, kv_bytes);
		cudaMemset(v_base, 0, kv_bytes);

		int s0 = engine.alloc_seq(), s1 = engine.alloc_seq();
		pool.append(s0, 15);
		pool.append(s1, 17);
		run_case(llm, engine, pool, input, qkv, {s0, s1}, {3, 2}, ExecutionPhase::PREFILL,
			10, qw, kw, vw, qb, kb, vb, has_bias, stats);
		run_case(llm, engine, pool, input, qkv, {s0, s1}, {3, 2}, ExecutionPhase::PREFILL,
			11, qw, kw, vw, qb, kb, vb, has_bias, stats);

		int s2 = engine.alloc_seq(), s3 = engine.alloc_seq();
		run_case(llm, engine, pool, input, qkv, {s2, s3}, {9, 10}, ExecutionPhase::PREFILL,
			12, qw, kw, vw, qb, kb, vb, has_bias, stats);
		run_case(llm, engine, pool, input, qkv, {s2, s3}, {1, 1}, ExecutionPhase::DECODE,
			13, qw, kw, vw, qb, kb, vb, has_bias, stats);
		run_case(llm, engine, pool, input, qkv, {s2, s3}, {1, 1}, ExecutionPhase::DECODE,
			14, qw, kw, vw, qb, kb, vb, has_bias, stats);
		assert(llm.num_graphs(ExecutionPhase::PREFILL) == 2);
		assert(llm.num_graphs(ExecutionPhase::DECODE) == 1);
	}
	cudaFree(k_base);
	cudaFree(v_base);
	std::remove(path.c_str());
}

int main() {
	ErrorStats bf16, f16;
	run_suite<__nv_bfloat16>(false, bf16);
	run_suite<__nv_bfloat16>(true, bf16);
	run_suite<half>(false, f16);
	run_suite<half>(true, f16);
	assert(bf16.above == 0 && f16.above == 0);
	std::printf("test_qkv_linear PASS BF16 max=%.7f mean=%.7f count=%d; F16 max=%.7f mean=%.7f count=%d\n",
		bf16.max_error, bf16.error_sum / bf16.count, bf16.count,
		f16.max_error, f16.error_sum / f16.count, f16.count);
	return 0;
}
