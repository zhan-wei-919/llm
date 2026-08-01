#include "../Gemm.h"
#include "../QKVGemm.h"
#include "../../attention/Scatter_kv.h"
#include "../../embedding/RoPE.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct BenchShape {
	const char *name;
	int K, NH, NKV, HS;
};

struct Timing {
	float median, best;
};

__global__ void flush_l2_kernel(int *data, int count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count) data[i] ^= i;
}

static int packed_column(int logical, int HS) {
	int head = logical / HS, d = logical % HS, half = HS / 2;
	return head * HS + (d < half ? 2 * d : 2 * (d - half) + 1);
}

template<typename Launch>
static float time_hot_once(Launch launch, int iterations, cudaStream_t stream) {
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, stream);
	for (int i = 0; i < iterations; ++i) launch();
	cudaEventRecord(end, stream);
	cudaEventSynchronize(end);
	float ms;
	cudaEventElapsedTime(&ms, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return ms * 1000.0f / iterations;
}

template<typename Launch>
static float time_cold_once(Launch launch, int *flush, int flush_count, cudaStream_t stream) {
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	flush_l2_kernel<<<(flush_count + 255) / 256, 256, 0, stream>>>(flush, flush_count);
	cudaEventRecord(start, stream);
	launch();
	cudaEventRecord(end, stream);
	cudaEventSynchronize(end);
	float ms;
	cudaEventElapsedTime(&ms, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return ms * 1000.0f;
}

static Timing summarize(std::vector<float> values) {
	std::sort(values.begin(), values.end());
	return {values[values.size() / 2], values.front()};
}

// 对同一组输入交替测量旧路径和融合路径，避免时钟爬升偏向固定一方。
template<typename Baseline, typename Fused>
static void measure_pair(Baseline baseline, Fused fused, int iterations, int *flush, int flush_count,
	cudaStream_t stream, Timing &base_hot, Timing &fused_hot, Timing &base_cold, Timing &fused_cold) {
	for (int i = 0; i < 20; ++i) {
		baseline();
		fused();
	}
	cudaStreamSynchronize(stream);
	std::vector<float> bh, fh, bc, fc;
	for (int r = 0; r < 9; ++r) {
		if (r & 1) {
			fh.push_back(time_hot_once(fused, iterations, stream));
			bh.push_back(time_hot_once(baseline, iterations, stream));
			fc.push_back(time_cold_once(fused, flush, flush_count, stream));
			bc.push_back(time_cold_once(baseline, flush, flush_count, stream));
		} else {
			bh.push_back(time_hot_once(baseline, iterations, stream));
			fh.push_back(time_hot_once(fused, iterations, stream));
			bc.push_back(time_cold_once(baseline, flush, flush_count, stream));
			fc.push_back(time_cold_once(fused, flush, flush_count, stream));
		}
	}
	base_hot = summarize(bh);
	fused_hot = summarize(fh);
	base_cold = summarize(bc);
	fused_cold = summarize(fc);
}

// 构造与生产路径一致的页表、位置和 Q/K 配对权重，测量单层 QKV 投影全链路。
static void run_shape(const BenchShape &shape, int M, int *flush, int flush_count, cudaStream_t stream) {
	using T = __nv_bfloat16;
	using Config = GemmConfig<T, T>;
	int QN = shape.NH * shape.HS, KN = shape.NKV * shape.HS, N = QN + 2 * KN;
	int start_pos = 32, W = (start_pos + M + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
	size_t pool_elements = (size_t)W * KV_BLOCK_SIZE * KN;

	std::vector<T> h_a((size_t)M * shape.K);
	std::vector<T> h_wq((size_t)shape.K * QN);
	std::vector<T> h_wk((size_t)shape.K * KN);
	std::vector<T> h_wv((size_t)shape.K * KN);
	std::vector<T> h_w((size_t)shape.K * N);
	for (size_t i = 0; i < h_a.size(); ++i) h_a[i] = __float2bfloat16(((int)(i * 13 % 29) - 14) * 0.015625f);
	for (size_t i = 0; i < h_wq.size(); ++i) h_wq[i] = __float2bfloat16(((int)(i * 7 % 31) - 15) * 0.0078125f);
	for (size_t i = 0; i < h_wk.size(); ++i) h_wk[i] = __float2bfloat16(((int)(i * 11 % 31) - 15) * 0.0078125f);
	for (size_t i = 0; i < h_wv.size(); ++i) h_wv[i] = __float2bfloat16(((int)(i * 17 % 31) - 15) * 0.0078125f);
	for (int k = 0; k < shape.K; ++k) {
		for (int d = 0; d < QN; ++d) h_w[(size_t)k * N + packed_column(d, shape.HS)] = h_wq[(size_t)k * QN + d];
		for (int d = 0; d < KN; ++d) h_w[(size_t)k * N + QN + packed_column(d, shape.HS)] = h_wk[(size_t)k * KN + d];
		for (int d = 0; d < KN; ++d) h_w[(size_t)k * N + QN + KN + d] = h_wv[(size_t)k * KN + d];
	}

	std::vector<int> h_positions(M), h_dst(M), h_seq(M, 0), h_table(W), h_cu{0, M}, h_pos{start_pos};
	for (int b = 0; b < W; ++b) h_table[b] = W - 1 - b;
	for (int t = 0; t < M; ++t) {
		int p = start_pos + t, phys = h_table[p / KV_BLOCK_SIZE];
		h_positions[t] = p;
		h_dst[t] = phys * KV_BLOCK_SIZE + p % KV_BLOCK_SIZE;
	}
	int max_pos = start_pos + M;
	std::vector<float> h_cos((size_t)max_pos * shape.HS / 2), h_sin(h_cos.size());
	for (int p = 0; p < max_pos; ++p)
		for (int i = 0; i < shape.HS / 2; ++i) {
			float angle = p * std::pow(10000.0f, -2.0f * i / shape.HS);
			h_cos[(size_t)p * shape.HS / 2 + i] = std::cos(angle);
			h_sin[(size_t)p * shape.HS / 2 + i] = std::sin(angle);
		}

	T *a, *wq, *wk, *wv, *w, *q0, *k0, *v0, *q1, *kp0, *vp0, *kp1, *vp1;
	int *positions, *dst, *seq, *table, *cu, *pos;
	float *cos_table, *sin_table;
	cudaMalloc(&a, h_a.size() * sizeof(T));
	cudaMalloc(&wq, h_wq.size() * sizeof(T));
	cudaMalloc(&wk, h_wk.size() * sizeof(T));
	cudaMalloc(&wv, h_wv.size() * sizeof(T));
	cudaMalloc(&w, h_w.size() * sizeof(T));
	cudaMalloc(&q0, (size_t)M * QN * sizeof(T));
	cudaMalloc(&k0, (size_t)M * KN * sizeof(T));
	cudaMalloc(&v0, (size_t)M * KN * sizeof(T));
	cudaMalloc(&q1, (size_t)M * QN * sizeof(T));
	cudaMalloc(&kp0, pool_elements * sizeof(T));
	cudaMalloc(&vp0, pool_elements * sizeof(T));
	cudaMalloc(&kp1, pool_elements * sizeof(T));
	cudaMalloc(&vp1, pool_elements * sizeof(T));
	cudaMalloc(&positions, M * sizeof(int));
	cudaMalloc(&dst, M * sizeof(int));
	cudaMalloc(&seq, M * sizeof(int));
	cudaMalloc(&table, W * sizeof(int));
	cudaMalloc(&cu, 2 * sizeof(int));
	cudaMalloc(&pos, sizeof(int));
	cudaMalloc(&cos_table, h_cos.size() * sizeof(float));
	cudaMalloc(&sin_table, h_sin.size() * sizeof(float));
	cudaMemcpy(a, h_a.data(), h_a.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(wq, h_wq.data(), h_wq.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(wk, h_wk.data(), h_wk.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(wv, h_wv.data(), h_wv.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(w, h_w.data(), h_w.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(positions, h_positions.data(), M * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dst, h_dst.data(), M * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(seq, h_seq.data(), M * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(table, h_table.data(), W * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cu, h_cu.data(), 2 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pos, h_pos.data(), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cos_table, h_cos.data(), h_cos.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(sin_table, h_sin.data(), h_sin.size() * sizeof(float), cudaMemcpyHostToDevice);

	auto baseline = [&] {
		launch_Gemm_forward<Config>(a, wq, q0, nullptr, 1.0f, 0.0f, M, QN, shape.K, stream);
		launch_Gemm_forward<Config>(a, wk, k0, nullptr, 1.0f, 0.0f, M, KN, shape.K, stream);
		launch_Gemm_forward<Config>(a, wv, v0, nullptr, 1.0f, 0.0f, M, KN, shape.K, stream);
		launch_rope(q0, cos_table, sin_table, cu, seq, pos, M, shape.NH, shape.HS, stream);
		launch_rope(k0, cos_table, sin_table, cu, seq, pos, M, shape.NKV, shape.HS, stream);
		launch_scatter_kv(kp0, vp0, k0, v0, table, cu, seq, pos, M, shape.NKV, shape.HS, W, stream);
	};
	auto fused = [&] {
		launch_qkv_gemm_bf16(a, w, nullptr, q1, kp1, vp1, positions, dst, cos_table, sin_table,
			M, shape.K, shape.NH, shape.NKV, shape.HS, stream);
	};

	baseline();
	fused();
	cudaStreamSynchronize(stream);
	std::vector<T> h_q0((size_t)M * QN), h_q1(h_q0.size()), h_k0(pool_elements), h_k1(pool_elements);
	std::vector<T> h_v0(pool_elements), h_v1(pool_elements);
	cudaMemcpy(h_q0.data(), q0, h_q0.size() * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_q1.data(), q1, h_q1.size() * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_k0.data(), kp0, h_k0.size() * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_k1.data(), kp1, h_k1.size() * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_v0.data(), vp0, h_v0.size() * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_v1.data(), vp1, h_v1.size() * sizeof(T), cudaMemcpyDeviceToHost);
	float max_error = 0.0f;
	for (size_t i = 0; i < h_q0.size(); ++i)
		max_error = std::max(max_error, std::fabs(__bfloat162float(h_q0[i]) - __bfloat162float(h_q1[i])));
	for (int t = 0; t < M; ++t)
		for (int d = 0; d < KN; ++d) {
			int index = h_dst[t] * KN + d;
			max_error = std::max(max_error, std::fabs(__bfloat162float(h_k0[index]) - __bfloat162float(h_k1[index])));
			max_error = std::max(max_error, std::fabs(__bfloat162float(h_v0[index]) - __bfloat162float(h_v1[index])));
		}

	int iterations = M <= 8 ? 200 : 40;
	Timing bh, fh, bc, fc;
	measure_pair(baseline, fused, iterations, flush, flush_count, stream, bh, fh, bc, fc);
	std::printf("%s_M%d,hot,%.3f,%.3f,%.3f,%.7f\n",
		shape.name, M, bh.median, fh.median, bh.median / fh.median, max_error);
	std::printf("%s_M%d,cold,%.3f,%.3f,%.3f,%.7f\n",
		shape.name, M, bc.median, fc.median, bc.median / fc.median, max_error);

	cudaFree(a); cudaFree(wq); cudaFree(wk); cudaFree(wv); cudaFree(w);
	cudaFree(q0); cudaFree(k0); cudaFree(v0); cudaFree(q1);
	cudaFree(kp0); cudaFree(vp0); cudaFree(kp1); cudaFree(vp1);
	cudaFree(positions); cudaFree(dst); cudaFree(seq); cudaFree(table); cudaFree(cu); cudaFree(pos);
	cudaFree(cos_table); cudaFree(sin_table);
}

int main() {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	const int flush_count = 128 * 1024 * 1024 / sizeof(int);
	int *flush;
	cudaMalloc(&flush, (size_t)flush_count * sizeof(int));
	cudaMemset(flush, 0, (size_t)flush_count * sizeof(int));
	std::printf("shape,cache,baseline_us,fused_us,speedup,max_error\n");
	const BenchShape shapes[] = {
		{"mqa_h1024", 1024, 8, 1, 64},
		{"mha_h1024", 1024, 8, 8, 64},
		{"gqa_h2048", 2048, 32, 4, 64},
		{"gqa_h4096", 4096, 32, 8, 128}
	};
	const int rows[] = {1, 8, 127, 128, 129, 256, 384, 415, 416, 417, 512};
	for (const BenchShape &shape : shapes)
		for (int M : rows) run_shape(shape, M, flush, flush_count, stream);
	cudaFree(flush);
	cudaStreamDestroy(stream);
	return 0;
}
