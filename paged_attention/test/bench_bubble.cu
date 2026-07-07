#include "../Engine.cuh"
#include "../Scheduler.h"
#include "../tensor/Arena.cuh"
#include <cstdio>
#include <vector>
#include <chrono>

// 手工性能基准: 测量 decode 闭环的"步间气泡". 只计时, 不验正确性, 不进默认测试.
//
// 复现真实推理闭环的同步结构:
//   schedule → Engine::decode(元数据同步H2D + kernel) → GPU伪采样 → token同步D2H → update
// 与 test_scheduler_gpu 的区别: q/k/v 常驻 device (真实系统里激活来自上一层 GEMM,
// 不过 PCIe), token 回读是闭环里唯一的 D2H, 且主循环真正依赖它判停 —— 这是气泡的来源.
//
// 测量口径:
//   GPU 侧: 每步用一对 cudaEvent 括住 [元数据上传 + scatter + attention + 采样],
//           相邻步 end(N)→start(N+1) 的间隙就是 GPU 空转的气泡.
//   CPU 侧: 每步四个阶段分别计时, 解释气泡由什么构成.
//
// 编译: nvcc -O2 -arch=native bench_bubble.cu -o /tmp/bench_bubble

constexpr int NH = 32, NKV = 8, HS = 128;		// LLaMA 尺寸的注意力头配置
constexpr int QS = NH * HS, KS = NKV * HS;		// 4096 / 1024
constexpr int B = 64;					// 满 batch decode
constexpr int PROMPT_LEN = 128, MAX_NEW = 384;		// len 128 → 512
constexpr int MAX_SEQ_LEN = PROMPT_LEN + MAX_NEW;
constexpr int WARMUP = 16;				// 前几步不计入统计
constexpr int BLOCKS = B * MAX_SEQ_LEN / KV_BLOCK_SIZE + 256;	// 够用+余量, 不触发抢占

__global__ void fill_pattern(float *p, size_t n) {
	for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
		p[i] = (float)((i * 2654435761u) & 0xFFFF) / 32768.0f - 1.0f;
}

// 伪采样: 从 attention 输出算一个确定性 token.
// 值本身无意义, 意义在于制造"下一步调度依赖本步 GPU 输出"的数据依赖.
__global__ void fake_sample(int *tokens, const float *out, int qs) {
	int b = blockIdx.x;
	tokens[b] = 100 + (int)(fabsf(out[(size_t)b * qs]) * 997.0f) % 1000;
}

static double us_since(std::chrono::steady_clock::time_point t0) {
	return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
	// ---------- 装配 ----------
	int W = (MAX_SEQ_LEN + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
	int L = W * KV_BLOCK_SIZE;
	int meta_ints = B * W + B + B + B * L + (B + 1);
	Arena a(4);
	Tensor *t_meta = a.alloc({meta_ints}, Dtype::F32);
	a.finalize();
	int *h_base;
	cudaHostAlloc(&h_base, meta_ints * sizeof(int), 0);
	size_t kv_bytes = (size_t)BLOCKS * KV_BLOCK_SIZE * KS * sizeof(float);
	void *k_base, *v_base;
	cudaMalloc(&k_base, kv_bytes);
	cudaMalloc(&v_base, kv_bytes);
	KV_Pool pool(k_base, v_base, Dtype::F32, KS, kv_bytes, B, MAX_SEQ_LEN);
	Engine eng(pool, NH, NKV, HS, (int *)t_meta->ptr, h_base);
	Scheduler sched(pool, {/*max_num_seqs=*/B, /*max_num_batched_tokens=*/B * PROMPT_LEN, /*eos=*/-1});

	// 激活常驻 device, 内容填一次伪随机即可 (attention 耗时与数值无关)
	size_t rows = (size_t)B * PROMPT_LEN;		// prefill 打包行数是峰值
	float *dq, *dk, *dv, *dout;
	cudaMalloc(&dq,   rows * QS * sizeof(float));
	cudaMalloc(&dk,   rows * KS * sizeof(float));
	cudaMalloc(&dv,   rows * KS * sizeof(float));
	cudaMalloc(&dout, rows * QS * sizeof(float));
	fill_pattern<<<256, 256>>>(dq, rows * QS);
	fill_pattern<<<256, 256>>>(dk, rows * KS);
	fill_pattern<<<256, 256>>>(dv, rows * KS);
	int *d_tokens, h_tokens[B];			// 基线故意用 pageable host 内存
	cudaMalloc(&d_tokens, B * sizeof(int));

	for (int i = 0; i < B; ++i)
		sched.add_request(std::vector<int>(PROMPT_LEN, i), MAX_NEW);

	// ---------- 主循环 ----------
	std::vector<cudaEvent_t> ev_start(MAX_NEW + 4), ev_end(MAX_NEW + 4);
	for (size_t i = 0; i < ev_start.size(); ++i) {
		cudaEventCreate(&ev_start[i]);
		cudaEventCreate(&ev_end[i]);
	}
	std::vector<double> t_sched, t_engine, t_wait, t_update;	// CPU 侧, 每 decode 步
	int step = 0;						// decode 步计数
	auto wall0 = std::chrono::steady_clock::now();
	while (true) {
		auto c0 = std::chrono::steady_clock::now();
		StepPlan plan = sched.schedule();
		if (plan.empty()) break;
		double dt_sched = us_since(c0);
		int nb = (int)plan.req_ids.size();

		if (plan.is_prefill) {				// 只为填 KV, 不计时
			eng.prefill(plan.slots.data(), nb, plan.lens.data(), dq, dk, dv, dout);
			fake_sample<<<nb, 1>>>(d_tokens, dout, QS);
			cudaMemcpy(h_tokens, d_tokens, nb * sizeof(int), cudaMemcpyDeviceToHost);
		} else {
			auto c1 = std::chrono::steady_clock::now();
			cudaEventRecord(ev_start[step]);
			eng.decode(plan.slots.data(), nb, dq, dk, dv, dout);
			fake_sample<<<nb, 1>>>(d_tokens, dout, QS);
			cudaEventRecord(ev_end[step]);
			double dt_engine = us_since(c1);

			auto c2 = std::chrono::steady_clock::now();
			cudaMemcpy(h_tokens, d_tokens, nb * sizeof(int), cudaMemcpyDeviceToHost);	// 同步点
			double dt_wait = us_since(c2);
			if (step >= WARMUP) {
				t_sched.push_back(dt_sched);
				t_engine.push_back(dt_engine);
				t_wait.push_back(dt_wait);
			}
			++step;
		}
		auto c3 = std::chrono::steady_clock::now();
		sched.update(plan, std::vector<int>(h_tokens, h_tokens + nb));
		if (!plan.is_prefill && step > WARMUP) t_update.push_back(us_since(c3));
	}
	double wall_ms = us_since(wall0) / 1000.0;
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

	// ---------- 汇总 ----------
	double busy = 0, gap = 0;
	int n = 0;
	for (int i = WARMUP; i < step; ++i) {
		float ms;
		cudaEventElapsedTime(&ms, ev_start[i], ev_end[i]);
		busy += ms * 1000.0;
		if (i > WARMUP) {
			cudaEventElapsedTime(&ms, ev_end[i - 1], ev_start[i]);
			gap += ms * 1000.0;
		}
		++n;
	}
	auto avg = [](const std::vector<double> &v) {
		double s = 0;
		for (double x : v) s += x;
		return v.empty() ? 0.0 : s / v.size();
	};
	printf("配置: B=%d NH=%d NKV=%d HS=%d, len %d→%d, 统计 %d 步 (跳过前 %d 步)\n\n",
	       B, NH, NKV, HS, PROMPT_LEN, MAX_SEQ_LEN, n, WARMUP);
	printf("CPU 侧每步分解:\n");
	printf("  schedule        %8.1f us\n", avg(t_sched));
	printf("  engine 调用     %8.1f us   (记账 + gather + 异步H2D + launch)\n", avg(t_engine));
	printf("  等 token D2H    %8.1f us   (≈ 纯 kernel 执行时间)\n", avg(t_wait));
	printf("  update          %8.1f us\n\n", avg(t_update));
	// 口径说明: ev_start 记在 eng.decode 之前, 而 decode 里上传+launch 那段时间
	// GPU 只有几 KB 的拷贝在跑, 计算引擎是闲的 —— 事件括出的 busy 混入了这段空转.
	// 真实气泡 ≈ 步间 gap(update+schedule 期间) + engine 调用全程(上一步已被 D2H 排干,
	// kernel 尚未 launch). 纯 kernel 时间 ≈ 等 token D2H 的阻塞时长.
	double bubble = gap / (n - 1) + avg(t_engine);
	double step_us = busy / n + gap / (n - 1);
	printf("GPU 时间线:\n");
	printf("  每步总时长      %8.1f us\n", step_us);
	printf("  纯 kernel       %8.1f us\n", avg(t_wait));
	printf("  气泡 (估)       %8.1f us   (gap %.1f + engine 调用 %.1f)\n",
	       bubble, gap / (n - 1), avg(t_engine));
	printf("  气泡占比        %8.1f %%\n\n", 100.0 * bubble / step_us);
	printf("总耗时 %.1f ms, decode 吞吐 %.0f tok/s\n",
	       wall_ms, (double)B * step / (wall_ms / 1000.0));

	for (size_t i = 0; i < ev_start.size(); ++i) {
		cudaEventDestroy(ev_start[i]);
		cudaEventDestroy(ev_end[i]);
	}
	cudaFreeHost(h_base);
	cudaFree(k_base); cudaFree(v_base);
	cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
	cudaFree(d_tokens);
	return 0;
}
