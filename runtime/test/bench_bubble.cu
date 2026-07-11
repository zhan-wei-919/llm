#include "../Engine.h"
#include "../Scheduler.h"
#include "../Driver.h"
#include "../../tensor/Arena.cuh"
#include "../GraphShape.h"
#include <cstdio>
#include <vector>
#include <chrono>
#include <cassert>

// 手工性能基准: 测量 decode 闭环的"步间气泡". 不进默认测试.
//
// 时序骨架由 PipelineDriver 提供 (深度 1 投机流水: 发射 N+1 在前, 消费 N 在后,
// chunk 与 decode 同走流水, 抢占只在排空态放行, 先 flush 后 update).
// 本文件只提供"怎么算一步"(launch 闭包) 与观测 (计时/气泡/守恒断言).
// q/k/v 常驻 device (真实系统里激活来自上一层 GEMM, 不过 PCIe);
// token 经双份 device buffer 异步 D2H 回 pinned host 内存, event 标记完成.
//
// 测量口径:
//   每步一对 cudaEvent 括住整步 GPU 命令 (上传+scatter+attention+采样+D2H),
//   相邻步 end(N)→start(N+1) 的间隙就是真实气泡 —— 流水线成功的标志是它趋近于零.
//   CPU 各阶段计时里, "engine 调用"与 GPU 执行重叠, 不再计入气泡.
//
// 兼作正确性冒烟: 结尾断言块/槽位全额归还、全部请求完成、
// 批条目总数恰为 B*MAX_NEW (每请求 1 个 final chunk 条目 + MAX_NEW-1 个 decode 条目;
// 本配置预算=B*PROMPT_LEN, 每条 prompt 一步排完, 无中间 chunk) ——
// eos 关闭时停止全部可预测, 记账正确 ⟺ 投机浪费严格为零.
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
// 值本身无意义, 意义在于制造"下一步判停依赖本步 GPU 输出"的数据依赖.
// rows[b] 是第 b 条序列在打包输出里的最后一行 (它的 logits 行, 即段尾);
// chunk 与 decode 混装后行号不再恒等于 b, 统一由 host 按 lens 前缀和算好传入.
__global__ void fake_sample(int *tokens, const float *out, int qs, const int *rows) {
	int b = blockIdx.x;
	int row = rows ? rows[b] : b;
	tokens[b] = 100 + (int)(fabsf(out[(size_t)row * qs]) * 997.0f) % 1000;
}

static double us_since(std::chrono::steady_clock::time_point t0) {
	return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
	cudaStream_t t;
	cudaStreamCreate(&t);
	// ---------- 装配 ----------
	int W = (MAX_SEQ_LEN + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
	int L = W * KV_BLOCK_SIZE;
	int meta_ints = B * W + B + B + B * L + (B + 1);
	Arena a(/*max_tensors=*/2); // Engine 共享 cos/sin table
	int *d_meta;
	cudaMalloc(&d_meta, (size_t)meta_ints * sizeof(int));
	int *h_base;					// 2 份: Engine 按 decode 步的奇偶轮换写入
	cudaHostAlloc(&h_base, (size_t)2 * meta_ints * sizeof(int), 0);
	size_t kv_bytes = (size_t)BLOCKS * KV_BLOCK_SIZE * KS * sizeof(float);
	void *k_base, *v_base;
	cudaMalloc(&k_base, kv_bytes);
	cudaMalloc(&v_base, kv_bytes);
	KV_Pool pool(k_base, v_base, Dtype::F32, KS, kv_bytes, 1, B, MAX_SEQ_LEN);
	Engine eng(a, pool, NH, NKV, HS, d_meta, h_base);
	a.finalize();
	Scheduler sched(pool, {/*max_num_seqs=*/B, /*max_num_batched_tokens=*/B * PROMPT_LEN, /*eos=*/-1});

	// 激活常驻 device, 内容填一次伪随机即可 (attention 耗时与数值无关)
	size_t rows_cap = (size_t)B * PROMPT_LEN;	// prefill 打包行数是峰值
	float *dq, *dk, *dv, *dout;
	cudaMalloc(&dq,   rows_cap * QS * sizeof(float));
	cudaMalloc(&dk,   rows_cap * KS * sizeof(float));
	cudaMalloc(&dv,   rows_cap * KS * sizeof(float));
	cudaMalloc(&dout, rows_cap * QS * sizeof(float));
	fill_pattern<<<256, 256, 0, t>>>(dq, rows_cap * QS);
	fill_pattern<<<256, 256, 0, t>>>(dk, rows_cap * KS);
	fill_pattern<<<256, 256, 0, t>>>(dv, rows_cap * KS);
	// token 通路: 双份 device buffer + pinned host, 奇偶与 Engine 的元数据半区同步轮换
	int *d_tok, *h_tok;
	cudaMalloc(&d_tok, 2 * B * sizeof(int));
	cudaHostAlloc(&h_tok, 2 * B * sizeof(int), 0);
	// 采样行号 (每序列段尾): 与 token 同样双份轮换, host 侧 pinned 才能 async 上传
	int *d_rows, *h_rows;
	cudaMalloc(&d_rows, 2 * B * sizeof(int));
	cudaHostAlloc(&h_rows, 2 * B * sizeof(int), 0);
	for (int i = 0; i < B; ++i)
		sched.add_request(std::vector<int>(PROMPT_LEN, i), MAX_NEW);

	// ---------- 流水线主循环 (深度 1) ----------
	std::vector<cudaEvent_t> ev_start(MAX_NEW + 8), ev_end(MAX_NEW + 8);
	for (size_t i = 0; i < ev_start.size(); ++i) {
		cudaEventCreate(&ev_start[i]);
		cudaEventCreate(&ev_end[i]);
	}
	std::vector<double> t_sched, t_engine, t_wait, t_update;	// CPU 侧, 每 launched 步
	int step = 0, finished = 0, rows = 0;
	double dt_engine = 0;				// launch 闭包量的本步入队耗时
	// "怎么算一步": 段尾行号 staging + 全部 GPU 命令异步入队 (契约: 不做任何同步).
	// 段尾行号复用与 token 相同的双缓冲纪律 —— 同奇偶的上一次使用
	// 已在两步前被 event 确认消费, 此刻 host/device 半区都是空闲的.
	auto launch_fn = [&](const StepPlan &plan, int p) {
		int nb = (int)plan.req_ids.size();
		for (int b = 0, acc = 0; b < nb; ++b) { acc += plan.lens[b]; h_rows[p * B + b] = acc - 1; }
		auto c1 = std::chrono::steady_clock::now();
		cudaEventRecord(ev_start[step], t);
		cudaMemcpyAsync(d_rows + p * B, h_rows + p * B, nb * sizeof(int), cudaMemcpyHostToDevice, t);
		GraphShape shape = eng.prepare(plan.slots.data(), nb, plan.lens.data(), t);
		eng.forward_layer(0, dq, dk, dv, dout, t);
		fake_sample<<<nb, 1, 0, t>>>(d_tok + p * B, dout, QS, d_rows + p * B);
		cudaMemcpyAsync(h_tok + p * B, d_tok + p * B, nb * sizeof(int), cudaMemcpyDeviceToHost, t);
		cudaEventRecord(ev_end[step], t);
		dt_engine = us_since(c1);
		rows += nb;
	};
	auto on_done = [&](std::vector<Request> &&done) { finished += (int)done.size(); };
	PipelineDriver drv(sched, h_tok, B, t, launch_fn, on_done);
	auto wall0 = std::chrono::steady_clock::now();
	while (true) {
		PipelineDriver::Pump r = drv.pump();
		if (r == PipelineDriver::Pump::IDLE) break;
		if (r != PipelineDriver::Pump::LAUNCHED) continue;	// 排空拍不计时
		if (step >= WARMUP) {
			const auto &t = drv.last_times();
			t_sched.push_back(t.sched_us);
			t_engine.push_back(dt_engine);
			t_wait.push_back(t.wait_us);
			t_update.push_back(t.update_us);
		}
		++step;
	}
	double wall_ms = us_since(wall0) / 1000.0;
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

	// ---------- 守恒 + 零浪费断言 ----------
	assert(finished == B);
	assert((int)pool.num_free_blocks() == BLOCKS);	// 延迟释放漏一个 slot 就会在这里现形
	assert((int)pool.num_free_slots() == B);
	// 每请求恰 MAX_NEW 个产 token 条目 (1 个 final chunk + MAX_NEW-1 个 decode);
	// eos 关闭 ⇒ 停止全部可预测 ⇒ 零幽灵行
	assert(rows == B * MAX_NEW);

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
	printf("  schedule        %8.1f us   (投机组下一步计划)\n", avg(t_sched));
	printf("  engine 调用     %8.1f us   (与 GPU 执行重叠, 不再是气泡)\n", avg(t_engine));
	printf("  等上步 token    %8.1f us   (event 同步, 与本步 GPU 执行重叠)\n", avg(t_wait));
	printf("  update+flush    %8.1f us\n\n", avg(t_update));
	printf("GPU 时间线:\n");
	printf("  每步 busy       %8.1f us   (上传 + scatter + attention + 采样 + D2H)\n", busy / n);
	printf("  每步 gap        %8.1f us   (真实气泡: 流水线断流的时刻)\n", gap / (n - 1));
	printf("  气泡占比        %8.1f %%\n\n", 100.0 * gap / (gap + busy));
	printf("总耗时 %.1f ms, 生成吞吐 %.0f tok/s, 完成 %d 请求, 产 token 条目 %d (零浪费)\n",
	       wall_ms, (double)rows / (wall_ms / 1000.0), finished, rows);

	for (size_t i = 0; i < ev_start.size(); ++i) {
		cudaEventDestroy(ev_start[i]);
		cudaEventDestroy(ev_end[i]);
	}
	cudaFreeHost(h_base);
	cudaFreeHost(h_tok);
	cudaFreeHost(h_rows);
	cudaFree(k_base); cudaFree(v_base);
	cudaFree(d_meta);
	cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
	cudaFree(d_tok); cudaFree(d_rows);
	cudaStreamDestroy(t);
	return 0;
}
