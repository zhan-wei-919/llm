#include "../Engine.cuh"
#include "../Scheduler.h"
#include "../tensor/Arena.cuh"
#include <cstdio>
#include <vector>
#include <chrono>
#include <cassert>

// 手工性能基准: 测量 decode 闭环的"步间气泡". 不进默认测试.
//
// 主循环是深度 1 的软件流水线 (TODO 2 异步调度):
//   发射第 N+1 步在前, 消费第 N 步的 token 在后 —— CPU 永远比 GPU 超前一步,
//   GPU 队列不见底. 判停晚一拍生效, 物理块延迟归还 (flush_release 时机见循环内注释).
//   prefill 和抢占走同步屏障 (先排干在途步).
// q/k/v 常驻 device (真实系统里激活来自上一层 GEMM, 不过 PCIe);
// token 经双份 device buffer 异步 D2H 回 pinned host 内存, event 标记完成.
//
// 测量口径:
//   每步一对 cudaEvent 括住整步 GPU 命令 (上传+scatter+attention+采样+D2H),
//   相邻步 end(N)→start(N+1) 的间隙就是真实气泡 —— 流水线成功的标志是它趋近于零.
//   CPU 各阶段计时里, "engine 调用"与 GPU 执行重叠, 不再计入气泡.
//
// 兼作正确性冒烟: 结尾断言块/槽位全额归还、全部请求完成、
// decode 总行数恰为 B*(MAX_NEW-1) —— eos 关闭时 max_new 可精确预测,
// in_flight 记账正确 ⟺ 投机浪费严格为零.
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
	int *h_base;					// 2 份: Engine 按 decode 步的奇偶轮换写入
	cudaHostAlloc(&h_base, (size_t)2 * meta_ints * sizeof(int), 0);
	size_t kv_bytes = (size_t)BLOCKS * KV_BLOCK_SIZE * KS * sizeof(float);
	void *k_base, *v_base;
	cudaMalloc(&k_base, kv_bytes);
	cudaMalloc(&v_base, kv_bytes);
	KV_Pool pool(k_base, v_base, Dtype::F32, KS, kv_bytes, B, MAX_SEQ_LEN);
	Engine eng(pool, NH, NKV, HS, (int *)t_meta->ptr, h_base);
	Scheduler sched(pool, {/*max_num_seqs=*/B, /*max_num_batched_tokens=*/B * PROMPT_LEN, /*eos=*/-1});

	// 激活常驻 device, 内容填一次伪随机即可 (attention 耗时与数值无关)
	size_t rows_cap = (size_t)B * PROMPT_LEN;	// prefill 打包行数是峰值
	float *dq, *dk, *dv, *dout;
	cudaMalloc(&dq,   rows_cap * QS * sizeof(float));
	cudaMalloc(&dk,   rows_cap * KS * sizeof(float));
	cudaMalloc(&dv,   rows_cap * KS * sizeof(float));
	cudaMalloc(&dout, rows_cap * QS * sizeof(float));
	fill_pattern<<<256, 256>>>(dq, rows_cap * QS);
	fill_pattern<<<256, 256>>>(dk, rows_cap * KS);
	fill_pattern<<<256, 256>>>(dv, rows_cap * KS);
	// token 通路: 双份 device buffer + pinned host, 奇偶与 Engine 的元数据半区同步轮换
	int *d_tok, *h_tok;
	cudaMalloc(&d_tok, 2 * B * sizeof(int));
	cudaHostAlloc(&h_tok, 2 * B * sizeof(int), 0);
	cudaEvent_t ev_tok[2];
	cudaEventCreate(&ev_tok[0]);
	cudaEventCreate(&ev_tok[1]);

	for (int i = 0; i < B; ++i)
		sched.add_request(std::vector<int>(PROMPT_LEN, i), MAX_NEW);

	// ---------- 流水线主循环 (深度 1) ----------
	std::vector<cudaEvent_t> ev_start(MAX_NEW + 8), ev_end(MAX_NEW + 8);
	for (size_t i = 0; i < ev_start.size(); ++i) {
		cudaEventCreate(&ev_start[i]);
		cudaEventCreate(&ev_end[i]);
	}
	std::vector<double> t_sched, t_engine, t_wait, t_update;	// CPU 侧, 每 decode 步
	StepPlan inflight;				// 已发射未消费的 decode 步; 空 = 流水线排空
	int calls = 0;					// decode 发射数, 奇偶镜像 Engine 内部的 step_
	int step = 0, finished = 0, rows = 0;
	// 排干: 等在途步的 token → 消费 → 全量归还 (屏障下无任何在途引用)
	auto drain = [&]() {
		if (inflight.empty()) return;
		int q = (calls - 1) & 1;		// 在途步的奇偶
		cudaEventSynchronize(ev_tok[q]);
		finished += (int)sched.update(inflight,
			std::vector<int>(h_tok + q * B, h_tok + q * B + inflight.req_ids.size())).size();
		sched.flush_release();
		inflight = StepPlan{};
	};
	auto wall0 = std::chrono::steady_clock::now();
	while (true) {
		auto c0 = std::chrono::steady_clock::now();
		// 投机调度: 在上一步 token 未回来时组下一步计划 (赌无人 EOS).
		// 有在途步时禁止抢占 —— 立即 release 会撕碎在途步引用的块.
		StepPlan plan = sched.schedule(/*may_preempt=*/inflight.empty());
		double dt_sched = us_since(c0);

		if (plan.is_prefill) {			// 屏障路径: 排干再执行, 不计时
			drain();
			int nb = (int)plan.req_ids.size();
			eng.prefill(plan.slots.data(), nb, plan.lens.data(), dq, dk, dv, dout);
			fake_sample<<<nb, 1>>>(d_tok, dout, QS);
			cudaMemcpy(h_tok, d_tok, nb * sizeof(int), cudaMemcpyDeviceToHost);
			finished += (int)sched.update(plan, std::vector<int>(h_tok, h_tok + nb)).size();
			sched.flush_release();
			continue;
		}
		if (plan.empty()) {
			// 可能是需要抢占/全员等待在途 token: 排干后重试; 真空转说明全部完成
			if (!inflight.empty()) { drain(); continue; }
			break;
		}

		// 发射第 N+1 步 (第 N 步 = inflight 还在 GPU 上跑)
		int nb = (int)plan.req_ids.size();
		int p = calls++ & 1;
		auto c1 = std::chrono::steady_clock::now();
		cudaEventRecord(ev_start[step]);
		eng.decode(plan.slots.data(), nb, dq, dk, dv, dout);
		fake_sample<<<nb, 1>>>(d_tok + p * B, dout, QS);
		cudaMemcpyAsync(h_tok + p * B, d_tok + p * B, nb * sizeof(int), cudaMemcpyDeviceToHost);
		cudaEventRecord(ev_tok[p]);
		cudaEventRecord(ev_end[step]);
		double dt_engine = us_since(c1);

		// 消费第 N 步: 只等它的 token event, 此刻 GPU 正在跑第 N+1 步
		double dt_wait = 0, dt_update = 0;
		if (!inflight.empty()) {
			int q = 1 - p;
			auto c2 = std::chrono::steady_clock::now();
			cudaEventSynchronize(ev_tok[q]);
			dt_wait = us_since(c2);
			auto c3 = std::chrono::steady_clock::now();
			// 先 flush 后 update, 顺序即不变量: 队列里躺着第 N-1 步的死者,
			// 它们最后被第 N 步引用, 而第 N 步刚被 event 确认完成 → 归还安全.
			// update(N) 的新死者还被在途的第 N+1 步引用着, 留到下一拍.
			sched.flush_release();
			finished += (int)sched.update(inflight,
				std::vector<int>(h_tok + q * B, h_tok + q * B + inflight.req_ids.size())).size();
			dt_update = us_since(c3);
		}
		inflight = plan;
		rows += nb;
		if (step >= WARMUP) {
			t_sched.push_back(dt_sched);
			t_engine.push_back(dt_engine);
			t_wait.push_back(dt_wait);
			t_update.push_back(dt_update);
		}
		++step;
	}
	drain();					// 循环因 plan 空退出时 inflight 必已空, 兜底而已
	double wall_ms = us_since(wall0) / 1000.0;
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

	// ---------- 守恒 + 零浪费断言 ----------
	assert(finished == B);
	assert((int)pool.num_free_blocks() == BLOCKS);	// 延迟释放漏一个 slot 就会在这里现形
	assert((int)pool.num_free_slots() == B);
	assert(rows == B * (MAX_NEW - 1));		// eos 关闭 ⇒ 停止全部可预测 ⇒ 零幽灵行

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
	printf("总耗时 %.1f ms, decode 吞吐 %.0f tok/s, 完成 %d 请求, decode 行数 %d (零浪费)\n",
	       wall_ms, (double)rows / (wall_ms / 1000.0), finished, rows);

	for (size_t i = 0; i < ev_start.size(); ++i) {
		cudaEventDestroy(ev_start[i]);
		cudaEventDestroy(ev_end[i]);
	}
	cudaEventDestroy(ev_tok[0]);
	cudaEventDestroy(ev_tok[1]);
	cudaFreeHost(h_base);
	cudaFreeHost(h_tok);
	cudaFree(k_base); cudaFree(v_base);
	cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
	cudaFree(d_tok);
	return 0;
}
