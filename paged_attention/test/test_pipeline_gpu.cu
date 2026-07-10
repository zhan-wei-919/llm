// chunked prefill 异步流水线的 GPU 实机差分测试.
//
// 同一组请求跑两遍, 断言逐 token 相等:
//   oracle 驱动:  全同步屏障 (每步 schedule → 执行 → 等完 → update), 时序上不可能错;
//   流水线驱动:  深度 1 投机 (与 bench_bubble 同构): 发射 N+1 在前, 消费 N 在后,
//                chunk 与 decode 同走流水, 无任何屏障.
//
// q/k/v 按 (请求, 位置) 内容寻址地现场生成, 与批内打包位置无关 —— 这是差分成立的前提:
// 两种驱动的批打包合法地不同 (流水线下晋升晚一拍), 但任一 token 的值只取决于
// 它自己的 q 和它序列的 KV 前缀. 于是 token 流相等 ⟺ scatter 写入位置、页表、
// attention 前缀读取、双缓冲轮换在流水线下全部正确.
//
// 配置故意制造边界: 预算 128 强制切 chunk (300→128/128/44), 129→128/1 (单 token final),
// 256 追平恰踩块边界 (pending_boundary 路径), 33/47 单 chunk 短请求穿插出混合批.
// 块给足, 不触发抢占 (抢占时机在两种驱动下合法不同, 会让差分失义).
//
// 编译: nvcc -O2 -arch=native test_pipeline_gpu.cu -o /tmp/test_pipeline_gpu
#include "../Engine.cuh"
#include "../Scheduler.h"
#include <cstdio>
#include <vector>
#include <cassert>

constexpr int NH = 32, NKV = 8, HS = 128;	// 与 bench 相同的注意力配置
constexpr int QS = NH * HS, KS = NKV * HS;
constexpr int MAX_SEQS = 8;			// pool 槽位 = 调度座位, >= NREQ 避免座位差分
constexpr int MAX_SEQ_LEN = 512;
constexpr int BUDGET = 128;			// 单步 token 预算, 也是 chunk 粒度
constexpr int BLOCKS = 256;			// 充裕: 抢占在两种驱动下时机不同, 必须避免
constexpr int NREQ = 6;
const int PROMPT_LEN[NREQ] = {300, 128, 47, 256, 129, 33};
const int MAX_NEW[NREQ]    = {20,  5,   33, 11,  3,   8};

// 内容寻址的确定性数值: 只由 (盐, 位置, 维度) 决定, 与批打包无关
__device__ __forceinline__ float hashf(unsigned a, unsigned b, unsigned c) {
	unsigned h = a * 1315423911u ^ b * 2654435761u ^ c * 2246822519u;
	h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
	return (float)(h & 0xFFFF) / 32768.0f - 1.0f;
}

// 每行一个 block: 按该行所属的 (请求 rid, 序列位置 pos) 填 q/k/v
__global__ void fill_qkv(float *q, float *k, float *v, const int *rid, const int *pos) {
	int t = blockIdx.x;
	unsigned r = (unsigned)rid[t], p = (unsigned)pos[t];
	for (int j = threadIdx.x; j < QS; j += blockDim.x)
		q[(size_t)t * QS + j] = hashf(r * 3u + 0u, p, j);
	for (int j = threadIdx.x; j < KS; j += blockDim.x) {
		k[(size_t)t * KS + j] = hashf(r * 3u + 1u, p, j);
		v[(size_t)t * KS + j] = hashf(r * 3u + 2u, p, j);
	}
}

// 伪采样: 与 bench 相同, token = f(该序列段尾行的 attention 输出)
__global__ void fake_sample(int *tokens, const float *out, int qs, const int *rows) {
	int b = blockIdx.x;
	tokens[b] = 100 + (int)(fabsf(out[(size_t)rows[b] * qs]) * 997.0f) % 1000;
}

struct Ctx {		// 两次 run 共用的 device/pinned 资源, 每次 run 重建 pool/engine/scheduler
	void *k_base, *v_base;
	size_t kv_bytes;
	int *d_meta, *h_meta;
	float *dq, *dk, *dv, *dout;
	int *d_rid, *h_rid, *d_pos, *h_pos, *d_rows, *h_rows;	// 每行/每条目元数据, 双份轮换
	int *d_tok, *h_tok;
};

// 组一步的输入: 按 plan 填每行的 (rid, pos) 与每条目的段尾行号.
// prog[] 是 driver 侧的排定进度镜像 —— 它只随发射推进, 天然与 update 时序无关.
static int stage_inputs(const StepPlan &plan, std::vector<int> &prog,
                        int *h_rid, int *h_pos, int *h_rows) {
	int total = 0;
	for (size_t b = 0; b < plan.req_ids.size(); ++b) {
		int id = plan.req_ids[b];
		for (int i = 0; i < plan.lens[b]; ++i) {
			h_rid[total + i] = id;
			h_pos[total + i] = prog[id] + i;
		}
		prog[id] += plan.lens[b];
		total += plan.lens[b];
		h_rows[b] = total - 1;
	}
	return total;
}

// 发射一步的全部 GPU 命令 (不含任何同步), off 是双缓冲半区起点
static void launch_step(Ctx &c, Engine &eng, const StepPlan &plan,
                        int total, int off_rows, int off_tok) {
	int nb = (int)plan.req_ids.size();
	cudaMemcpyAsync(c.d_rid + off_rows, c.h_rid + off_rows, total * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(c.d_pos + off_rows, c.h_pos + off_rows, total * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(c.d_rows + off_tok, c.h_rows + off_tok, nb * sizeof(int), cudaMemcpyHostToDevice);
	fill_qkv<<<total, 128>>>(c.dq, c.dk, c.dv, c.d_rid + off_rows, c.d_pos + off_rows);
	eng.forward(plan.slots.data(), nb, plan.lens.data(), c.dq, c.dk, c.dv, c.dout);
	fake_sample<<<nb, 1>>>(c.d_tok + off_tok, c.dout, QS, c.d_rows + off_tok);
	cudaMemcpyAsync(c.h_tok + off_tok, c.d_tok + off_tok, nb * sizeof(int), cudaMemcpyDeviceToHost);
}

static void collect(std::vector<std::vector<int>> &gen, std::vector<Request> &&done) {
	for (auto &r : done)
		gen[r.id] = std::vector<int>(r.token_ids.begin() + r.prompt_len, r.token_ids.end());
}

// oracle: 全同步屏障驱动
static std::vector<std::vector<int>> run_sync(Ctx &c) {
	KV_Pool pool(c.k_base, c.v_base, Dtype::F32, KS, c.kv_bytes, MAX_SEQS, MAX_SEQ_LEN);
	Engine eng(pool, NH, NKV, HS, c.d_meta, c.h_meta);
	Scheduler sched(pool, {MAX_SEQS, BUDGET, /*eos=*/-1});
	for (int i = 0; i < NREQ; ++i)
		sched.add_request(std::vector<int>(PROMPT_LEN[i], i), MAX_NEW[i]);
	std::vector<std::vector<int>> gen(NREQ);
	std::vector<int> prog(NREQ, 0);
	while (true) {
		StepPlan plan = sched.schedule();
		if (plan.empty()) break;		// 同步下无在途, 空 plan = 全部完成
		int nb = (int)plan.req_ids.size();
		int total = stage_inputs(plan, prog, c.h_rid, c.h_pos, c.h_rows);
		launch_step(c, eng, plan, total, 0, 0);
		cudaDeviceSynchronize();		// 屏障: 每步等到底再结算
		collect(gen, sched.update(plan, std::vector<int>(c.h_tok, c.h_tok + nb)));
		sched.flush_release();			// 无在途引用, 立即归还安全
	}
	assert(sched.num_preemptions() == 0);
	sched.flush_release();
	assert((int)pool.num_free_blocks() == BLOCKS && (int)pool.num_free_slots() == MAX_SEQS);
	return gen;
}

// 被测: 深度 1 投机流水驱动 (与 bench_bubble 主循环同构)
static std::vector<std::vector<int>> run_pipelined(Ctx &c) {
	KV_Pool pool(c.k_base, c.v_base, Dtype::F32, KS, c.kv_bytes, MAX_SEQS, MAX_SEQ_LEN);
	Engine eng(pool, NH, NKV, HS, c.d_meta, c.h_meta);
	Scheduler sched(pool, {MAX_SEQS, BUDGET, /*eos=*/-1});
	for (int i = 0; i < NREQ; ++i)
		sched.add_request(std::vector<int>(PROMPT_LEN[i], i), MAX_NEW[i]);
	std::vector<std::vector<int>> gen(NREQ);
	std::vector<int> prog(NREQ, 0);
	cudaEvent_t ev_tok[2];
	cudaEventCreate(&ev_tok[0]);
	cudaEventCreate(&ev_tok[1]);
	StepPlan inflight;
	int calls = 0;
	auto drain = [&]() {
		if (inflight.empty()) return;
		int q = (calls - 1) & 1;
		cudaEventSynchronize(ev_tok[q]);
		collect(gen, sched.update(inflight, std::vector<int>(
			c.h_tok + q * MAX_SEQS, c.h_tok + q * MAX_SEQS + inflight.req_ids.size())));
		sched.flush_release();
		inflight = StepPlan{};
	};
	while (true) {
		// 投机: 上一步 token 未回就组计划; 有在途步时禁止抢占
		StepPlan plan = sched.schedule(/*may_preempt=*/inflight.empty());
		if (plan.empty()) {
			if (!inflight.empty()) { drain(); continue; }
			break;
		}
		int p = calls++ & 1;
		int total = stage_inputs(plan, prog, c.h_rid + p * BUDGET, c.h_pos + p * BUDGET,
		                         c.h_rows + p * MAX_SEQS);
		launch_step(c, eng, plan, total, p * BUDGET, p * MAX_SEQS);
		cudaEventRecord(ev_tok[p]);
		if (!inflight.empty()) {		// 消费第 N 步, 此刻 GPU 正跑第 N+1 步
			int q = 1 - p;
			cudaEventSynchronize(ev_tok[q]);
			sched.flush_release();		// N-1 步的死者最后被第 N 步引用, 现在归还安全
			collect(gen, sched.update(inflight, std::vector<int>(
				c.h_tok + q * MAX_SEQS, c.h_tok + q * MAX_SEQS + inflight.req_ids.size())));
		}
		inflight = plan;
	}
	drain();
	assert(sched.num_preemptions() == 0);
	sched.flush_release();
	assert((int)pool.num_free_blocks() == BLOCKS && (int)pool.num_free_slots() == MAX_SEQS);
	cudaEventDestroy(ev_tok[0]);
	cudaEventDestroy(ev_tok[1]);
	return gen;
}

int main() {
	Ctx c;
	c.kv_bytes = (size_t)BLOCKS * KV_BLOCK_SIZE * KS * sizeof(float);
	cudaMalloc(&c.k_base, c.kv_bytes);
	cudaMalloc(&c.v_base, c.kv_bytes);
	int W = MAX_SEQ_LEN / KV_BLOCK_SIZE, L = W * KV_BLOCK_SIZE;
	int meta_ints = MAX_SEQS * W + MAX_SEQS + MAX_SEQS + MAX_SEQS * L + (MAX_SEQS + 1);
	cudaMalloc(&c.d_meta, meta_ints * sizeof(int));
	cudaHostAlloc(&c.h_meta, (size_t)2 * meta_ints * sizeof(int), 0);	// Engine 双半区
	cudaMalloc(&c.dq,   (size_t)BUDGET * QS * sizeof(float));
	cudaMalloc(&c.dk,   (size_t)BUDGET * KS * sizeof(float));
	cudaMalloc(&c.dv,   (size_t)BUDGET * KS * sizeof(float));
	cudaMalloc(&c.dout, (size_t)BUDGET * QS * sizeof(float));
	cudaMalloc(&c.d_rid,  2 * BUDGET * sizeof(int));	// 行级元数据: 双份轮换
	cudaMalloc(&c.d_pos,  2 * BUDGET * sizeof(int));
	cudaMalloc(&c.d_rows, 2 * MAX_SEQS * sizeof(int));	// 条目级: 段尾行号
	cudaMalloc(&c.d_tok,  2 * MAX_SEQS * sizeof(int));
	cudaHostAlloc(&c.h_rid,  2 * BUDGET * sizeof(int), 0);
	cudaHostAlloc(&c.h_pos,  2 * BUDGET * sizeof(int), 0);
	cudaHostAlloc(&c.h_rows, 2 * MAX_SEQS * sizeof(int), 0);
	cudaHostAlloc(&c.h_tok,  2 * MAX_SEQS * sizeof(int), 0);

	auto ref = run_sync(c);
	auto got = run_pipelined(c);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

	for (int i = 0; i < NREQ; ++i) {
		assert((int)ref[i].size() == MAX_NEW[i]);	// oracle 自身完整性
		assert(got[i] == ref[i]);			// 流水线逐 token 等于屏障 oracle
	}
	printf("test_pipeline_gpu PASS: %d 请求, 流水线与同步 oracle 逐 token 一致\n", NREQ);

	cudaFreeHost(c.h_meta); cudaFreeHost(c.h_rid); cudaFreeHost(c.h_pos);
	cudaFreeHost(c.h_rows); cudaFreeHost(c.h_tok);
	cudaFree(c.k_base); cudaFree(c.v_base); cudaFree(c.d_meta);
	cudaFree(c.dq); cudaFree(c.dk); cudaFree(c.dv); cudaFree(c.dout);
	cudaFree(c.d_rid); cudaFree(c.d_pos); cudaFree(c.d_rows); cudaFree(c.d_tok);
	return 0;
}
