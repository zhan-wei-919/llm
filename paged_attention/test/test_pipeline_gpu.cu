// chunked prefill 异步流水线的 GPU 实机差分测试 —— 被测对象现在是 PipelineDriver 本身.
//
// 同一组请求用同一个 driver 类跑两遍, 断言逐 token 相等:
//   oracle 模式:  每次 pump 后立即 drain (退化为全同步屏障), 时序上不可能错;
//   流水线模式:  run_to_idle (深度 1 投机: 发射 N+1 在前, 消费 N 在后, 无屏障).
// 两种模式只差主循环两行 —— 差分验证的就是 driver 的时序纪律.
//
// q/k/v 按 (请求, 位置) 内容寻址地现场生成, 与批内打包位置无关 —— 这是差分成立的前提:
// 两种模式的批打包合法地不同 (流水线下晋升晚一拍), 但任一 token 的值只取决于
// 它自己的 q 和它序列的 KV 前缀. 于是 token 流相等 ⟺ scatter 写入位置、页表、
// attention 前缀读取、双缓冲轮换在流水线下全部正确.
//
// 配置故意制造边界: 预算 128 强制切 chunk (300→128/128/44), 129→128/1 (单 token final),
// 256 追平恰踩块边界 (pending_boundary 路径), 33/47 单 chunk 短请求穿插出混合批.
// 块给足, 不触发抢占 (抢占时机在两种模式下合法不同, 会让差分失义).
//
// 编译: nvcc -O2 -arch=native test_pipeline_gpu.cu -o /tmp/test_pipeline_gpu
#include "../Engine.cuh"
#include "../Scheduler.h"
#include "../Driver.h"
#include <cstdio>
#include <vector>
#include <cassert>

constexpr int NH = 32, NKV = 8, HS = 128;	// 与 bench 相同的注意力配置
constexpr int QS = NH * HS, KS = NKV * HS;
constexpr int MAX_SEQS = 8;			// pool 槽位 = 调度座位, >= NREQ 避免座位差分
constexpr int MAX_SEQ_LEN = 512;
constexpr int BUDGET = 128;			// 单步 token 预算, 也是 chunk 粒度
constexpr int BLOCKS = 256;			// 充裕: 抢占在两种模式下时机不同, 必须避免
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
	cudaStream_t t;
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

// 发射一步的全部 GPU 命令 (不含任何同步), off 是双缓冲半区起点.
// 满足 LaunchFn 契约: 返回时命令全部入队, token 的 D2H 落在 h_tok 对应半区.
static void launch_step(Ctx &c, Engine &eng, const StepPlan &plan,
                        int total, int off_rows, int off_tok) {
	int nb = (int)plan.req_ids.size();
	cudaMemcpyAsync(c.d_rid + off_rows, c.h_rid + off_rows, total * sizeof(int), cudaMemcpyHostToDevice, c.t);
	cudaMemcpyAsync(c.d_pos + off_rows, c.h_pos + off_rows, total * sizeof(int), cudaMemcpyHostToDevice, c.t);
	cudaMemcpyAsync(c.d_rows + off_tok, c.h_rows + off_tok, nb * sizeof(int), cudaMemcpyHostToDevice, c.t);
	fill_qkv<<<total, 128, 0, c.t>>>(c.dq, c.dk, c.dv, c.d_rid + off_rows, c.d_pos + off_rows);
	eng.forward(plan.slots.data(), nb, plan.lens.data(), c.dq, c.dk, c.dv, c.dout, c.t);
	fake_sample<<<nb, 1, 0, c.t>>>(c.d_tok + off_tok, c.dout, QS, c.d_rows + off_tok);
	cudaMemcpyAsync(c.h_tok + off_tok, c.d_tok + off_tok, nb * sizeof(int), cudaMemcpyDeviceToHost, c.t);
}

// 同一个 driver, 两种用法; pipelined=false 即同步 oracle
static std::vector<std::vector<int>> run(Ctx &c, bool pipelined) {
	KV_Pool pool(c.k_base, c.v_base, Dtype::F32, KS, c.kv_bytes, MAX_SEQS, MAX_SEQ_LEN);
	Engine eng(pool, NH, NKV, HS, c.d_meta, c.h_meta);
	Scheduler sched(pool, {MAX_SEQS, BUDGET, /*eos=*/-1});
	for (int i = 0; i < NREQ; ++i)
		sched.add_request(std::vector<int>(PROMPT_LEN[i], i), MAX_NEW[i]);

	std::vector<std::vector<int>> gen(NREQ);
	std::vector<int> prog(NREQ, 0);
	// "怎么算一步": 捕获计算资源 (c/eng/prog), plan 与半区编号由 driver 传入
	auto launch_fn = [&c, &eng, &prog](const StepPlan &plan, int p) {
		int total = stage_inputs(plan, prog, c.h_rid + p * BUDGET,
		                         c.h_pos + p * BUDGET, c.h_rows + p * MAX_SEQS);
		launch_step(c, eng, plan, total, p * BUDGET, p * MAX_SEQS);
	};
	// "完成的请求去哪": 只留生成段
	auto on_done = [&gen](std::vector<Request> &&done) {
		for (auto &r : done)
			gen[r.id] = std::vector<int>(r.token_ids.begin() + r.prompt_len, r.token_ids.end());
	};
	PipelineDriver drv(sched, c.h_tok, MAX_SEQS, c.t, launch_fn, on_done);	// 声明在 sched 之后: 先析构, 先排空
	if (pipelined) drv.run_to_idle();
	else while (drv.pump() != PipelineDriver::Pump::IDLE) drv.drain();

	assert(sched.num_preemptions() == 0);
	assert((int)pool.num_free_blocks() == BLOCKS && (int)pool.num_free_slots() == MAX_SEQS);
	return gen;
}

int main() {
	Ctx c;
	cudaStreamCreate(&c.t);
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

	auto ref = run(c, /*pipelined=*/false);
	auto got = run(c, /*pipelined=*/true);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

	for (int i = 0; i < NREQ; ++i) {
		assert((int)ref[i].size() == MAX_NEW[i]);	// oracle 自身完整性
		assert(got[i] == ref[i]);			// 流水线逐 token 等于同步 oracle
	}
	printf("test_pipeline_gpu PASS: %d 请求, PipelineDriver 流水模式与同步 oracle 逐 token 一致\n", NREQ);

	cudaFreeHost(c.h_meta); cudaFreeHost(c.h_rid); cudaFreeHost(c.h_pos);
	cudaFreeHost(c.h_rows); cudaFreeHost(c.h_tok);
	cudaFree(c.k_base); cudaFree(c.v_base); cudaFree(c.d_meta);
	cudaFree(c.dq); cudaFree(c.dk); cudaFree(c.dv); cudaFree(c.dout);
	cudaFree(c.d_rid); cudaFree(c.d_pos); cudaFree(c.d_rows); cudaFree(c.d_tok);
	cudaStreamDestroy(c.t);
	return 0;
}
