#include "../Engine.cuh"
#include "../Scheduler.h"
#include "../tensor/Arena.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// 端到端集成: Scheduler(决策) + Engine(执行) 真跑 GPU kernel.
// 本测试扮演两个缺席的角色:
//   模型   —— 每请求的 q/k/v 由固定种子随机数生成(位置的确定性函数,
//              抢占重算时能复现相同的 K/V, 与真实模型的确定性 forward 同构);
//   采样器 —— 恒返回 100+id, 请求靠 max_new_tokens 收尾.
// 每步把 GPU 输出与 CPU 稠密参照逐元素对比:
//   prefill 验证每一行 (因果: 位置 p 只看 0..p);
//   decode  验证新 token 对全部历史的 attention (跨块查页表读出的历史).
// 池子沿用 CPU 调度测试的 12 块 / 4 槽配置, 抢占→重算路径同样被真实覆盖.

constexpr int NH = 6, NKV = 2, HS = 12;
constexpr int QS = NH * HS, KS = NKV * HS;	// 72 / 24
constexpr int BLOCKS = 12, MAX_SEQS = 4, MAX_SEQ_LEN = 64;
constexpr int NUM_REQ = 8;
constexpr int PROMPT[NUM_REQ]  = {20, 25, 30, 18, 22, 28, 16, 24};
constexpr int MAX_NEW[NUM_REQ] = {44, 39, 34, 46, 42, 36, 48, 40};
constexpr int ROWS_CAP = 128;			// 单步打包行数上限 (预算 64, 留余量)

static float frand() { return (rand() % 200 - 100) / 100.0f; }

// 每请求的"模型输出": MAX_SEQ_LEN 行 q/k/v, 按位置索引
static float hq[NUM_REQ][MAX_SEQ_LEN * QS];
static float hk[NUM_REQ][MAX_SEQ_LEN * KS];
static float hv[NUM_REQ][MAX_SEQ_LEN * KS];

// CPU 参照: 请求 id 的位置 p 对历史 0..p 做 GQA attention
static void ref_attn(int id, int p, float *out) {
	const int GROUP = NH / NKV;
	for (int h = 0; h < NH; ++h) {
		int g = h / GROUP;
		const float *qh = hq[id] + (size_t)p * QS + h * HS;
		float s[MAX_SEQ_LEN], mx = -INFINITY, z = 0;
		for (int j = 0; j <= p; ++j) {
			float acc = 0;
			for (int d = 0; d < HS; ++d) acc += qh[d] * hk[id][j * KS + g * HS + d];
			s[j] = acc / sqrtf((float)HS);
			mx = fmaxf(mx, s[j]);
		}
		for (int j = 0; j <= p; ++j) { s[j] = expf(s[j] - mx); z += s[j]; }
		for (int d = 0; d < HS; ++d) {
			float acc = 0;
			for (int j = 0; j <= p; ++j) acc += s[j] / z * hv[id][j * KS + g * HS + d];
			out[h * HS + d] = acc;
		}
	}
}

int main() {
	// ---------- 装配: Arena → KV_Pool → Engine → Scheduler ----------
	int W = (MAX_SEQ_LEN + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
	Arena a(8);
	Tensor *t_table = a.alloc({MAX_SEQS, W},            Dtype::F32);
	Tensor *t_len   = a.alloc({MAX_SEQS},               Dtype::F32);
	Tensor *t_pos   = a.alloc({MAX_SEQS},               Dtype::F32);
	Tensor *t_ids   = a.alloc({MAX_SEQS * MAX_SEQ_LEN}, Dtype::F32);
	Tensor *t_cu    = a.alloc({MAX_SEQS + 1},           Dtype::F32);
	a.finalize();
	// KV 显存不走 alloc_kv_pool (那会吃满整卡, 永远不会抢占),
	// 手动只给 12 块, 让调度器的记账和抢占真的被逼出来
	size_t kv_bytes = (size_t)BLOCKS * KV_BLOCK_SIZE * KS * sizeof(float);
	void *k_base, *v_base;
	cudaMalloc(&k_base, kv_bytes);
	cudaMalloc(&v_base, kv_bytes);
	KV_Pool pool(k_base, v_base, Dtype::F32, KS, kv_bytes, MAX_SEQS, MAX_SEQ_LEN);
	Engine eng(pool, NH, NKV, HS,
	           (int *)t_table->ptr, (int *)t_len->ptr, (int *)t_pos->ptr,
	           (int *)t_ids->ptr, (int *)t_cu->ptr);
	Scheduler sched(pool, {/*max_num_seqs=*/MAX_SEQS, /*max_num_batched_tokens=*/64, /*eos=*/-1});

	// ---------- 造数据 + 入队 ----------
	for (int i = 0; i < NUM_REQ; ++i) {
		srand(1000 + i);
		for (int t = 0; t < MAX_SEQ_LEN * QS; ++t) hq[i][t] = frand();
		for (int t = 0; t < MAX_SEQ_LEN * KS; ++t) hk[i][t] = frand();
		for (int t = 0; t < MAX_SEQ_LEN * KS; ++t) hv[i][t] = frand();
		int id = sched.add_request(std::vector<int>(PROMPT[i], i), MAX_NEW[i]);
		assert(id == i);
	}

	float *dq, *dk, *dv, *dout;
	cudaMalloc(&dq,   sizeof(float) * ROWS_CAP * QS);
	cudaMalloc(&dk,   sizeof(float) * ROWS_CAP * KS);
	cudaMalloc(&dv,   sizeof(float) * ROWS_CAP * KS);
	cudaMalloc(&dout, sizeof(float) * ROWS_CAP * QS);
	static float pq[ROWS_CAP * QS], pk[ROWS_CAP * KS], pv[ROWS_CAP * KS];
	static float gout[ROWS_CAP * QS], ref[QS];

	// ---------- executor 主循环: 计划 → 打包 → 发射 → 验证 → 回灌 ----------
	int fed[NUM_REQ] = {};	// 每请求已喂进 pool 的行数 (== 它的 KV 长度)
	float max_diff = 0;
	int steps = 0, finished = 0, verified_rows = 0;
	while (true) {
		StepPlan plan = sched.schedule();
		if (plan.empty()) break;
		++steps;
		int B = (int)plan.req_ids.size();
		if (plan.is_prefill) {
			// 打包: 每成员取位置 0..len-1 的行 (重算时 len > prompt, 同一函数复现)
			int total = 0;
			for (int b = 0; b < B; ++b) {
				int id = plan.req_ids[b], len = plan.lens[b];
				memcpy(pq + (size_t)total * QS, hq[id], sizeof(float) * len * QS);
				memcpy(pk + (size_t)total * KS, hk[id], sizeof(float) * len * KS);
				memcpy(pv + (size_t)total * KS, hv[id], sizeof(float) * len * KS);
				total += len;
			}
			cudaMemcpy(dq, pq, sizeof(float) * total * QS, cudaMemcpyHostToDevice);
			cudaMemcpy(dk, pk, sizeof(float) * total * KS, cudaMemcpyHostToDevice);
			cudaMemcpy(dv, pv, sizeof(float) * total * KS, cudaMemcpyHostToDevice);
			eng.prefill(plan.slots.data(), B, plan.lens.data(), dq, dk, dv, dout);
			cudaMemcpy(gout, dout, sizeof(float) * total * QS, cudaMemcpyDeviceToHost);
			printf("step %3d  prefill [", steps);
			for (int b = 0, off = 0; b < B; ++b) {
				int id = plan.req_ids[b], len = plan.lens[b];
				printf("%s%d(%d tok%s)", b ? "  " : "", id, len,
				       len > PROMPT[id] ? ", 重算" : "");
				for (int p = 0; p < len; ++p, ++verified_rows) {
					ref_attn(id, p, ref);
					for (int x = 0; x < QS; ++x)
						max_diff = fmaxf(max_diff,
							fabsf(gout[(size_t)(off + p) * QS + x] - ref[x]));
				}
				off += len;
				fed[id] = len;
			}
			printf("]  free=%d\n", (int)pool.num_free_blocks());
		} else {
			// 打包: 每成员取下一个位置 (fed[id]) 的那一行
			for (int b = 0; b < B; ++b) {
				int id = plan.req_ids[b];
				memcpy(pq + (size_t)b * QS, hq[id] + (size_t)fed[id] * QS, sizeof(float) * QS);
				memcpy(pk + (size_t)b * KS, hk[id] + (size_t)fed[id] * KS, sizeof(float) * KS);
				memcpy(pv + (size_t)b * KS, hv[id] + (size_t)fed[id] * KS, sizeof(float) * KS);
			}
			cudaMemcpy(dq, pq, sizeof(float) * B * QS, cudaMemcpyHostToDevice);
			cudaMemcpy(dk, pk, sizeof(float) * B * KS, cudaMemcpyHostToDevice);
			cudaMemcpy(dv, pv, sizeof(float) * B * KS, cudaMemcpyHostToDevice);
			eng.decode(plan.slots.data(), B, dq, dk, dv, dout);
			cudaMemcpy(gout, dout, sizeof(float) * B * QS, cudaMemcpyDeviceToHost);
			for (int b = 0; b < B; ++b, ++verified_rows) {
				int id = plan.req_ids[b];
				ref_attn(id, fed[id], ref);	// 新 token 看全部历史 0..fed
				for (int x = 0; x < QS; ++x)
					max_diff = fmaxf(max_diff, fabsf(gout[(size_t)b * QS + x] - ref[x]));
				fed[id]++;
			}
		}
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

		std::vector<int> toks(B, 0);
		for (int b = 0; b < B; ++b) toks[b] = 100 + plan.req_ids[b];
		for (const Request &r : sched.update(plan, toks)) {
			printf("          -> req %d 完成 (%d token)\n", r.id, (int)r.token_ids.size());
			++finished;
		}
	}

	// ---------- 收尾: 守恒 + 精度 ----------
	assert(finished == NUM_REQ);
	assert((int)pool.num_free_blocks() == BLOCKS);
	assert((int)pool.num_free_slots() == MAX_SEQS);
	printf("\n%d 步, %d 次抢占, 验证 %d 行, max |gpu - ref| = %g  ->  %s\n",
	       steps, sched.num_preemptions(), verified_rows, max_diff,
	       max_diff < 1e-4 ? "OK" : "MISMATCH");
	cudaFree(k_base); cudaFree(v_base);
	cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
	return max_diff < 1e-4 ? 0 : 1;
}
