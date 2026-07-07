#include "../Scheduler.h"
#include <cstdio>
#include <cassert>

// 纯 CPU 调度测试, 不碰 GPU.
// 假 executor 只复刻 Engine 对 pool 的记账动作 (prefill: append(slot,len);
// decode: append(slot,1)), 每序列每步产出 1 个 token, 采样用确定性规则代替.
// 池子故意配小 (12 块 / 4 槽), 8 条最终长度 64 的请求挤进来:
// 4 条并发最坏要 16 块 > 12 块, 抢占必然发生.
// 验证: 全部请求跑完 + 各自 token 无错乱 (抢占重算不丢不重) + 块和槽全部归还.

constexpr int NUM_REQ = 8;
constexpr int PROMPT[NUM_REQ]  = {20, 25, 30, 18, 22, 28, 16, 24};
constexpr int MAX_NEW[NUM_REQ] = {44, 39, 34, 46, 42, 36, 48, 40};
constexpr int EOS = 2;
constexpr int EOS_REQ = 6, EOS_AFTER = 10;	// 请求 6 生成第 10 个 token 时出 EOS, 验证提前退场

int main() {
	// k/v 基址传 nullptr: 本测试只走记账路径, 不触碰 KV 数据
	constexpr int BLOCKS = 12, MAX_SEQS = 4, MAX_SEQ_LEN = 64;
	KV_Pool pool(nullptr, nullptr, Dtype::F32, /*kv_stride=*/1,
	             (size_t)BLOCKS * KV_BLOCK_SIZE * 1 * sizeof(float),
	             MAX_SEQS, MAX_SEQ_LEN);
	Scheduler sched(pool, {/*max_num_seqs=*/4, /*max_num_batched_tokens=*/64, EOS});

	for (int i = 0; i < NUM_REQ; ++i) {
		std::vector<int> prompt(PROMPT[i], i);	// 填充值 = 请求 id, 结束时验证内容用
		int id = sched.add_request(std::move(prompt), MAX_NEW[i]);
		assert(id == i);
	}

	int gen_count[NUM_REQ] = {};	// 每请求已产出的 token 数 (含跨越抢占)
	int finished = 0, steps = 0, seen_preempt = 0;
	while (true) {
		StepPlan plan = sched.schedule();
		if (plan.empty()) break;
		++steps;
		if (sched.num_preemptions() > seen_preempt) {
			printf("          !! preempt x%d\n", sched.num_preemptions() - seen_preempt);
			seen_preempt = sched.num_preemptions();
		}
		// 假 Engine: 记账动作与真 Engine 完全一致, 每序列产出 1 个 token
		std::vector<int> toks(plan.req_ids.size());
		for (size_t b = 0; b < plan.req_ids.size(); ++b) {
			int id = plan.req_ids[b];
			pool.append(plan.slots[b], plan.is_prefill ? plan.lens[b] : 1);
			++gen_count[id];
			toks[b] = (id == EOS_REQ && gen_count[id] == EOS_AFTER) ? EOS : 100 + id;
		}
		printf("step %3d  %s B=%d free=%2d reqs=[", steps,
		       plan.is_prefill ? "prefill" : "decode ", (int)plan.req_ids.size(),
		       (int)pool.num_free_blocks());
		for (size_t b = 0; b < plan.req_ids.size(); ++b)
			printf("%s%d", b ? " " : "", plan.req_ids[b]);
		printf("]\n");

		for (const Request &r : sched.update(plan, toks)) {
			// 生成数: EOS 请求恰好 EOS_AFTER 个, 其余跑满 max_new
			int expect_gen = r.id == EOS_REQ ? EOS_AFTER : MAX_NEW[r.id];
			assert(gen_count[r.id] == expect_gen);
			assert((int)r.token_ids.size() == PROMPT[r.id] + expect_gen);
			// 内容: prompt 段是 id 本身, 生成段是 100+id (末尾可能是 EOS).
			// 抢占重算若丢 token 或串了序列, 这里会当场揪出来
			for (int t = 0; t < (int)r.token_ids.size(); ++t) {
				int expect = t < r.prompt_len ? r.id : 100 + r.id;
				if (r.id == EOS_REQ && t == (int)r.token_ids.size() - 1) expect = EOS;
				assert(r.token_ids[t] == expect);
			}
			printf("          -> req %d done: %d tokens (prompt %d + gen %d)\n",
			       r.id, (int)r.token_ids.size(), r.prompt_len, expect_gen);
			++finished;
		}
	}

	// 守恒: 所有请求交付, 队列清空, 块和槽一个不少地回到 pool
	assert(finished == NUM_REQ);
	assert(sched.num_waiting() == 0 && sched.num_running() == 0);
	assert((int)pool.num_free_blocks() == BLOCKS);
	assert((int)pool.num_free_slots() == MAX_SEQS);
	printf("\nOK: %d requests in %d steps, %d preemptions, all blocks/slots returned\n",
	       finished, steps, sched.num_preemptions());
	return 0;
}
