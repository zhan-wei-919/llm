// chunked prefill 的纯逻辑单测: 不碰 GPU, KV_Pool 用空指针构造 (调度路径不解引用 k/v base).
// 测试扮演 Engine: 按 driver 契约, 每个 plan 返回后、下一次 schedule 前调 pool.append 落账.
//
// 场景 A: 单条长 prompt (80) > 预算 (32) → 切成 32/32/16 三个 chunk;
//         中间 chunk 不产 token、slot 跨步保持; 最后一个 chunk 才进 running; decode 至完成.
// 场景 B: 两条 prompt (40/40) 共享预算 (32) → FCFS 装箱:
//         [r0:32] → [r0:8(final), r1:24] → [r1:16(final)];
//         验证"每步至多一个 partial 且必在队头".
//
// 编译: g++ -std=c++17 -I/usr/local/cuda/include -o test_scheduler_chunk test_scheduler_chunk.cpp
#include "../Scheduler.h"
#include <cstdio>

// 扮演 Engine 的 launch 时记账: prefill 按 lens append, decode 每条 1 个
static void engine_append(KV_Pool &pool, const StepPlan &plan) {
	for (size_t b = 0; b < plan.req_ids.size(); ++b)
		pool.append(plan.slots[b], plan.is_prefill ? plan.lens[b] : 1);
}

static KV_Pool make_pool(int num_blocks, int max_seqs, int max_seq_len) {
	// kv_stride=1, F16(2字节) → 每块 32 字节, capacity 反推出想要的块数
	return KV_Pool(nullptr, nullptr, Dtype::F16, 1,
	               (size_t)num_blocks * KV_BLOCK_SIZE * 2, max_seqs, max_seq_len);
}

static void test_single_long_prompt() {
	const int NB = 100;
	KV_Pool pool = make_pool(NB, /*max_seqs=*/4, /*max_seq_len=*/256);
	Scheduler sched(pool, {/*max_num_seqs=*/4, /*max_num_batched_tokens=*/32, /*eos=*/-1});
	sched.add_request(std::vector<int>(80, 7), /*max_new_tokens=*/4);

	// chunk 1: 预算截断出 32
	StepPlan p1 = sched.schedule();
	assert(p1.is_prefill && p1.req_ids.size() == 1 && p1.lens[0] == 32);
	int slot = p1.slots[0];
	engine_append(pool, p1);
	assert((int)pool.num_free_blocks() == NB - 2);		// 32 token = 2 块
	assert(sched.update(p1, {999}).empty());		// 中间 chunk: 假 token 必须被丢弃
	assert(sched.num_waiting() == 1 && sched.num_running() == 0);	// partial 回 waiting

	// chunk 2: 还是 32, slot 不变
	StepPlan p2 = sched.schedule();
	assert(p2.is_prefill && p2.lens[0] == 32 && p2.slots[0] == slot);
	engine_append(pool, p2);
	assert(sched.update(p2, {999}).empty());

	// chunk 3 (final): 剩余 16, 产出第一个 token, 请求进 running
	StepPlan p3 = sched.schedule();
	assert(p3.is_prefill && p3.lens[0] == 16 && p3.slots[0] == slot);
	engine_append(pool, p3);
	assert(sched.update(p3, {100}).empty());		// 产 token 但没到停止条件
	assert(sched.num_waiting() == 0 && sched.num_running() == 1);

	// decode 3 步后按 max_new_tokens=4 收尾 (prefill 产 1 + decode 产 3)
	std::vector<Request> done;
	for (int i = 0; i < 3; ++i) {
		StepPlan d = sched.schedule();
		assert(!d.is_prefill && d.req_ids.size() == 1 && d.slots[0] == slot);
		engine_append(pool, d);
		auto out = sched.update(d, {101 + i});
		done.insert(done.end(), out.begin(), out.end());
	}
	assert(done.size() == 1);
	assert((int)done[0].token_ids.size() == 80 + 4);	// prompt + 4 个生成 token
	assert(done[0].token_ids[80] == 100 && done[0].token_ids[83] == 103);

	sched.flush_release();
	assert((int)pool.num_free_blocks() == NB);		// 块全额归还
	assert((int)pool.num_free_slots() == 4);
	assert(sched.num_preemptions() == 0);
	printf("test_single_long_prompt PASS\n");
}

static void test_two_prompts_share_budget() {
	const int NB = 100;
	KV_Pool pool = make_pool(NB, /*max_seqs=*/4, /*max_seq_len=*/256);
	Scheduler sched(pool, {/*max_num_seqs=*/4, /*max_num_batched_tokens=*/32, /*eos=*/-1});
	int r0 = sched.add_request(std::vector<int>(40, 7), /*max_new_tokens=*/2);
	int r1 = sched.add_request(std::vector<int>(40, 8), /*max_new_tokens=*/2);

	// 步 1: r0 独占预算, 成为唯一 partial
	StepPlan p1 = sched.schedule();
	assert(p1.req_ids == std::vector<int>{r0} && p1.lens[0] == 32);
	engine_append(pool, p1);
	sched.update(p1, {999});

	// 步 2: 队头 partial 先拿 (r0 final 8), 剩余预算给后来者 (r1 拿 24 成为新 partial)
	StepPlan p2 = sched.schedule();
	assert((p2.req_ids == std::vector<int>{r0, r1}));
	assert((p2.lens == std::vector<int>{8, 24}));
	engine_append(pool, p2);
	assert(sched.update(p2, {100, 999}).empty());		// r0 的 token 被消费, r1 的被丢弃
	assert(sched.num_running() == 1 && sched.num_waiting() == 1);

	// 步 3: prefill 独占步, r1 收尾, r0 的 decode 这一步停摆 (当前设计如此)
	StepPlan p3 = sched.schedule();
	assert(p3.is_prefill && p3.req_ids == std::vector<int>{r1} && p3.lens[0] == 16);
	engine_append(pool, p3);
	assert(sched.update(p3, {200}).empty());
	assert(sched.num_running() == 2 && sched.num_waiting() == 0);

	// 步 4: 混不进 prefill 了, decode 双请求齐跑, 各自第 2 个 token 触发 max_new_tokens
	StepPlan p4 = sched.schedule();
	assert(!p4.is_prefill && p4.req_ids.size() == 2);
	engine_append(pool, p4);
	auto done = sched.update(p4, {101, 201});
	assert(done.size() == 2);
	for (const Request &r : done)
		assert((int)r.token_ids.size() == 40 + 2);

	sched.flush_release();
	assert((int)pool.num_free_blocks() == NB);
	assert((int)pool.num_free_slots() == 4);
	assert(sched.num_preemptions() == 0);
	printf("test_two_prompts_share_budget PASS\n");
}

int main() {
	test_single_long_prompt();
	test_two_prompts_share_budget();
	printf("all tests PASS\n");
	return 0;
}
