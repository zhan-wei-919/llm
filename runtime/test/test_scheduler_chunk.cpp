#include "../Scheduler.h"
#include <cassert>
#include <cstdio>

static void engine_append(KV_Pool &pool, const ScheduledBatch &batch) {
	for (size_t b = 0; b < batch.plan.req_ids.size(); ++b) pool.append(batch.plan.slots[b], batch.plan.lens[b]);
}

static KV_Pool make_pool(int num_blocks, int max_seqs, int max_seq_len) {
	return KV_Pool(nullptr, nullptr, Dtype::F16, 1, (size_t)num_blocks * KV_BLOCK_SIZE * 2, 1, max_seqs, max_seq_len);
}

static ScheduledBatch launch(Scheduler &sched, KV_Pool &pool, ExecutionPhase phase) {
	ScheduledBatch batch = sched.schedule(phase);
	if (!batch.empty()) engine_append(pool, batch);
	return batch;
}

static void test_single_long_prompt() {
	const int NB = 100;
	KV_Pool pool = make_pool(NB, 4, 256);
	LocalKvHandoff handoff;
	Scheduler sched(pool, {4, 32, -1}, handoff);
	sched.add_request(std::vector<int>(80, 7), 4);

	ScheduledBatch p1 = launch(sched, pool, ExecutionPhase::PREFILL);
	assert(p1.phase == ExecutionPhase::PREFILL && p1.plan.lens == std::vector<int>{32});
	int slot = p1.plan.slots[0];
	assert(sched.update(p1, {999}).empty());

	ScheduledBatch p2 = launch(sched, pool, ExecutionPhase::PREFILL);
	assert(p2.plan.lens == std::vector<int>{32} && p2.plan.slots[0] == slot);
	assert(sched.update(p2, {999}).empty());

	ScheduledBatch p3 = launch(sched, pool, ExecutionPhase::PREFILL);
	assert(p3.plan.lens == std::vector<int>{16} && p3.plan.slots[0] == slot);
	assert(sched.update(p3, {100}).empty());

	std::vector<Request> done;
	for (int i = 0; i < 3; ++i) {
		ScheduledBatch d = launch(sched, pool, ExecutionPhase::DECODE);
		assert(d.phase == ExecutionPhase::DECODE && d.plan.lens == std::vector<int>{1});
		auto out = sched.update(d, {101 + i});
		done.insert(done.end(), out.begin(), out.end());
	}
	assert(done.size() == 1);
	assert(done[0].token_ids.size() == 84 && done[0].token_ids[80] == 100 && done[0].token_ids[83] == 103);
	sched.flush_release();
	assert((int)pool.num_free_blocks() == NB && (int)pool.num_free_slots() == 4);
}

static void test_phase_batches_are_separate() {
	const int NB = 100;
	KV_Pool pool = make_pool(NB, 4, 256);
	LocalKvHandoff handoff;
	Scheduler sched(pool, {4, 32, -1}, handoff);
	int r0 = sched.add_request({7}, 3);
	ScheduledBatch first = launch(sched, pool, ExecutionPhase::PREFILL);
	sched.update(first, {100});
	int r1 = sched.add_request({8}, 2);

	ScheduledBatch p = launch(sched, pool, ExecutionPhase::PREFILL);
	ScheduledBatch d = launch(sched, pool, ExecutionPhase::DECODE);
	assert(p.phase == ExecutionPhase::PREFILL && p.plan.req_ids == std::vector<int>{r1});
	assert(d.phase == ExecutionPhase::DECODE && d.plan.req_ids == std::vector<int>{r0});
	assert(p.plan.lens == std::vector<int>{1} && d.plan.lens == std::vector<int>{1});
	sched.update(p, {200});
	sched.update(d, {101});

	ScheduledBatch d2 = launch(sched, pool, ExecutionPhase::DECODE);
	assert((d2.plan.req_ids == std::vector<int>{r0, r1}));
	auto done = sched.update(d2, {102, 201});
	assert(done.size() == 2);
	sched.flush_release();
	assert((int)pool.num_free_blocks() == NB && (int)pool.num_free_slots() == 4);
}

static void test_chunks_schedule_ahead() {
	const int NB = 100;
	KV_Pool pool = make_pool(NB, 4, 256);
	LocalKvHandoff handoff;
	Scheduler sched(pool, {4, 32, -1}, handoff);
	sched.add_request(std::vector<int>(80, 7), 2);

	ScheduledBatch p1 = launch(sched, pool, ExecutionPhase::PREFILL);
	ScheduledBatch p2 = launch(sched, pool, ExecutionPhase::PREFILL);
	ScheduledBatch p3 = launch(sched, pool, ExecutionPhase::PREFILL);
	assert(p1.plan.starts == std::vector<int>{0} && p2.plan.starts == std::vector<int>{32} && p3.plan.starts == std::vector<int>{64});
	assert(sched.schedule(ExecutionPhase::PREFILL).empty());
	sched.update(p1, {999});
	sched.update(p2, {999});
	sched.update(p3, {100});
	ScheduledBatch d = launch(sched, pool, ExecutionPhase::DECODE);
	auto done = sched.update(d, {101});
	assert(done.size() == 1);
	sched.flush_release();
	assert((int)pool.num_free_blocks() == NB);
}

static void test_decode_waits_for_previous_token() {
	KV_Pool pool = make_pool(100, 4, 256);
	LocalKvHandoff handoff;
	Scheduler sched(pool, {4, 32, -1}, handoff);
	sched.add_request({7}, 3);
	ScheduledBatch p = launch(sched, pool, ExecutionPhase::PREFILL);
	sched.update(p, {100});
	ScheduledBatch d1 = launch(sched, pool, ExecutionPhase::DECODE);
	assert(d1.plan.ids == std::vector<int>{100});
	assert(sched.schedule(ExecutionPhase::DECODE).empty());
	sched.update(d1, {101});
	ScheduledBatch d2 = launch(sched, pool, ExecutionPhase::DECODE);
	assert(d2.plan.ids == std::vector<int>{101});
	auto done = sched.update(d2, {102});
	assert(done.size() == 1 && done[0].token_ids == std::vector<int>({7, 100, 101, 102}));
	sched.flush_release();
}

class DelayedHandoff final : public KvHandoff {
public:
	void begin(int request_id, KvLease source) override {pending_ = {request_id, source};}
	std::vector<HandoffResult> poll() override {
		if (!ready_) return {};
		ready_ = false;
		return {pending_};
	}
	void release() {ready_ = true;}

private:
	HandoffResult pending_{};
	bool ready_ = false;
};

static void test_decode_waits_for_handoff() {
	KV_Pool pool = make_pool(100, 4, 256);
	DelayedHandoff handoff;
	Scheduler sched(pool, {4, 32, -1}, handoff);
	sched.add_request({7}, 2);
	ScheduledBatch p = launch(sched, pool, ExecutionPhase::PREFILL);
	sched.update(p, {100});
	assert(sched.schedule(ExecutionPhase::DECODE).empty());
	handoff.release();
	ScheduledBatch d = launch(sched, pool, ExecutionPhase::DECODE);
	assert(d.plan.ids == std::vector<int>{100});
	auto done = sched.update(d, {101});
	assert(done.size() == 1);
	sched.flush_release();
}

static void test_prefill_reserve_uses_scheduled_length() {
	KV_Pool pool = make_pool(2, 2, 16);
	LocalKvHandoff handoff;
	Scheduler sched(pool, {2, 16, -1}, handoff);
	sched.add_request(std::vector<int>(5, 7), 2);
	sched.add_request(std::vector<int>(5, 8), 2);
	ScheduledBatch p = launch(sched, pool, ExecutionPhase::PREFILL);
	assert(p.plan.req_ids.size() == 2 && p.plan.lens == std::vector<int>({5, 5}));
	sched.update(p, {100, 200});
	ScheduledBatch d = launch(sched, pool, ExecutionPhase::DECODE);
	assert(d.plan.req_ids.size() == 2);
	auto done = sched.update(d, {101, 201});
	assert(done.size() == 2);
	sched.flush_release();
	assert(pool.num_free_blocks() == 2 && pool.num_free_slots() == 2);
}

static void test_decode_preemption_returns_to_prefill() {
	KV_Pool pool = make_pool(3, 2, 32);
	LocalKvHandoff handoff;
	Scheduler sched(pool, {2, 16, -1}, handoff);
	sched.add_request({7}, 20);
	ScheduledBatch p = launch(sched, pool, ExecutionPhase::PREFILL);
	sched.update(p, {100});
	for (int i = 0; i < 15; ++i) {
		ScheduledBatch d = launch(sched, pool, ExecutionPhase::DECODE);
		sched.update(d, {101 + i});
	}
	int external = pool.alloc_seq();
	pool.append(external, 32);
	assert(sched.schedule(ExecutionPhase::DECODE).empty());
	assert(sched.num_preemptions() == 1 && sched.num_prefill() == 1);
	pool.release(external);
	ScheduledBatch recompute0 = launch(sched, pool, ExecutionPhase::PREFILL);
	ScheduledBatch recompute1 = launch(sched, pool, ExecutionPhase::PREFILL);
	sched.update(recompute0, {999});
	sched.update(recompute1, {200});
	std::vector<Request> done;
	for (int i = 0; i < 3; ++i) {
		ScheduledBatch d = launch(sched, pool, ExecutionPhase::DECODE);
		auto out = sched.update(d, {201 + i});
		done.insert(done.end(), out.begin(), out.end());
	}
	assert(done.size() == 1);
	sched.flush_release();
	assert(pool.num_free_blocks() == 3 && pool.num_free_slots() == 2);
}

int main() {
	test_single_long_prompt();
	test_phase_batches_are_separate();
	test_chunks_schedule_ahead();
	test_decode_waits_for_previous_token();
	test_decode_waits_for_handoff();
	test_prefill_reserve_uses_scheduled_length();
	test_decode_preemption_returns_to_prefill();
	std::printf("test_scheduler_chunk PASS\n");
	return 0;
}
