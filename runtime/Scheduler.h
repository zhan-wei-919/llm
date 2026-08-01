#pragma once
#include "../kv/KV_pool.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

enum class ExecutionPhase { PREFILL, DECODE };
enum class RequestPhase { PREFILL, HANDOFF, DECODE, FINISHED };

// KVCache的使用权
struct KvLease {
	int executor_id = 0;	// KV 属于哪张卡或哪个执行器
	int slot = -1;		// 该执行器的 KV Pool 序列槽
};

// Prefill 完成后调用 begin(request_id, source)：请求 request_id 的 KV 目前在 source, 请开始交给 Decode 端
// 之后调度器反复调用 poll(), 拿到已经完成的 HandoffResult
// 其中 destination 是 Decode 此后应使用的 KV 位置
struct HandoffResult {
	int request_id;
	KvLease destination;
};

class KvHandoff {
public:
	virtual ~KvHandoff() = default;
	virtual void begin(int request_id, KvLease source) = 0;
	virtual std::vector<HandoffResult> poll() = 0;
};

// 当前单卡实现：begin() 直接把 {request_id, source} 塞进 ready_；poll() 立刻取出
class LocalKvHandoff final : public KvHandoff {
public:
	void begin(int request_id, KvLease source) override {ready_.push_back({request_id, source});}
	std::vector<HandoffResult> poll() override {
		std::vector<HandoffResult> ready;
		ready.swap(ready_);
		return ready;
	}

private:
	std::vector<HandoffResult> ready_;
};

struct Request {
	int			id;
	std::vector<int>	token_ids;
	int			prompt_len;
	int			max_new_tokens;
	RequestPhase		phase = RequestPhase::PREFILL;
	KvLease			kv;
	int			in_flight = 0;
	int			cached_len = 0;
	int			scheduled_len = 0;
};

struct SchedulerConfig {
	int	max_num_seqs;
	int	max_num_batched_tokens;
	int	eos_token_id;
};

struct StepPlan {
	std::vector<int>	req_ids;
	std::vector<int>	slots;
	std::vector<int>	lens;
	std::vector<int>	starts;
	std::vector<int>	ids;
	bool empty() const {return req_ids.empty();}
};

struct ScheduledBatch {
	uint64_t	id = 0;
	ExecutionPhase	phase = ExecutionPhase::PREFILL;
	StepPlan	plan;
	bool empty() const {return plan.empty();}
};

class Scheduler {
public:
	Scheduler(KV_Pool &pool, SchedulerConfig cfg, KvHandoff &handoff)
	: pool_(pool), cfg_(cfg), handoff_(handoff), total_blocks_((int)pool.num_free_blocks()) {
		assert(cfg.max_num_seqs <= cfg.max_num_batched_tokens);
		assert(cfg.max_num_seqs <= pool.max_seqs());
	}

	int add_request(std::vector<int> prompt, int max_new_tokens) {
		int worst = (int)prompt.size() + max_new_tokens;
		if (prompt.empty() || worst > pool_.max_blocks_per_seq() * KV_BLOCK_SIZE || blocks_needed(worst) > total_blocks_) return -1;
		int id = next_request_id_++;
		Request r;
		r.id = id;
		r.prompt_len = (int)prompt.size();
		r.token_ids = std::move(prompt);
		r.max_new_tokens = max_new_tokens;
		requests_.emplace(id, std::move(r));
		prefill_ready_.push_back(id);
		return id;
	}

	ScheduledBatch schedule(ExecutionPhase phase) {
		poll_handoffs();
		StepPlan plan = phase == ExecutionPhase::PREFILL ? schedule_prefill() : schedule_decode();
		if (plan.empty()) return {};
		return {next_batch_id_++, phase, std::move(plan)};
	}

	std::vector<Request> update(const ScheduledBatch &batch, const std::vector<int> &new_tokens) {
		std::vector<Request> done;
		const StepPlan &plan = batch.plan;
		for (size_t b = 0; b < plan.req_ids.size(); ++b) {
			auto it = requests_.find(plan.req_ids[b]);
			if (it == requests_.end()) continue;
			Request &r = it->second;
			r.cached_len += plan.lens[b];
			if (batch.phase == ExecutionPhase::PREFILL && r.cached_len < (int)r.token_ids.size()) continue;
			r.in_flight--;
			assert(r.in_flight >= 0);
			r.token_ids.push_back(new_tokens[b]);
			if (finished(r, new_tokens[b])) {
				finish(it, done);
				continue;
			}
			if (batch.phase == ExecutionPhase::PREFILL) {
				r.phase = RequestPhase::HANDOFF;
				handoff_.begin(r.id, r.kv);
			} else {
				r.phase = RequestPhase::DECODE;
				decode_ready_.push_back(r.id);
			}
		}
		return done;
	}

	void flush_release() {
		for (KvLease lease : pending_release_) {
			pool_.release(lease.slot);
			active_leases_--;
		}
		pending_release_.clear();
	}

	size_t num_prefill() const {return prefill_ready_.size();}
	size_t num_decode() const {return decode_ready_.size();}
	size_t num_requests() const {return requests_.size();}
	int num_preemptions() const {return preempt_count_;}

private:
	using RequestMap = std::unordered_map<int, Request>;

	static int blocks_needed(int len) {return (len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;}

	StepPlan schedule_prefill() {
		StepPlan plan;
		int batched_tokens = 0, promised_blocks = 0;
		while (!prefill_ready_.empty()) {
			int id = prefill_ready_.front();
			Request &r = requests_.at(id);
			int target_len = (int)r.token_ids.size();
			if (batched_tokens == cfg_.max_num_batched_tokens || (int)plan.req_ids.size() == cfg_.max_num_seqs) break;
			assert(r.phase == RequestPhase::PREFILL && r.scheduled_len < target_len);
			assert(r.kv.slot == -1 || r.scheduled_len == pool_.seq_len(r.kv.slot));
			int chunk_len = std::min(target_len - r.scheduled_len, cfg_.max_num_batched_tokens - batched_tokens);
			int old_blocks = blocks_needed(r.scheduled_len);
			int new_blocks = blocks_needed(r.scheduled_len + chunk_len);
			int need_blocks = new_blocks - old_blocks;
			bool is_final = r.scheduled_len + chunk_len == target_len;
			int reserve = decode_reserve_blocks() + ((is_final && target_len % KV_BLOCK_SIZE == 0) ? 1 : 0);
			if ((int)pool_.num_free_blocks() - promised_blocks < need_blocks + reserve) break;
			if (r.kv.slot == -1 && active_leases_ == cfg_.max_num_seqs) break;
			if (r.kv.slot == -1) {
				r.kv = {0, pool_.alloc_seq()};
				active_leases_++;
			}
			plan.starts.push_back(r.scheduled_len);
			plan.ids.insert(plan.ids.end(), r.token_ids.begin() + r.scheduled_len, r.token_ids.begin() + r.scheduled_len + chunk_len);
			r.scheduled_len += chunk_len;
			plan.req_ids.push_back(id);
			plan.slots.push_back(r.kv.slot);
			plan.lens.push_back(chunk_len);
			batched_tokens += chunk_len;
			promised_blocks += need_blocks;
			if (is_final) {
				prefill_ready_.pop_front();
				r.in_flight = 1;
			}
		}
		return plan;
	}

	StepPlan schedule_decode() {
		while (!decode_ready_.empty() && decode_blocks_needed() > (int)pool_.num_free_blocks()) preempt_last_decode();
		StepPlan plan;
		while (!decode_ready_.empty() && (int)plan.req_ids.size() < cfg_.max_num_seqs) {
			int id = decode_ready_.front();
			decode_ready_.pop_front();
			Request &r = requests_.at(id);
			assert(r.phase == RequestPhase::DECODE && r.in_flight == 0);
			r.in_flight = 1;
			r.scheduled_len++;
			plan.req_ids.push_back(id);
			plan.slots.push_back(r.kv.slot);
			plan.lens.push_back(1);
			plan.starts.push_back(pool_.seq_len(r.kv.slot));
			plan.ids.push_back(r.token_ids.back());
		}
		return plan;
	}

	int decode_blocks_needed() const {
		int need = 0, batch = 0;
		for (int id : decode_ready_) {
			if (batch++ == cfg_.max_num_seqs) break;
			const Request &r = requests_.at(id);
			if (pool_.seq_len(r.kv.slot) % KV_BLOCK_SIZE == 0) need++;
		}
		return need;
	}

	int decode_reserve_blocks() const {
		int reserve = 0;
		for (const auto &item : requests_) {
			const Request &r = item.second;
			bool will_decode = r.phase == RequestPhase::DECODE || r.phase == RequestPhase::HANDOFF;
			bool final_prefill = r.phase == RequestPhase::PREFILL && r.in_flight != 0 && r.scheduled_len == (int)r.token_ids.size();
			int len = final_prefill ? r.scheduled_len : (r.kv.slot == -1 ? 0 : pool_.seq_len(r.kv.slot));
			if ((will_decode || final_prefill) && r.kv.slot != -1 && len % KV_BLOCK_SIZE == 0) reserve++;
		}
		return reserve;
	}

	void preempt_last_decode() {
		int id = decode_ready_.back();
		decode_ready_.pop_back();
		Request &r = requests_.at(id);
		pool_.release(r.kv.slot);
		active_leases_--;
		r.kv = {};
		r.cached_len = 0;
		r.scheduled_len = 0;
		r.phase = RequestPhase::PREFILL;
		prefill_ready_.push_front(id);
		preempt_count_++;
	}

	void poll_handoffs() {
		for (HandoffResult result : handoff_.poll()) {
			Request &r = requests_.at(result.request_id);
			assert(r.phase == RequestPhase::HANDOFF);
			r.kv = result.destination;
			r.phase = RequestPhase::DECODE;
			decode_ready_.push_back(r.id);
		}
	}

	bool finished(const Request &r, int token) const {
		return token == cfg_.eos_token_id || (int)r.token_ids.size() - r.prompt_len >= r.max_new_tokens;
	}

	void finish(RequestMap::iterator it, std::vector<Request> &done) {
		Request &r = it->second;
		if (r.kv.slot != -1) {
			pending_release_.push_back(r.kv);
			r.kv = {};
		}
		r.phase = RequestPhase::FINISHED;
		done.push_back(std::move(r));
		requests_.erase(it);
	}

	KV_Pool			&pool_;
	const SchedulerConfig	cfg_;
	KvHandoff		&handoff_;
	const int		total_blocks_;
	int			next_request_id_ = 0;
	uint64_t		next_batch_id_ = 0;
	int			active_leases_ = 0;
	int			preempt_count_ = 0;
	RequestMap		requests_;
	std::deque<int>		prefill_ready_;
	std::deque<int>		decode_ready_;
	std::vector<KvLease>	pending_release_;
};
