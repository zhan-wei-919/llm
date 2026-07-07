#pragma once
#include "KV_pool.h"
#include <deque>
#include <unordered_map>
#include <algorithm>

enum class ReqState { WAITING, RUNNING, FINISHED };

struct Request {
	int			id;
	std::vector<int>	token_ids;	// prompt + 已生成; 唯一真相源, 抢占后重算全靠它
	int			prompt_len;	// token_ids 前多少个是 prompt
	int			max_new_tokens;
	ReqState		state = ReqState::WAITING;
	int			slot = -1;	// RUNNING 期间持有的 pool 行号, 其余时刻 -1
};

struct SchedulerConfig {
	int	max_num_seqs;		// running 数上限, 即每步 B 的上限
	int	max_num_batched_tokens;	// 单步 prefill 的 token 总预算
	int	eos_token_id;
};

// req_ids, slots, lens下标对齐，共同构成批内的坐标系
// req_ids[b]		slots[b]		lens[b]
// 请求的逻辑ID	请求在KV_pool里的行号		本步要算的token数
struct StepPlan {
	bool			is_prefill = false;
	std::vector<int>	req_ids;
	std::vector<int>	slots;
	std::vector<int>	lens;	// 仅 prefill 使用: 各序列本步要算的 token 数
	bool empty() const { return req_ids.empty(); }
};

class Scheduler {
public:
	Scheduler(KV_Pool &pool, SchedulerConfig cfg)
	: pool_(pool), cfg_(cfg), total_blocks_((int)pool.num_free_blocks()) {}

	int add_request(std::vector<int> prompt, int max_new_tokens) {
		int worst = (int)prompt.size() + max_new_tokens;
		if (prompt.empty()
		 || worst > pool_.max_blocks_per_seq() * KV_BLOCK_SIZE
		 || blocks_needed(worst) > total_blocks_)
			return -1;
		int id = next_id_++;
		Request r;
		r.id = id;
		r.prompt_len = (int)prompt.size();
		r.token_ids = std::move(prompt);
		r.max_new_tokens = max_new_tokens;
		requests_.emplace(id, std::move(r));
		waiting_.push_back(id);
		return id;
	}

	StepPlan schedule() {
		StepPlan p = try_prefill();
		if (!p.empty()) return p;
		return try_decode();
	}

	std::vector<Request> update(const StepPlan &plan, const std::vector<int> &new_tokens) {
		std::vector<Request> done;
		for (size_t b = 0; b < plan.req_ids.size(); ++b) {
			auto it = requests_.find(plan.req_ids[b]);
			Request &r = it->second;
			r.token_ids.push_back(new_tokens[b]);
			bool stop = new_tokens[b] == cfg_.eos_token_id
			         || (int)r.token_ids.size() - r.prompt_len >= r.max_new_tokens;
			if (!stop) continue;
			pool_.release(r.slot);
			r.slot = -1;
			r.state = ReqState::FINISHED;
			running_.erase(std::find(running_.begin(), running_.end(), r.id));
			done.push_back(std::move(r));
			requests_.erase(it);
		}
		return done;
	}

	size_t num_waiting() const { return waiting_.size(); }
	size_t num_running() const { return running_.size(); }
	int num_preemptions() const { return preempt_count_; }

private:
	static int blocks_needed(int len) {
		return (len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
	}

	StepPlan try_prefill() {
		StepPlan p;
		p.is_prefill = true;
		int promised = 0, batched_tokens = 0;
		while (!waiting_.empty()) {
			Request &r = requests_.at(waiting_.front());
			int len = (int)r.token_ids.size();
			if ((int)running_.size() >= cfg_.max_num_seqs) break;
			// 如果已经有请求了, 且这条请求的长度会超预算, 就break, 或者目前没有请求, 直接放行
			if (!p.req_ids.empty() && batched_tokens + len > cfg_.max_num_batched_tokens) break;
			if (pool_.num_free_slots() == 0) break;
			int need = blocks_needed(len);
			int seqs_after = (int)running_.size() + 1;
			// 换句话说, 这行的意思是, 当前所有真正的空闲的物理块, 减去我承诺要分配的, 够不够分配
			// need是加入这块后, 需要分配的块数, seqs_after是我下一步正在的decode的, 他们最坏情况下会要一个新的block
			// 也就是我当前的情况, 够不够我的下一轮分配
			if ((int)pool_.num_free_blocks() - promised < need + seqs_after) break;
			r.slot = pool_.alloc_seq();
			r.state = ReqState::RUNNING;
			waiting_.pop_front();
			running_.push_back(r.id);
			p.req_ids.push_back(r.id);
			p.slots.push_back(r.slot);
			p.lens.push_back(len);
			promised += need;
			batched_tokens += len;
		}
		return p;
	}

	// decode 一步的块需求可以精确算出: 只有 len 恰好踩在块边界上的序列
	// 这一步才需要新块. 不够就抢占, 直到需求被余额盖住.
	StepPlan try_decode() {
		StepPlan p;
		while (!running_.empty() && decode_blocks_needed() > (int)pool_.num_free_blocks())
			preempt_last();
		for (int id : running_) {
			p.req_ids.push_back(id);
			p.slots.push_back(requests_.at(id).slot);
		}
		return p;
	}

	int decode_blocks_needed() const {
		int need = 0;
		for (int id : running_)
			if (pool_.seq_len(requests_.at(id).slot) % KV_BLOCK_SIZE == 0) ++need;
		return need;
	}

	// 重算式抢占: 牺牲最晚入场的, 保护先来的.
	// 释放它的全部块, 塞回 waiting 队头 (它已经排过队了, 不重新排);
	// KV 丢了没关系, token_ids 完整留在 CPU, 重新 prefill 即可.
	// 队头请求永远只进不退 → 至少它在持续推进, 抢占不会活锁.
	void preempt_last() {
		int id = running_.back();
		running_.pop_back();
		Request &r = requests_.at(id);
		pool_.release(r.slot);
		r.slot = -1;
		r.state = ReqState::WAITING;
		waiting_.push_front(id);
		++preempt_count_;
	}

	KV_Pool			&pool_;
	const SchedulerConfig	cfg_;
	const int		total_blocks_;	// 装配时的块总量, add_request 边界验证用
	int			next_id_ = 0;
	int			preempt_count_ = 0;	// 记录总共发生了多少次抢占
	std::unordered_map<int, Request> requests_;	// 所有权; FINISHED 的移交后即删除
	std::deque<int>		waiting_;	// FCFS: 尾进头出; 被抢占的插回队头
	std::vector<int>	running_;	// 本步 decode 的候选名单
};
