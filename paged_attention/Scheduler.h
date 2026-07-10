#pragma once
#include "KV_pool.h"
#include <deque>
#include <unordered_map>
#include <algorithm>

enum class ReqState { WAITING, PREFILLING, RUNNING, FINISHED };

struct Request {
	int			id;
	std::vector<int>	token_ids;	// prompt + 已生成的token
	int			prompt_len;	// token_ids 前多少个是 prompt
	int			max_new_tokens;
	ReqState		state = ReqState::WAITING;
	int			slot = -1;	// RUNNING 期间持有的 pool 行号, 其余时刻 -1
	int			in_flight = 0;
	int			cached_len = 0;	// prefill chunk进的行数
};

struct SchedulerConfig {
	int	max_num_seqs;		// running 数上限, 即每步 B 的上限
	int	max_num_batched_tokens;	// 单步 prefill 的 token 总预算; 同时也是 chunk 粒度:
					// 长 prompt 被预算截断成块, 每步至多产生一个 partial 且必在队头
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

	StepPlan schedule(bool may_preempt = true) {
		StepPlan p = try_prefill();
		if (!p.empty()) return p;
		return try_decode(may_preempt);
	}

	std::vector<Request> update(const StepPlan &plan, const std::vector<int> &new_tokens) {
		std::vector<Request> done;
		std::vector<int> partial;
		for (size_t b = 0; b < plan.req_ids.size(); ++b) {
			auto it = requests_.find(plan.req_ids[b]);
			if (it == requests_.end()) continue;
			Request &r = it->second;

			if (plan.is_prefill) {
				r.cached_len += plan.lens[b];
				if (r.cached_len < r.token_ids.size()) {
					r.state = ReqState::WAITING;
					partial.push_back(r.id);
					continue;
				}
			}
			r.in_flight--;
			assert(r.in_flight >= 0);	// 负数说明有产 token 的步没被记账 (prefill 或 decode)
			r.token_ids.push_back(new_tokens[b]);
			if (!(
				new_tokens[b] == cfg_.eos_token_id ||
				r.token_ids.size() - r.prompt_len >= r.max_new_tokens
			)){
				if (plan.is_prefill) {
					r.state = ReqState::RUNNING;
					running_.push_back(r.id);
				}
				continue;
			}
			// 第 N-1 步的请求 EOS → update(N-1) → 进 pending_release_
			// 第 N 步的 kernel 还在引用这些 slot 的物理块
			// decode kernel 会往这些块里 WRITE 一条新 KV entry
			// 第 N 步完成，event 同步确认 → flush_release() → 现在才真正归还
			// update(N) → 可能产生新的死者 → 进 pending_release_, 留给下一轮处理

			// 第 N-1 步 EOS → 直接 release → 物理块回到 free_list
			// 下一次 schedule() 把这些块分给新请求 → 新请求 prefill 写入新 KV
			// 但第 N 步的 kernel 还在 GPU 上跑，它也要往同一批物理块里写 KV
			// 就会发生之前的同一片 GPU 显存上的写-写冲突的BUG
			if (r.slot != -1) {
				pending_release_.push_back(r.slot);
				r.slot = -1;
			}
			r.state = ReqState::FINISHED;
			auto rpos = std::find(running_.begin(), running_.end(), r.id);
			if (rpos != running_.end()) {
				running_.erase(rpos);
			} else {
				auto wpos = std::find(waiting_.begin(), waiting_.end(), r.id);
				if (wpos != waiting_.end()) waiting_.erase(wpos);
			}
			done.push_back(std::move(r));
			requests_.erase(it);
		}
		waiting_.insert(waiting_.begin(), partial.begin(), partial.end());
		return done;
	}

	void flush_release() {
		for (int slot : pending_release_) pool_.release(slot);
		pending_release_.clear();
	}

	size_t num_waiting() const { return waiting_.size(); }
	size_t num_running() const { return running_.size(); }
	int num_preemptions() const { return preempt_count_; }

private:
	static int blocks_needed(int len) {
		return (len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
	}

	// 在chunked prefill里面, 要注意的是, 现在一个request有两种语意
	// 可能是已经在run的, 并且最后一个chunk, 会产出token
	// 另一种是不是最后一个chunk, 这种不会产出新的token, 并且这种情况下需要一直抢占slot
	// 伴随而来的, 在waiting_队列里的, 也有两种request
	// 	1.完全还没进KV Pool	slot == -1, cached_len == 0
	// 	2.已经写了一部分cache了	slot != -1, cached_len > 0
	// 因为进入waitting的一定至少跑了一个chunk或者一个chunk都没跑, 所以不可能出现slot != -1 && cached_len == 0的情况
	StepPlan try_prefill() {
		StepPlan p;
		p.is_prefill = 1;
		int batched_tokens = 0;
		int promissed_blocks = 0;
		int promised_decode = 0;	// 本轮准入的 final chunk 里, 完成后首步 decode 就要新块的条数
		int decode_reserve = decode_blocks_needed();	// running_ 在本轮内不变, 是循环不变量
		while (!waiting_.empty()) {
			int id = waiting_.front();
			Request &r = requests_.at(id);
			int target_len = r.token_ids.size();
			assert(r.cached_len < target_len);	// 如果这儿出现了已经完成了最后一个chunk但是还是进入了try_prefill的情况一定是bug
			if (cfg_.max_num_batched_tokens - batched_tokens <= 0) break;
			if ((int)running_.size() >= cfg_.max_num_seqs) break;
			int chunk_len = std::min(
				target_len - r.cached_len,
				cfg_.max_num_batched_tokens - batched_tokens);
			assert(chunk_len > 0);	// 零进度 chunk 一旦被准入, 系统会无声空转
			int old_blocks = blocks_needed(r.cached_len);
			int new_blocks = blocks_needed(r.cached_len + chunk_len);
			int need_blocks = new_blocks - old_blocks;
			bool is_final = (r.cached_len + chunk_len == target_len);
			// prefill 完的 len 恰好踩块边界时, 首个 decode token 就要新块
			bool ends_on_boundary = (target_len % KV_BLOCK_SIZE == 0);
			// 保证本次准入不会成为下一个 decode 步抢占的直接原因.
			// 预留量只由需求侧决定, 不引用空闲量 —— 否则块紧张时保护恰好失效.
			int reserve = decode_reserve + promised_decode
				+ ((is_final && ends_on_boundary) ? 1 : 0);
			if ((int)pool_.num_free_blocks() - promissed_blocks < need_blocks + reserve) break;
			if (r.slot == -1 && pool_.num_free_slots() == 0) break;
			// —— 检查全部通过, 以下开始提交 ——
			if (r.slot == -1) {
				r.slot = pool_.alloc_seq();
				r.state = ReqState::PREFILLING;
			}
			waiting_.pop_front();
			p.req_ids.push_back(id);
			p.slots.push_back(r.slot);
			p.lens.push_back(chunk_len);
			batched_tokens += chunk_len;
			promissed_blocks += need_blocks;
			if (is_final && ends_on_boundary) promised_decode += 1;
			if (is_final) r.in_flight += 1;
		}
		return p;
	}

	// decode 一步的块需求可以精确算出: 只有 len 恰好踩在块边界上的序列
	// 这一步才需要新块. 不够就抢占, 直到需求被余额盖住.
	StepPlan try_decode(bool may_preempt = true) {
		StepPlan p;
		while (!running_.empty() && decode_blocks_needed() > (int)pool_.num_free_blocks()){
			if (!may_preempt) return p;
			preempt_last();
		}
		for (int id : running_) {
			Request &r = requests_.at(id);
			int generated = (int)r.token_ids.size() - r.prompt_len + r.in_flight;	// 这个请求承诺要生成的 token 总数
			if (generated >= r.max_new_tokens) continue;
			r.in_flight++;		// 前一步的update可能还没回来
			p.req_ids.push_back(id);
			p.slots.push_back(r.slot);
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
		r.cached_len = 0;
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
	std::vector<int>	pending_release_;	// update 时入队, flush_releases 时归还
};
