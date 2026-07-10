#pragma once
#include "../kv/KV_pool.h"
#include <deque>
#include <unordered_map>
#include <algorithm>

enum class ReqState { WAITING, RUNNING, FINISHED };

struct Request {
	int			id;
	std::vector<int>	token_ids;	// prompt + 已生成的token
	int			prompt_len;	// token_ids 前多少个是 prompt
	int			max_new_tokens;
	ReqState		state = ReqState::WAITING;
	int			slot = -1;	// 持有 KV 期间的 pool 行号 (追赶中或 RUNNING), 其余时刻 -1
	int			in_flight = 0;	// 已排定、update 还没消费的产 token 步数
	int			cached_len = 0;	// 已写入 KV 的 token 数 (追赶进度)
	int			scheduled_len = 0;	// 已经排定的KV进度
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
	std::vector<int>	req_ids;
	std::vector<int>	slots;
	std::vector<int>	lens;	// 每条目本步要算的 token 数, decode 条目为 1
	std::vector<int>	starts;	// 各条目的起始位置
	std::vector<int>	ids;
	bool empty() const { return req_ids.empty(); }
};

class Scheduler {
public:
	Scheduler(KV_Pool &pool, SchedulerConfig cfg)
	: pool_(pool), cfg_(cfg), total_blocks_((int)pool.num_free_blocks()) {
		// decode 永不停摆的前提: 全员 decode (每条 1 token) 也吃不完一步预算
		assert(cfg.max_num_seqs <= cfg.max_num_batched_tokens);
	}

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

	// 混合批: 一步 = decode 全员 (每条 len=1) + 剩余预算给队头的追赶 chunk.
	// decode 优先: 对延迟敏感, 且每条只花 1 token 预算 (max_num_seqs <= 预算, 必然装得下),
	// 于是 decode 不再因 prefill 停摆; 长 prompt 只慢它自己.
	// 调用前提: 所有已返回的 plan 均已 launch (append 已落账), num_free_blocks 才是新鲜的.
	StepPlan schedule(bool may_preempt = true) {
		StepPlan p;
		// decode 全员上车. 块不够先抢占
		while (!running_.empty() && decode_blocks_needed() > (int)pool_.num_free_blocks()) {
			if (!may_preempt) return p;
			preempt_last();
		}
		int batched_tokens = 0;		// token 预算的消费计数
		int promissed_blocks = 0;	// 本步已承诺、但 pool 还没扣的块
		int decode_reserve = 0;		// 本步 +1 后, 下一步 decode 踩线的序列数
		for (int id : running_) {
			Request &r = requests_.at(id);
			int generated = (int)r.token_ids.size() - r.prompt_len + r.in_flight;
			if (generated >= r.max_new_tokens) continue;
			r.in_flight++;
			p.req_ids.push_back(id);
			p.slots.push_back(r.slot);
			p.lens.push_back(1);
			p.starts.push_back(pool_.seq_len(r.slot));
			batched_tokens += 1;
			int len = pool_.seq_len(r.slot);
			if (len % KV_BLOCK_SIZE == 0)       promissed_blocks++;	// 本步 append 就要新块
			if ((len + 1) % KV_BLOCK_SIZE == 0) decode_reserve++;	// 下一步 append 要新块
		}
		// 剩余预算喂给追赶中的请求. FCFS: 队头拿不满就 break, 后面的不跳队.
		while (!waiting_.empty()) {
			int id = waiting_.front();
			Request &r = requests_.at(id);
			int target_len = (int)r.token_ids.size();
			// break 必须在 assert 之前: 被预算截断的队头本次已提交过一个 chunk,
			// 循环重访它时 scheduled_len 已领先 seq_len (plan 还没返回, 谈不上 launch),
			// 但此时预算必然已耗尽, 先 break 才轮不到哨兵误报.
			if (cfg_.max_num_batched_tokens - batched_tokens <= 0) break;
			if ((int)running_.size() + pending_seats_ >= cfg_.max_num_seqs) break;
			assert(r.scheduled_len < target_len);	// 排完的必已出队, 不该挂在 waiting_
			// 哨兵: schedule 入口处排定进度与 pool 落账一致, 抓 driver 漏 launch
			assert(r.slot == -1 || r.scheduled_len == pool_.seq_len(r.slot));
			int chunk_len = std::min(
				target_len - r.scheduled_len,
				cfg_.max_num_batched_tokens - batched_tokens);
			assert(chunk_len > 0);	// 零进度 chunk 一旦被准入, 系统会无声空转
			int old_blocks = blocks_needed(r.scheduled_len);
			int new_blocks = blocks_needed(r.scheduled_len + chunk_len);
			int need_blocks = new_blocks - old_blocks;
			bool is_final = (r.scheduled_len + chunk_len == target_len);
			// 追平后的 len 恰好踩块边界时, 它下一步的首个 decode token 就要新块
			bool ends_on_boundary = (target_len % KV_BLOCK_SIZE == 0);
			// 预留"明天的账": 保证本次准入不会成为下一步 decode 抢占的直接原因.
			// 预留量只由需求侧决定, 不引用空闲量 —— 否则块紧张时保护恰好失效.
			int reserve = decode_reserve + pending_boundary_ + ((is_final && ends_on_boundary) ? 1 : 0);
			if ((int)pool_.num_free_blocks() - promissed_blocks < need_blocks + reserve) break;
			if (r.slot == -1 && pool_.num_free_slots() == 0) break;
			// —— 检查全部通过, 以下开始提交 ——
			if (r.slot == -1) r.slot = pool_.alloc_seq();
			p.starts.push_back(r.scheduled_len);
			p.ids.insert(p.ids.end(),
    				r.token_ids.begin() + r.scheduled_len,
        			r.token_ids.begin() + r.scheduled_len + chunk_len);
			r.scheduled_len += chunk_len;
			p.req_ids.push_back(id);
			p.slots.push_back(r.slot);
			p.lens.push_back(chunk_len);
			batched_tokens += chunk_len;
			promissed_blocks += need_blocks;
			if (is_final) {
				waiting_.pop_front();
				r.in_flight += 1;
				pending_seats_ += 1;
				if (ends_on_boundary) pending_boundary_ += 1;
			}
		}
		return p;
	}

	std::vector<Request> update(const StepPlan &plan, const std::vector<int> &new_tokens) {
		std::vector<Request> done;
		for (size_t b = 0; b < plan.req_ids.size(); ++b) {
			auto it = requests_.find(plan.req_ids[b]);
			if (it == requests_.end()) continue;
			Request &r = it->second;

			r.cached_len += plan.lens[b];
			if (r.cached_len < (int)r.token_ids.size()) continue;
			r.in_flight--;
			assert(r.in_flight >= 0);	// 负数说明有产 token 的步没被记账
			if (r.state != ReqState::RUNNING) {
				pending_seats_--;
				if (r.cached_len % KV_BLOCK_SIZE == 0) pending_boundary_--;
				assert(pending_seats_ >= 0 && pending_boundary_ >= 0);
			}
			r.token_ids.push_back(new_tokens[b]);
			if (!(
				new_tokens[b] == cfg_.eos_token_id ||
				(int)r.token_ids.size() - r.prompt_len >= r.max_new_tokens
			)){
				// 如果能走到这儿, 肯定是产生了token
				// 这个token不是eos, 也没有到长度的限制
				// 那么他肯定就进入了稳定产生一个token的decode阶段了
				if (r.state != ReqState::RUNNING) {
					r.state = ReqState::RUNNING;
					running_.push_back(r.id);
				}
				continue;
			}
			// 第 N-1 步的请求eos → update(N-1) → 进 pending_release_
			// 第 N 步的 kernel 还在引用这些 slot 的物理块
			// decode kernel 会往这些块里 WRITE 一条新 KV entry
			// 第 N 步完成，event 同步确认 → flush_release() → 现在才真正归还
			// update(N) → 可能产生新的死者 → 进 pending_release_, 留给下一轮处理

			// 第 N-1 步eos → 直接 release → 物理块回到 free_list
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

	// decode 一步的块需求可以精确算出: 只有 len 恰好踩在块边界上的序列
	// 这一步才需要新块. 不够就抢占, 直到需求被余额盖住.
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
		r.scheduled_len = 0;
		r.state = ReqState::WAITING;
		waiting_.push_front(id);
		++preempt_count_;
	}

	KV_Pool			&pool_;
	const SchedulerConfig	cfg_;
	const int		total_blocks_;	// 装配时的块总量, add_request 边界验证用
	int			next_id_ = 0;
	int			preempt_count_ = 0;	// 记录总共发生了多少次抢占
	int			pending_seats_ = 0;	// final chunk已经确定了, 但是当前还在waiting
	int			pending_boundary_ = 0;	// 其中追平后恰好踩在边界的条数
	std::unordered_map<int, Request> requests_;	// 所有权; FINISHED 的移交后即删除
	std::deque<int>		waiting_;	// FCFS: 尾进头出; 被抢占的插回队头
	std::vector<int>	running_;	// 已追平的序列: 每步作为 decode 条目优先上车
	std::vector<int>	pending_release_;	// update 时入队, flush_releases 时归还
};
