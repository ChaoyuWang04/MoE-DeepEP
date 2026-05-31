"""
[Phase 2] dispatch_combine_optimized.py — 5 轴优化版 dispatch/combine

做什么:
    在 naive 版基础上沿 5 个优化轴改写并逐项 benchmark:
      轴1 通信轮数: dispatch 3 次 all2all -> 2 次
          recv_splits = recv_expert_counts.view(world_size, E_local).sum(dim=1)
      轴2 通信量:   dispatch 用 FP8；同目标 rank 的重复 token 去重只发一次
      轴3 重叠:     见 overlap.py
      轴4 执行效率: 干掉所有 .item() 同步，路由用 argsort/bincount 向量化
      轴5 显存:     packed 紧凑 buffer 取代 (E_local, T*world_size, H) 大 padding

输入/输出: 与 dispatch_combine_naive 对齐（便于 tests 直接对拍）
原则: 先自己实现，再看老师 improved 版查漏补缺。
"""
import torch


def dispatch_opt(x, topk_idx, num_tokens, num_experts, world_size, rank):
    raise NotImplementedError("Phase 2 优化: 先自己写，再对老师 improved 版")


def combine_opt(*args, **kwargs):
    raise NotImplementedError
