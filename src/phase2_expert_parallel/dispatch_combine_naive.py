"""
[Phase 2] dispatch_combine_naive.py — 专家并行 dispatch/combine（清理版基准）

做什么:
    把老师的原始对拍代码（见 references/teacher_original_dispatch_combine.py）整理成
    可读的"朴素但正确"基准: dispatch 用 3 次 all-to-all，combine 用 2 次。
    本文件是 Phase 2 优化的对照组，正确性由 tests/test_dispatch_combine.py 守护。

并行设定:
    专家切到 world_size 个 rank，每 rank 持有 num_experts//world_size 个本地专家。
    每 rank 自带一批 token，token 选中的专家可能在别的 rank -> 需要 all-to-all。

dispatch 输入:
    x        : (T, H)   本 rank 的 token
    topk_idx : (T, k)   路由（来自 Phase 0 真实 trace）
dispatch 输出:
    recv_x        : (E_local, max_len, H) 按本地专家分桶的接收 buffer
    recv_count    : (E_local,)            每个本地专家实收 token 数
    token_map     : 收据，combine 用它把结果认领回原 token
    expert_counts : 每个 (源 rank, 本地专家) 的计数，combine 反向路由用

铁律: 变长 all-to-all 必须"先报数(metadata) 再发数据"——接收 buffer 须先开好。
"""
import torch
import torch.distributed as dist


def dispatch_naive(x, topk_idx, num_tokens, num_experts, world_size, rank):
    # TODO(Phase 2): 从 references/teacher_original 迁移，逐次 all2all 加注释
    #   all2all #1: send_splits -> recv_splits（每 rank 总数，粗）
    #   all2all #2: expert_counts -> recv_expert_counts（每专家数，细）
    #   all2all #3: 真正的 token 数据
    raise NotImplementedError("Phase 2: 迁移并讲清每次 all2all 在干嘛")


def combine_naive(*args, **kwargs):
    # TODO(Phase 2): dispatch 的逆操作 = 寄回(2 次 all2all) + 本地按 token_map 加权合成
    raise NotImplementedError
