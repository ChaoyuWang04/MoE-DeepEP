"""
[Phase 1] moe_layer_optimized.py — 单卡 MoE（torch 向量化版）

做什么:
    用 "展平 -> 按专家 argsort 排序置换 -> bincount/cumsum 算分组边界 -> 分段大 GEMM
    -> 逆置换 + 加权散射" 替换朴素版的 64 次逐专家循环。
    消除 launch 开销与冷门专家空跑；GEMM 仍用 torch（分段），是理解 Triton grouped GEMM 的台阶。

    核心思想（记进 design_notes）:
        向量化的本质不是"算得快"，而是把"按专家分散查找"换成
        "先排序让同专家连续、再批量处理"——用一次排序换掉 64 次 kernel launch。

签名/输入/输出: 见 common_moe.MoEForward
    x(T,H), topk_idx(T,k), topk_weight(T,k), weights(ExpertWeights) -> out(T,H)
"""
import torch
import torch.nn.functional as F
from .common_moe import ExpertWeights


def moe_forward_optimized(x, topk_idx, topk_weight, weights: ExpertWeights):
    T, H = x.shape
    k = topk_idx.shape[1]
    E = weights.num_experts
    device = x.device

    # --- 1. 展平: (T,k) -> (T*k,)。每个元素是一个 (token, slot) 配对选中的专家 ---
    flat_expert = topk_idx.reshape(-1)                    # (T*k,) 每个配对选的专家 id
    flat_token = torch.arange(T, device=device).repeat_interleave(k)  # (T*k,) 该配对属于哪个 token
    flat_weight = topk_weight.reshape(-1)                 # (T*k,) 该配对的权重

    # --- 2. 按专家 id 排序: 让选同一专家的配对在物理上连续 ---
    # 关键行: argsort 返回排列 perm；perm[i]=排序后第 i 位原本的下标
    sort_expert, perm = torch.sort(flat_expert)
    gather_token = flat_token[perm]                       # 排序后每个位置对应的原 token 行号
    x_sorted = x[gather_token]                            # (T*k, H) 关键行: 按专家分段排好的输入

    # --- 3. 分组边界: 每个专家在排序后占哪一段 ---
    counts = torch.bincount(flat_expert, minlength=E)     # (E,) 每个专家有多少配对
    offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)             # 关键行: 前缀和 -> 每段 [offsets[e], offsets[e+1])

    # --- 4. 分段 GEMM: 逐专家对其连续段做一次大 SwiGLU（段大小随倾斜剧烈不同）---
    y_sorted = torch.empty_like(x_sorted)
    counts_cpu = counts.tolist()                          # 关键行: 一次性搬到 CPU，避免循环内逐次 .item() 同步
    for e in range(E):
        s, ein = int(offsets[e]), int(offsets[e + 1])
        if ein == s:
            continue                                      # 冷门专家段为空，跳过
        xe = x_sorted[s:ein]                              # (n_e, H) 该专家的所有 token
        gate = F.silu(xe @ weights.W_gate[e])             # (n_e, I)
        up = xe @ weights.W_up[e]
        y_sorted[s:ein] = ((gate * up) @ weights.W_down[e]).to(y_sorted.dtype)

    # --- 5. 加权 + 逆置换散射回原 token 顺序 ---
    y_sorted = y_sorted * flat_weight[perm].unsqueeze(-1).to(y_sorted.dtype)  # 排序态下先乘权重
    out = torch.zeros_like(x)
    # 关键行: index_add_ 把每个配对的结果按原 token 行号累加（一个 token 的 k 份在此求和）
    out.index_add_(0, gather_token, y_sorted)
    return out