"""
[Phase 1] moe_layer_naive.py — 单卡 MoE 专家计算（朴素版，作正确性基准）

做什么:
    单张 GPU 上完整走一遍 "按专家分发 -> 逐专家 FFN -> 加权合成"，用最直白的循环实现，
    便于对照验证。故意慢，作为 Phase 1 优化(向量化版)的数值基准与速度基线。

输入:
    x          : (T, H)        本层输入 token
    topk_idx   : (T, k)        每个 token 选中的专家
    topk_weight: (T, k)        对应权重
    experts    : list[Module]  每个专家一个 FFN
输出:
    out        : (T, H)        加权合成后的输出（形状回到输入）
"""
import torch


def moe_forward_naive(x, topk_idx, topk_weight, experts):
    out = torch.zeros_like(x)
    for e in range(len(experts)):                    # 关键行: 逐专家串行——慢点所在
        mask = (topk_idx == e)                        # (T, k) 哪些 (token,k) 选了专家 e
        if mask.sum() == 0:
            continue
        tok_ids, k_ids = mask.nonzero(as_tuple=True)
        y = experts[e](x[tok_ids])                    # 该专家对这些 token 做 FFN
        w = topk_weight[tok_ids, k_ids].unsqueeze(-1) # 对应权重
        out.index_add_(0, tok_ids, y * w)             # 关键行: 加权累加回原 token 位置
    return out
