"""
[Phase 1] moe_layer_naive.py — 单卡 MoE 专家计算（朴素版，正确性 & 性能基准）

做什么:
    最直白实现: 逐专家循环，每个专家挑出选中它的 token、做一次小 SwiGLU FFN、加权写回。
    故意保留 64 次循环 -> 64 次独立 kernel launch。在 10x 负载倾斜下，58 个冷门专家
    几乎空跑，暴露"launch 开销 + GPU 空泡"这一真正瓶颈（不是算力）。它是另两版的对拍基准。

签名: 见 common_moe.MoEForward
输入: x(T,H), topk_idx(T,k), topk_weight(T,k), weights(ExpertWeights)
输出: out(T,H)
"""
import torch
from .common_moe import ExpertWeights, single_expert_ffn


def moe_forward_naive(x, topk_idx, topk_weight, weights: ExpertWeights):
    out = torch.zeros_like(x)
    for e in range(weights.num_experts):                  # 关键行: 64 次循环 = 64 次 kernel launch
        mask = (topk_idx == e)                            # (T,k) 哪些 (token,slot) 选了专家 e
        if not mask.any():
            continue                                      # 冷门专家直接跳过(但循环本身仍有开销)
        tok_ids, slot_ids = mask.nonzero(as_tuple=True)   # 选了 e 的 token 行号 / 第几个 slot
        y = single_expert_ffn(x[tok_ids], weights, e)     # 关键行: 一次小 GEMM(行数随倾斜剧烈波动)
        w = topk_weight[tok_ids, slot_ids].unsqueeze(-1)  # (n_e,1) 对应权重
        out.index_add_(0, tok_ids, (y * w).to(out.dtype)) # 关键行: 加权累加回原 token 位置
    return out