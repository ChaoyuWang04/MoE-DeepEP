"""
[Phase 1] moe_layer_bmm.py — torch 的"真·批量"变体（验证 per-expert launch 是瓶颈）

做什么:
    与 moe_layer_optimized 一样先排序分组，但把"逐专家 for 循环小 GEMM"换成:
    把所有专家段 padding 到同一长度 cap，堆成 (E, cap, H)，用 torch.bmm 一次性批量算。
    => 计算部分从"64 次 GEMM launch"变成"3 次 bmm launch(gate/up/down)"。
    若它明显快于 torch_vec，就证明: 拖后腿的正是 per-expert GEMM launch，而非排序/同步。

代价(故意暴露):
    padding 到 cap=max(段长) 会浪费算力 —— 这正是为什么真实方案用 Triton grouped GEMM
    (变长、不 padding)，而不是 bmm。这个变体只为验证瓶颈，不是最终方案。

签名/输入/输出: 同 common_moe.MoEForward
"""
import torch
import torch.nn.functional as F
from .common_moe import ExpertWeights


def moe_forward_bmm(x, topk_idx, topk_weight, weights: ExpertWeights):
    T, H = x.shape
    k = topk_idx.shape[1]
    E = weights.num_experts
    I = weights.intermediate
    device = x.device

    flat_expert = topk_idx.reshape(-1)
    flat_token = torch.arange(T, device=device).repeat_interleave(k)
    flat_weight = topk_weight.reshape(-1)

    sort_expert, perm = torch.sort(flat_expert)
    gather_token = flat_token[perm]
    x_sorted = x[gather_token]                            # (N=T*k, H) 按专家排好

    counts = torch.bincount(flat_expert, minlength=E)     # (E,)
    offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    cap = int(counts.max().item()) if counts.numel() else 0   # 关键行: padding 目标长度=最长段
    if cap == 0:
        return torch.zeros_like(x)

    # --- 把每个专家段散射进规整张量 (E, cap, H)，空位为 0 ---
    xb = torch.zeros(E, cap, H, dtype=x.dtype, device=device)
    # 每个 token 在其专家段内的"段内位置" = 全局位置 - 段起点
    pos_in_seg = torch.arange(x_sorted.shape[0], device=device) - offsets[sort_expert]
    xb[sort_expert, pos_in_seg] = x_sorted                # 关键行: 一次性散射，无 for 循环

    # --- 真·批量: 3 次 bmm 把所有专家一起算完(而非 64 次) ---
    gate = F.silu(torch.bmm(xb, weights.W_gate))          # (E,cap,I)  关键行: 1 次 bmm 代替 64 次
    up = torch.bmm(xb, weights.W_up)                      # (E,cap,I)
    yb = torch.bmm(gate * up, weights.W_down)             # (E,cap,H)

    # --- 取回有效位置，加权，逆置换散射回原 token ---
    y_sorted = yb[sort_expert, pos_in_seg]                # (N,H) 收回有效行
    y_sorted = y_sorted * flat_weight[perm].unsqueeze(-1).to(y_sorted.dtype)
    out = torch.zeros_like(x)
    out.index_add_(0, gather_token, y_sorted)
    return out