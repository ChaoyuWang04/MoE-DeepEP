"""
[Phase 2] expert_compute_fused.py — 用 Phase 1 融合 Triton kernel 做本地专家计算

做什么 (Phase1/Phase2 合流):
    Phase 2 的 dispatch 已把"命中本 rank 专家的 token"按本地专家分组送来(recv_x)。
    本 rank 只需用【本地 E_local 个专家】算这批 token —— 不需要再排序/combine(那是 dispatch/
    combine 的职责)。本文件复用 Phase 1 的融合 grouped GEMM kernel 完成这段本地计算，
    替代 ep_dist_naive.expert_compute 的逐专家 for 循环(小 GEMM 慢)。

    与 Phase 1 forward 的区别:
      Phase1 moe_forward_triton_fused: 全局视角，内部自排序+算+combine。
      本文件: 只做"本地专家计算"一段，输入已按 (源rank, 本地专家) 排好，直接构造段边界喂 kernel。

输入:
    recv_x            : (N_recv, H)   dispatch 送来的 token(已按 源rank→本地专家 排列)
    recv_per_expert   : (E_local 或 world_size*E_local 视实现)  每段行数，用来构造段边界
    weights_local     : ExpertWeights 仅本 rank 的 E_local 个专家
    world_size        : int
输出:
    out               : (N_recv, H)   每行用其所属本地专家算出的 FFN 结果(顺序不变)

注意段边界构造:
    recv_x 排列是"源 rank 0 的[本地专家0..E_local-1] | 源 rank 1 的[本地专家0..] | ..."。
    同一个本地专家 le 的 token 分散在各源 rank 段里(不连续!)。grouped GEMM 要求"同专家连续"，
    所以这里要先把 recv_x 按【本地专家】重排聚到一起，算完再还原顺序。
"""
import torch
from ..phase1_single_gpu_moe.common_moe import ExpertWeights
from ..phase1_single_gpu_moe.grouped_gemm_fused import fused_gate_up_silu
from ..phase1_single_gpu_moe.grouped_gemm_triton import grouped_gemm
from ..phase1_single_gpu_moe import moe_layer_triton as mt

BLOCK_M, BLOCK_N, BLOCK_K = mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K
ALIGN = mt.ALIGN


def expert_compute_fused(recv_x, recv_local_expert, weights_local: ExpertWeights):
    """用融合 Triton kernel 算本地专家。recv_local_expert 给出每行的本地专家 id。

    步骤: 按本地专家排序聚拢 -> 对齐分段 -> fused_gate_up_silu + down grouped GEMM -> 还原顺序。
    """
    device = recv_x.device
    N, H = recv_x.shape
    E_local = weights_local.num_experts

    if N == 0:
        return recv_x.clone()

    # 1. 按本地专家排序，让同专家 token 连续(dispatch 后同一本地专家的行分散在各源 rank 段)
    order = torch.argsort(recv_local_expert)                 # 关键行: 聚拢同专家
    inv_order = torch.argsort(order)                         # 逆置换，算完还原原顺序
    x_sorted_valid = recv_x[order]

    # 2. 段边界: 每个本地专家多少行 -> 对齐到 ALIGN -> 构造对齐 buffer(空位填0)
    counts = torch.bincount(recv_local_expert, minlength=E_local)
    aligned = ((counts + ALIGN - 1) // ALIGN) * ALIGN
    m_offsets = torch.zeros(E_local + 1, dtype=torch.long, device=device)
    m_offsets[1:] = torch.cumsum(aligned, dim=0)
    M = int(m_offsets[-1].item())

    valid_off = torch.zeros(E_local + 1, dtype=torch.long, device=device)
    valid_off[1:] = torch.cumsum(counts, dim=0)
    seg_id = torch.repeat_interleave(torch.arange(E_local, device=device), counts)  # 每行所属专家
    pos_in_seg = torch.arange(x_sorted_valid.shape[0], device=device) - valid_off[seg_id]
    valid_pos = m_offsets[seg_id] + pos_in_seg               # 有效行 -> 对齐 buffer 绝对位置

    x_sorted = torch.zeros(M, H, dtype=recv_x.dtype, device=device)
    x_sorted[valid_pos] = x_sorted_valid

    # 3. 融合 grouped GEMM: gate+up+silu 一个 kernel，down 一个 kernel
    h = fused_gate_up_silu(x_sorted, weights_local.W_gate, weights_local.W_up, m_offsets,
                           BLOCK_M, BLOCK_N, BLOCK_K)
    y_sorted = grouped_gemm(h, weights_local.W_down, m_offsets, BLOCK_M, BLOCK_N, BLOCK_K)

    # 4. 取回有效行 -> 还原成 recv_x 的原始顺序(供 combine 按原路径寄回)
    y_valid = y_sorted[valid_pos]                            # (N, H) 按"本地专家排序"的顺序
    out = y_valid[inv_order]                                 # 关键行: 逆置换回 recv_x 原顺序
    return out