"""
[Phase 1] moe_layer_triton.py — 完整 Triton grouped-GEMM SwiGLU MoE（第二步）

做什么:
    用已验证的 grouped_gemm kernel 拼出完整 MoE 专家计算:
      排序分组(复用 torch_vec 思路) + 对齐到 BLOCK_M
      -> grouped GEMM #1: gate = x_sorted @ W_gate[e]
      -> grouped GEMM #2: up   = x_sorted @ W_up[e]
      -> 逐元素: h = silu(gate) * up         (规整 shape，普通 torch，非瓶颈)
      -> grouped GEMM #3: y    = h @ W_down[e]   (hidden 维变 intermediate，kernel 复用)
      -> 逆置换 + 加权散射回原 token

    对比前三版:
      naive/torch_vec: 逐专家小 GEMM(32x32 豆腐块，喂不饱)
      torch_bmm:       padding 到最长段(浪费 ~87%)
      本版(triton):    按 BLOCK_M 行 tile 切，热点段多占 tile、冷门段占 1 个 tile，
                       既无小 GEMM 也几乎无 padding(每段对齐浪费 < BLOCK_M 行)。

签名/输入/输出: 同 common_moe.MoEForward
    x(T,H), topk_idx(T,k), topk_weight(T,k), weights(ExpertWeights) -> out(T,H)
"""
import torch
import torch.nn.functional as F
from .common_moe import ExpertWeights
from .grouped_gemm_triton import grouped_gemm

BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64   # 探针实测 64x128x64 在大输入下最优
ALIGN = 64     # 段对齐粒度固定为 64，与 BLOCK_M 解耦，避免大 BLOCK_M 时冷门段 padding 暴增


def _sort_and_align(x, topk_idx, topk_weight, E):
    """展平 -> 按专家排序 -> 对齐每段到 BLOCK_M，构造 kernel 需要的 x_sorted / m_offsets。

    返回:
        x_sorted    : (N_aligned, H)  对齐后的排序输入(padding 行=0)
        m_offsets   : (E+1,) long     对齐后的段边界
        valid_pos   : (M,) long       每个有效行在 x_sorted 里的绝对位置(供取回结果)
        gather_token: (M,) long       每个有效行对应的原 token 行号
        sort_weight : (M,) float      每个有效行的权重(已按排序顺序)
    其中 M = T*k(有效配对总数)
    """
    T, H = x.shape
    k = topk_idx.shape[1]
    device = x.device

    flat_expert = topk_idx.reshape(-1)
    flat_token = torch.arange(T, device=device).repeat_interleave(k)
    flat_weight = topk_weight.reshape(-1)

    sort_expert, perm = torch.sort(flat_expert)
    gather_token = flat_token[perm]
    sort_weight = flat_weight[perm]
    x_valid = x[gather_token]                         # (M, H) 有效行，按专家排序但未对齐

    counts = torch.bincount(flat_expert, minlength=E) # (E,)
    aligned = ((counts + ALIGN - 1) // ALIGN) * ALIGN          # 关键行: 每段对齐到 ALIGN(=64)
    m_offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    m_offsets[1:] = torch.cumsum(aligned, dim=0)
    N = int(m_offsets[-1].item())

    # 计算每个有效行在【对齐后】x_sorted 中的绝对位置:
    #   该行所属专家段的对齐起点 + 段内序号
    valid_offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    valid_offsets[1:] = torch.cumsum(counts, dim=0)   # 有效行的(未对齐)前缀和
    pos_in_seg = torch.arange(x_valid.shape[0], device=device) - valid_offsets[sort_expert]
    valid_pos = m_offsets[sort_expert] + pos_in_seg   # 关键行: 有效行 -> 对齐 buffer 的绝对位置

    x_sorted = torch.zeros(N, H, dtype=x.dtype, device=device)
    x_sorted[valid_pos] = x_valid                     # 关键行: 有效行散射进对齐 buffer，其余为 0
    return x_sorted, m_offsets, valid_pos, gather_token, sort_weight


def moe_forward_triton(x, topk_idx, topk_weight, weights: ExpertWeights):
    E = weights.num_experts
    x_sorted, m_offsets, valid_pos, gather_token, sort_weight = _sort_and_align(
        x, topk_idx, topk_weight, E)

    # 三次 grouped GEMM 拼 SwiGLU。前两次 hidden->intermediate，第三次 intermediate->hidden。
    gate = grouped_gemm(x_sorted, weights.W_gate, m_offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    up = grouped_gemm(x_sorted, weights.W_up, m_offsets, BLOCK_M, BLOCK_N, BLOCK_K)
    h = (F.silu(gate.float()) * up.float()).to(x.dtype)        # 逐元素融合，规整 shape，非瓶颈
    y_sorted = grouped_gemm(h, weights.W_down, m_offsets, BLOCK_M, BLOCK_N, BLOCK_K)  # kernel 复用

    # 取回有效行 -> 加权 -> 逆置换散射回原 token
    y_valid = y_sorted[valid_pos]                              # (M, H)
    y_valid = y_valid * sort_weight.unsqueeze(-1).to(y_valid.dtype)
    out = torch.zeros_like(x)
    out.index_add_(0, gather_token, y_valid)                   # 一个 token 的 k 份在此求和
    return out