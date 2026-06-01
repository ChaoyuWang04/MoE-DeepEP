"""
[Phase 1] grouped_gemm_fused.py — 融合版 grouped GEMM kernel（gate+up 合并 + SILU epilogue + autotune）

做什么:
    1. fused_gate_up_silu: 一个 kernel 同时算 gate=x@Wg 与 up=x@Wu，读一遍 x_sorted，
       K 循环里并行累加两个 acc，循环结束在寄存器里直接 silu(gate)*up -> h。
       替代"2 次 GEMM 读 2 遍 x + 1 次独立 silu kernel(实测 3.5ms)"。
    2. down GEMM 复用 grouped_gemm_triton.grouped_gemm。
    3. autotune 搜索 BLOCK_N/BLOCK_K/num_stages/num_warps(提 occupancy)。

    autotune 用法要点(踩坑总结，记 design_notes):
      a) 交给 @triton.autotune 搜索的符号(BLOCK_*/num_stages/num_warps)，launch 时【不能】
         再手动传，否则 "Conflicting meta-parameters"。
      b) grid 依赖 BLOCK_*(autotune 动态选)，必须写成 grid=lambda META: ...，从 META 取。
      c) tile_expert 依赖 BLOCK_M。为避免 tile_expert 随 BLOCK_M 变化的复杂性，
         configs 里【固定 BLOCK_M=64】，只搜 BLOCK_N/BLOCK_K/stages/warps —— tile_expert 按
         BLOCK_M=64 预算一次即可。(段对齐 ALIGN=64 与之匹配，保证 tile 不跨专家。)

输入: x_sorted(N,H) 已排序+对齐; Wg/Wu(E,H,I); m_offsets(E+1,)
输出: h(N,I) = silu(x@Wg[e]) * (x@Wu[e])
"""
import torch
import triton
import triton.language as tl
from .grouped_gemm_triton import _build_tile_expert, grouped_gemm

_FIXED_BLOCK_M = 64   # tile_expert 按此预算；configs 的 BLOCK_M 必须都等于它


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['N', 'H', 'I'],   # 这些值变化时重新搜索；不变则复用最优 config
)
@triton.jit
def _fused_gate_up_silu_kernel(
    x_ptr, wg_ptr, wu_ptr, h_ptr,
    tile_expert_ptr, m_offsets_ptr,
    N, H, I,
    stride_xn, stride_xh,
    stride_we, stride_wh, stride_wi,
    stride_hn, stride_hi,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(tile_expert_ptr + pid_m)

    row_start = pid_m * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    seg_end = tl.load(m_offsets_ptr + e + 1)
    row_mask = offs_m < seg_end

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    wg_base = wg_ptr + e * stride_we
    wu_base = wu_ptr + e * stride_we
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < H
        x_blk = tl.load(
            x_ptr + offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xh,
            mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        wg_blk = tl.load(
            wg_base + offs_k[:, None] * stride_wh + offs_n[None, :] * stride_wi,
            mask=k_mask[:, None] & (offs_n[None, :] < I), other=0.0)
        wu_blk = tl.load(
            wu_base + offs_k[:, None] * stride_wh + offs_n[None, :] * stride_wi,
            mask=k_mask[:, None] & (offs_n[None, :] < I), other=0.0)
        acc_g += tl.dot(x_blk, wg_blk)
        acc_u += tl.dot(x_blk, wu_blk)

    gate_silu = acc_g * tl.sigmoid(acc_g)     # silu(z)=z*sigmoid(z)
    h_val = gate_silu * acc_u
    h_mask = row_mask[:, None] & (offs_n[None, :] < I)
    tl.store(h_ptr + offs_m[:, None] * stride_hn + offs_n[None, :] * stride_hi,
             h_val.to(h_ptr.dtype.element_ty), mask=h_mask)


def fused_gate_up_silu(x_sorted, Wg, Wu, m_offsets, *_ignored, **_kw):
    """host 包装(autotune 版)。BLOCK_* 不再手动传，由 autotune 决定；grid 用 lambda 取 META。

    *_ignored / **_kw: 吸收旧调用里传入的 BLOCK_M/N/K，保持向后兼容(被忽略)。
    """
    N, H = x_sorted.shape
    E, H2, I = Wg.shape
    assert H == H2
    h = torch.empty((N, I), dtype=x_sorted.dtype, device=x_sorted.device)

    # tile_expert 按固定 BLOCK_M=64 预算(configs 的 BLOCK_M 全是 64)
    num_tiles_m = triton.cdiv(N, _FIXED_BLOCK_M)
    tile_expert = _build_tile_expert(m_offsets, _FIXED_BLOCK_M, num_tiles_m)

    # 关键: grid 写成 lambda，从 META 取 autotune 当前选中的 BLOCK_N(BLOCK_M 固定 64)
    grid = lambda META: (num_tiles_m, triton.cdiv(I, META['BLOCK_N']))
    _fused_gate_up_silu_kernel[grid](
        x_sorted, Wg, Wu, h, tile_expert, m_offsets,
        N, H, I,
        x_sorted.stride(0), x_sorted.stride(1),
        Wg.stride(0), Wg.stride(1), Wg.stride(2),
        h.stride(0), h.stride(1),
        # 注意: 不传 BLOCK_*/num_stages/num_warps —— 全由 autotune 决定
    )
    return h