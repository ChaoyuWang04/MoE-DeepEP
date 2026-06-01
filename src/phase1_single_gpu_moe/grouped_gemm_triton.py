"""
[Phase 1] grouped_gemm_triton.py — Triton 分组 GEMM（v3: 预计算 tile→expert + autotune）

做什么:
    对"已按专家排序、变长拼接、每段对齐到 ALIGN"的 x_sorted (N,H)，每段乘各自权重 W[e]
    (H,K) -> y (N,K)。切块单位是 BLOCK_M 行 tile，每 tile 由 host 预计算的 tile_expert
    O(1) 查表定专家(去掉 v1 的 O(E) 扫描)。autotune 搜 BLOCK_N/K/stages/warps。

    autotune 规则(同 grouped_gemm_fused):
      - 搜索符号 launch 时不重传; grid 用 lambda 从 META 取;
      - configs 固定 BLOCK_M=64(tile_expert 按它预算)，只搜 BLOCK_N/K/stages/warps。

输入: x_sorted(N,H); W(E,H,K); m_offsets(E+1,) long(段对齐后边界)
输出: y(N,K)
"""
import torch
import triton
import triton.language as tl

_FIXED_BLOCK_M = 64


def _build_tile_expert(m_offsets, BLOCK_M, num_tiles_m):
    """host 端预计算: 第 t 个行-tile 属于哪个专家。返回 (num_tiles_m,) int32。"""
    device = m_offsets.device
    tile_row_start = torch.arange(num_tiles_m, device=device) * BLOCK_M
    tile_expert = torch.searchsorted(m_offsets[1:], tile_row_start, right=True)
    return tile_expert.to(torch.int32)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['N', 'H', 'K'],
)
@triton.jit
def _grouped_gemm_kernel(
    x_ptr, w_ptr, y_ptr,
    tile_expert_ptr, m_offsets_ptr,
    N, H, K,
    stride_xn, stride_xh,
    stride_we, stride_wh, stride_wk,
    stride_yn, stride_yk,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_base = w_ptr + e * stride_we
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < H
        x_blk = tl.load(
            x_ptr + offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xh,
            mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        w_blk = tl.load(
            w_base + offs_k[:, None] * stride_wh + offs_n[None, :] * stride_wk,
            mask=k_mask[:, None] & (offs_n[None, :] < K), other=0.0)
        acc += tl.dot(x_blk, w_blk)
    y_mask = row_mask[:, None] & (offs_n[None, :] < K)
    tl.store(y_ptr + offs_m[:, None] * stride_yn + offs_n[None, :] * stride_yk,
             acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def grouped_gemm(x_sorted, W, m_offsets, *_ignored, **_kw):
    """host 包装(autotune 版)。BLOCK_* 由 autotune 决定；grid 用 lambda。
    *_ignored/**_kw 吸收旧调用的 BLOCK_M/N/K(被忽略)。
    """
    N, H = x_sorted.shape
    E, H2, K = W.shape
    assert H == H2, f"hidden 不匹配: x {H} vs W {H2}"
    num_tiles_m = triton.cdiv(N, _FIXED_BLOCK_M)
    tile_expert = _build_tile_expert(m_offsets, _FIXED_BLOCK_M, num_tiles_m)

    y = torch.empty((N, K), dtype=x_sorted.dtype, device=x_sorted.device)
    grid = lambda META: (num_tiles_m, triton.cdiv(K, META['BLOCK_N']))
    _grouped_gemm_kernel[grid](
        x_sorted, W, y, tile_expert, m_offsets,
        N, H, K,
        x_sorted.stride(0), x_sorted.stride(1),
        W.stride(0), W.stride(1), W.stride(2),
        y.stride(0), y.stride(1),
    )
    return y