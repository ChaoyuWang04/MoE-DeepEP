"""
[tests] test_grouped_gemm.py — 验证 Triton 单投影 grouped GEMM 的分组调度正确性

做什么:
    在扩到完整 SwiGLU 之前，先单独验证最难的部分: 每个 tile 是否正确地按 offsets
    找到所属专家、读对了权重。用纯 torch 的逐段 matmul 作基准对拍。
    特意构造【倾斜】的段长(热点段几千行、冷门段几行)，复现真实分布。

运行:
    uv run python -m pytest tests/test_grouped_gemm.py -s
    或直接: uv run python -m tests.test_grouped_gemm
"""
import torch
from src.phase1_single_gpu_moe.grouped_gemm_triton import grouped_gemm

BLOCK_M = 64


def _align_segments(counts, block_m):
    """把每段长向上对齐到 block_m，返回 (对齐后段长, 对齐后总行数)。

    为什么要对齐: kernel 假设一个 BLOCK_M 行的 tile 不跨专家边界。把每段补齐到 block_m
    的倍数即可保证。对齐 padding 每段 < block_m 行，远小于 torch_bmm 的对齐到最长段。
    """
    aligned = [((c + block_m - 1) // block_m) * block_m for c in counts]
    return aligned, sum(aligned)


def _build_sorted_input(counts, hidden, device, dtype=torch.bfloat16, seed=0):
    """按给定每专家段长，构造对齐后的 x_sorted 及 offsets。空 padding 行填 0。

    返回:
        x_sorted   : (N_aligned, H)  各段对齐到 BLOCK_M，段内有效行随机、padding 行=0
        m_offsets  : (E+1,) 对齐后的段边界
        seg_valid  : list[(start, valid_len)] 每段有效区间，供基准对拍用
    """
    E = len(counts)
    aligned, N = _align_segments(counts, BLOCK_M)
    g = torch.Generator(device=device).manual_seed(seed)
    x_sorted = torch.zeros(N, hidden, dtype=dtype, device=device)
    m_offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    seg_valid = []
    cur = 0
    for e in range(E):
        valid = counts[e]
        if valid > 0:
            x_sorted[cur:cur + valid] = (torch.randn(valid, hidden, generator=g,
                                          device=device, dtype=torch.float32) * (hidden ** -0.5)).to(dtype)
        seg_valid.append((cur, valid))
        cur += aligned[e]
        m_offsets[e + 1] = cur
    return x_sorted, m_offsets, seg_valid


def test_grouped_gemm_skewed():
    assert torch.cuda.is_available(), "需要 CUDA"
    device = "cuda"
    H, K, dtype = 2048, 1408, torch.bfloat16

    # 倾斜段长: 4 个热点(几千行) + 一堆冷门(几行/空)
    counts = [3000, 2500, 1800, 900] + [5, 3, 0, 1, 7, 2, 0, 0] + [4] * 8
    E = len(counts)

    x_sorted, m_offsets, seg_valid = _build_sorted_input(counts, H, device, dtype)
    g = torch.Generator(device=device).manual_seed(1)
    W = (torch.randn(E, H, K, generator=g, device=device, dtype=torch.float32) * (H ** -0.5)).to(dtype)

    # --- Triton ---
    y = grouped_gemm(x_sorted, W, m_offsets, BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_K=32)

    # --- 基准: 逐段 torch matmul，只对有效行 ---
    ref = torch.zeros_like(y)
    for e, (start, valid) in enumerate(seg_valid):
        if valid > 0:
            ref[start:start + valid] = (x_sorted[start:start + valid] @ W[e]).to(ref.dtype)

    # 只比较有效行(padding 行不关心)
    max_abs = 0.0
    for e, (start, valid) in enumerate(seg_valid):
        if valid > 0:
            d = (y[start:start + valid].float() - ref[start:start + valid].float()).abs().max().item()
            max_abs = max(max_abs, d)
    print(f"[test] grouped_gemm 倾斜段长对拍 max_abs_diff={max_abs:.4e}")
    assert max_abs < 5e-1, f"分组调度可能选错专家权重! max_abs={max_abs}"
    print("[test] PASS — 每个 tile 都正确按 offsets 找到了所属专家的权重")


if __name__ == "__main__":
    test_grouped_gemm_skewed()