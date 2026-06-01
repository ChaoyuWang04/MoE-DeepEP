"""
[tests] test_fused_moe.py — 融合版 & CUDA Graph 版对拍朴素版

判据: 与 naive 的最大绝对误差在 bf16 容差内。
运行: uv run python -m tests.test_fused_moe
"""
import torch
from src.phase1_single_gpu_moe.common_moe import ExpertWeights, build_inputs_random
from src.phase1_single_gpu_moe.moe_layer_naive import moe_forward_naive
from src.phase1_single_gpu_moe.moe_layer_triton_fused import (
    moe_forward_triton_fused, MoEGraphRunner)


def _check(name, out, ref, atol=5e-2):
    d = (out.float() - ref.float()).abs().max().item()
    ok = d < atol
    print(f"[test] {name:14s} max_abs_diff={d:.4e}  {'PASS' if ok else 'FAIL'}")
    assert ok, f"{name} 误差过大: {d}"


def main():
    assert torch.cuda.is_available()
    device = "cuda"
    E, H, I, topk = 64, 2048, 1408, 6
    x, idx, w = build_inputs_random(4096, H, E, topk, device=device, skew=1.5)
    weights = ExpertWeights.random(E, H, I, device=device)

    ref = moe_forward_naive(x, idx, w, weights)
    _check("fused", moe_forward_triton_fused(x, idx, w, weights), ref)

    runner = MoEGraphRunner(weights)
    _check("fused+graph", runner(x, idx, w), ref)
    # 再跑一次验证 replay 路径(非首次构图)也对
    _check("fused+graph(2)", runner(x, idx, w), ref)
    print("[test] ALL PASS")


if __name__ == "__main__":
    main()