"""
[tests] test_ep_reference.py — 验证专家并行的"分卡"不改变数学结果

做什么:
    对拍两个单进程参考实现:
      dense_reference (不分卡，直接逐 token 算 topk)
      ep_reference    (按 world_size 分卡，走 dispatch→expert→combine 模拟)
    两者应完全一致 —— 证明"专家分散到多卡 + all-to-all 搬运"在数学上等价于"单卡直接算"。
    这是 Phase 2 的地基: 通信只是搬运数据，不改变结果。

运行: uv run python -m tests.test_ep_reference
"""
import torch
from src.phase1_single_gpu_moe.common_moe import ExpertWeights, build_inputs_random
from src.phase2_expert_parallel.ep_reference import (
    ep_reference_forward, dense_reference_forward)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    E, H, I, topk = 8, 256, 512, 2          # 小规模，逐 token 循环也快
    T = 64
    x, idx, w = build_inputs_random(T, H, E, topk, device=device, skew=1.0)
    weights = ExpertWeights.random(E, H, I, device=device)

    ref = dense_reference_forward(x, idx, w, weights)
    for world_size in [1, 2, 4]:
        out = ep_reference_forward(x, idx, w, weights, world_size)
        d = (out.float() - ref.float()).abs().max().item()
        ok = d < 5e-2
        print(f"[test] world_size={world_size}: max_abs_diff={d:.4e}  {'PASS' if ok else 'FAIL'}")
        assert ok, f"分卡 world_size={world_size} 结果与不分卡不一致!"
    print("[test] ALL PASS — 分卡 + dispatch/combine 与单卡直接算数学等价")


if __name__ == "__main__":
    main()