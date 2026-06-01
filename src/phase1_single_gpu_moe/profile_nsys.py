"""
[Phase 1] profile_nsys.py — 用 NVTX 区间标注 naive / torch_vec，配合 nsys 看时间线

做什么:
    分别跑 naive 与 torch_vec 各一次(已预热)，各套一个 NVTX 区间。
    在 Nsight Systems 时间线上即可肉眼数到:两版都有一长串密集的小 GEMM kernel
    （每个专家一次），从而亲眼确认"排好队后仍逐专家 launch 64 次"。

运行:
    nsys profile -o phase1_timeline --trace=cuda,nvtx --force-overwrite true \
      uv run python -m src.phase1_single_gpu_moe.profile_nsys
    # 然后用 Nsight Systems GUI 打开 phase1_timeline.nsys-rep，
    # 或: nsys stats phase1_timeline.nsys-rep   # 命令行看 kernel 计数
输出: phase1_timeline.nsys-rep（时间线）；控制台打印两版 kernel 数提示
"""
import torch
from .common_moe import ExpertWeights, build_inputs_random
from .moe_layer_naive import moe_forward_naive
from .moe_layer_optimized import moe_forward_optimized


def main():
    device = "cuda"
    E, H, I, topk = 64, 2048, 1408, 6
    x, idx, w = build_inputs_random(4096, H, E, topk, device=device, skew=1.5)
    weights = ExpertWeights.random(E, H, I, device=device)

    nvtx = torch.cuda.nvtx
    # 预热(触发 cudnn/cublas 算法选择，避免污染时间线)
    for _ in range(5):
        moe_forward_naive(x, idx, w, weights)
        moe_forward_optimized(x, idx, w, weights)
    torch.cuda.synchronize()

    nvtx.range_push("NAIVE")          # 关键行: 时间线上这一段里数小 GEMM 个数
    moe_forward_naive(x, idx, w, weights)
    torch.cuda.synchronize()
    nvtx.range_pop()

    nvtx.range_push("TORCH_VEC")      # 关键行: 这一段里同样能数到 ~64 个小 GEMM
    moe_forward_optimized(x, idx, w, weights)
    torch.cuda.synchronize()
    nvtx.range_pop()

    print("[profile] 完成。用 Nsight 打开 .nsys-rep，在 NAIVE / TORCH_VEC 两个区间内")
    print("[profile] 数一下 GEMM kernel 数量 —— 两版都应是几十个(≈活跃专家数)，这就是 64 次 launch 的铁证。")
    print("[profile] 命令行快速看: nsys stats phase1_timeline.nsys-rep | grep -i gemm")


if __name__ == "__main__":
    main()