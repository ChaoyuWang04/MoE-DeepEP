"""
[Phase 1] profile_ncu.py — 用 Nsight Compute 剖析 triton vs naive 的 GPU 效率

做什么:
    speedup 比值会被 naive 的固定开销干扰，无法反映"triton 到底喂饱 GPU 没有"。
    本脚本各跑一次 naive 与 triton 的【单次】专家计算(已预热)，配合 ncu 抓:
      - sm__throughput / achieved occupancy : SM 是否忙
      - sm__pipe_tensor_op ... : tensor core 利用率(MoE GEMM 的关键)
      - dram throughput : 是否被显存带宽卡住
    对比两版的这些指标，才能定论 triton 的真实效率，而非被 launch 开销污染的比值。

运行(只抓关键 section，避免 ncu 过慢):
    ncu --set basic --target-processes all \
        --kernel-name-base demangled \
        -o phase1_ncu --force-overwrite \
        uv run python -m src.phase1_single_gpu_moe.profile_ncu
    # 然后: ncu -i phase1_ncu.ncu-rep  (或用 Nsight Compute GUI 打开对比)
    # 快速看: ncu --import phase1_ncu.ncu-rep --page details | grep -iE "tensor|occupancy|throughput"

提示: ncu 会让 kernel 跑很多遍取指标，整体会慢(几十秒~分钟)，正常。
"""
import torch
from .common_moe import ExpertWeights, build_inputs_random
from .moe_layer_naive import moe_forward_naive
from .moe_layer_triton import moe_forward_triton


def main():
    device = "cuda"
    E, H, I, topk = 64, 2048, 1408, 6
    # 用中等 token，倾斜真实；不要太大以免 ncu 太慢
    x, idx, w = build_inputs_random(4096, H, E, topk, device=device, skew=1.5)
    weights = ExpertWeights.random(E, H, I, device=device)

    # 预热(触发 Triton JIT + cublas 算法选择)
    for _ in range(5):
        moe_forward_naive(x, idx, w, weights)
        moe_forward_triton(x, idx, w, weights)
    torch.cuda.synchronize()

    # 各跑一次，ncu 会自动抓这两次涉及的所有 kernel 的指标
    torch.cuda.nvtx.range_push("NAIVE")
    moe_forward_naive(x, idx, w, weights)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("TRITON")
    moe_forward_triton(x, idx, w, weights)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print("[ncu] done. 对比 NAIVE 的 cutlass 小 GEMM 与 TRITON 的 grouped kernel:")
    print("[ncu]  - tensor core 利用率: triton 应显著更高(naive 的 32x32 块喂不饱)")
    print("[ncu]  - achieved occupancy: 看哪版 SM 更忙")


if __name__ == "__main__":
    main()