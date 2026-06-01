"""
[Phase 1] profile_nsys_fused.py — 用 nsys 抓 fused 版时间线，确认"空泡已消失"

做什么:
    在大输入下跑一次 fused 版 MoE，用 NVTX 标出计算段。配合 nsys 看时间线:
    与之前 triton 版那张"小方块带缝"的图对比，fused 版应是【少数几个饱满大 kernel
    连成密实一片】，kernel 之间几乎无空泡 —— 这就解释了为什么再加 CUDA Graph 也没收益
    (没有空泡可消)。

运行:
    nsys profile -o phase1_fused --trace=cuda,nvtx --force-overwrite true \
      uv run python -m src.phase1_single_gpu_moe.profile_nsys_fused
    nsys stats --report cuda_gpu_kern_sum phase1_fused.nsys-rep | head -20
    nsys stats --report nvtx_sum phase1_fused.nsys-rep | head -15
"""
import torch
from .common_moe import ExpertWeights, build_inputs_random
from .moe_layer_triton_fused import moe_forward_triton_fused


def main():
    device = "cuda"
    E, H, I, topk = 64, 2048, 1408, 6
    x, idx, w = build_inputs_random(16384, H, E, topk, device=device, skew=1.5)
    weights = ExpertWeights.random(E, H, I, device=device)

    for _ in range(5):
        moe_forward_triton_fused(x, idx, w, weights)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("FUSED_forward")
    moe_forward_triton_fused(x, idx, w, weights)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print("[nsys] done. 看 kernel 排名与时间线:")
    print("  nsys stats --report cuda_gpu_kern_sum phase1_fused.nsys-rep | head -20")
    print("  对比之前 triton 版: fused 的 _grouped_gemm/_fused_kernel 应连成密实片，缝隙极少。")


if __name__ == "__main__":
    main()