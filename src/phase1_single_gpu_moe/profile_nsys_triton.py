"""
[Phase 1] profile_nsys_triton.py — 用 nsys 通查 triton MoE，找出最耗时的 kernel

正确的 profiler 分工(记取教训):
    nsys = 听诊器，通查整个系统、看哪个 kernel/阶段占大头(本脚本)。
    ncu  = 显微镜，只对 nsys 定位出的那一个可疑 kernel 深挖(后续单独做)。

做什么:
    在大输入(16384, 真实倾斜)下跑一次 triton MoE，用 NVTX 标出三次 grouped GEMM
    (GATE / UP / DOWN)各自的区间，让 nsys 时间线显示:
      - 三次 GEMM 各占多少时间、是否串行等待
      - 单个 grouped GEMM kernel 的 GPU 时长 vs 启动间隙
    从而判断瓶颈是 compute 还是 memory(反复读 x_sorted)，再决定优化方向。

运行:
    nsys profile -o phase1_triton --trace=cuda,nvtx --force-overwrite true \
      uv run python -m src.phase1_single_gpu_moe.profile_nsys_triton
    # 看各 kernel 总耗时排名:
    nsys stats --report cuda_gpu_kern_sum phase1_triton.nsys-rep | head -25
    # 看 NVTX 区间耗时:
    nsys stats --report nvtx_sum phase1_triton.nsys-rep | head -25
"""
import torch
import torch.nn.functional as F
from .common_moe import ExpertWeights, build_inputs_random
from .grouped_gemm_triton import grouped_gemm
from . import moe_layer_triton as mt


def main():
    device = "cuda"
    E, H, I, topk = 64, 2048, 1408, 6
    x, idx, w = build_inputs_random(16384, H, E, topk, device=device, skew=1.5)
    weights = ExpertWeights.random(E, H, I, device=device)

    # 预热(Triton JIT + 算法选择)
    for _ in range(5):
        mt.moe_forward_triton(x, idx, w, weights)
    torch.cuda.synchronize()

    x_sorted, m_offsets, valid_pos, gather_token, sort_weight = mt._sort_and_align(x, idx, w, E)
    nvtx = torch.cuda.nvtx

    nvtx.range_push("GATE_gemm")
    gate = grouped_gemm(x_sorted, weights.W_gate, m_offsets, mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K)
    torch.cuda.synchronize(); nvtx.range_pop()

    nvtx.range_push("UP_gemm")
    up = grouped_gemm(x_sorted, weights.W_up, m_offsets, mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K)
    torch.cuda.synchronize(); nvtx.range_pop()

    nvtx.range_push("SILU_mul")
    h = (F.silu(gate.float()) * up.float()).to(x.dtype)
    torch.cuda.synchronize(); nvtx.range_pop()

    nvtx.range_push("DOWN_gemm")
    y = grouped_gemm(h, weights.W_down, m_offsets, mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K)
    torch.cuda.synchronize(); nvtx.range_pop()

    print("[nsys] done. 看两份报告:")
    print("  nsys stats --report nvtx_sum phase1_triton.nsys-rep        # GATE/UP/DOWN 各占多少")
    print("  nsys stats --report cuda_gpu_kern_sum phase1_triton.nsys-rep # 最耗时 kernel 排名")


if __name__ == "__main__":
    main()