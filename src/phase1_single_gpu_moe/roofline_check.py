"""
[Phase 1] roofline_check.py — 快速判断 triton grouped GEMM 是 compute-bound 还是 memory-bound

为什么:
    triton 大输入下单位成本上升，怀疑卡在显存带宽(反复读 x_sorted/中间结果)而非算力。
    与其直接上 nsys，先做一个 30 秒的 roofline 粗算: 用实测 GEMM 时间反推达到的
    TFLOP/s 和有效带宽，对比 5090 的峰值(~理论 BF16 算力 / ~1.8TB/s 带宽)，
    若算力利用率很低、带宽接近打满 -> memory-bound -> 优化方向是减少显存往返(融合 gate+up)。

运行: uv run python -m src.phase1_single_gpu_moe.roofline_check
注: 峰值按 5090 估算填在下方常量，可按官方规格修正。
"""
import torch
import torch.nn.functional as F
from .common_moe import ExpertWeights, build_inputs_random
from .grouped_gemm_triton import grouped_gemm
from . import moe_layer_triton as mt

# 5090 (Blackwell) 粗略峰值，按需修正：BF16 tensor core 约 ~200+ TFLOP/s(dense)，显存带宽 ~1.79 TB/s
PEAK_BF16_TFLOPS = 210.0
PEAK_BW_TBS = 1.79


def _time(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]


def main():
    device = "cuda"
    E, H, I, topk = 64, 2048, 1408, 6
    weights = ExpertWeights.random(E, H, I, device=device)
    for tk in [4096, 16384]:
        x, idx, w = build_inputs_random(tk, H, E, topk, device=device, skew=1.5)
        x_sorted, m_offsets, *_ = mt._sort_and_align(x, idx, w, E)
        M = x_sorted.shape[0]                      # 对齐后总行数(含少量 padding)
        # 单次 gate GEMM: (M,H) @ (H,I) -> (M,I)
        t = _time(lambda: grouped_gemm(x_sorted, weights.W_gate, m_offsets,
                                       mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K))
        # 有效计算量(用有效行 M_valid 而非对齐 M，反映"有用"算力)
        M_valid = tk * topk
        flops = 2 * M_valid * H * I                # 一次 GEMM 的有效 FLOPs
        tflops = flops / (t * 1e-3) / 1e12
        # 显存读写量(粗略): 读 x_sorted(M*H) + 读 W(E*H*I 最坏) + 写 y(M*I)，bf16=2B
        bytes_rw = (M * H + M * I) * 2 + E * H * I * 2
        bw = bytes_rw / (t * 1e-3) / 1e12          # TB/s
        print(f"tokens={tk:>6}  单次gate {t:6.3f}ms | "
              f"算力 {tflops:6.1f} TFLOP/s ({tflops/PEAK_BF16_TFLOPS*100:4.1f}% peak) | "
              f"带宽 {bw:5.2f} TB/s ({bw/PEAK_BW_TBS*100:4.1f}% peak)")
    print("\n判读: 若算力%很低而带宽%偏高 -> memory-bound -> 该融合 gate+up 减少读 x_sorted。")
    print("      若两者都低 -> 仍是调度/占用问题 -> 调 BLOCK 或看 occupancy。")


if __name__ == "__main__":
    main()