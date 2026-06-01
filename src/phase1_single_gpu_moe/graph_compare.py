"""
[Phase 1] graph_compare.py — CUDA Graph 对照实验: 碎 kernel vs 大 kernel

做什么:
    验证"CUDA Graph 只在 kernel 多而碎时有收益"这一认知。对两种 forward 各做
    "无 graph vs 有 graph"对比:
      A. torch_vec 风格(碎 kernel): 逐专家 for 循环 + 大量小 GEMM/elementwise/index_add
         —— 预期: graph 明显加速(碎 kernel 间空泡多，被 graph 消除)。
      B. fused 风格(大 kernel): 2 个饱满 GEMM kernel
         —— 预期: graph 几乎无收益甚至略亏(本就无空泡，还多了 replay 的 copy)。

    为公平: 两者都把"固定形状的计算段"录进 graph，排序/散射留图外。
    为了让 A 的计算段形状固定可图，这里用【固定每专家容量 cap】的写法(padding 到 cap)，
    使逐专家循环的 shape 不依赖数据 —— 仅用于本对照实验，不追求 A 本身的效率。

运行: uv run python -m src.phase1_single_gpu_moe.graph_compare
"""
import torch
import torch.nn.functional as F
from .common_moe import ExpertWeights, build_inputs_random


def _time(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2]


# ---------- A: 碎 kernel 计算段(固定 cap，逐专家小 GEMM) ----------
def make_scatter_compute(x_sorted_static, weights, offsets_static, E, cap):
    """返回一个闭包: 在固定 buffer 上做逐专家小 GEMM(碎 kernel)。shape 全固定，可图。"""
    def compute():
        outs = []
        for e in range(E):                       # 关键: E 次循环 -> E×(若干小 kernel)，碎且多
            seg = x_sorted_static[e * cap:(e + 1) * cap]   # 固定长度 cap
            g = F.silu(seg @ weights.W_gate[e])
            u = seg @ weights.W_up[e]
            outs.append((g * u) @ weights.W_down[e])
        return torch.stack(outs)
    return compute


def run_case(name, build_compute, device):
    """对一个计算段做 无graph vs 有graph 对比。"""
    compute = build_compute()
    t_plain = _time(compute)

    # 录制 graph
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            compute()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        compute()
    t_graph = _time(lambda: g.replay())

    speedup = t_plain / t_graph
    print(f"  [{name:16s}] 无graph {t_plain:7.3f}ms | 有graph {t_graph:7.3f}ms | "
          f"graph 提升 x{speedup:4.2f}")
    return t_plain, t_graph


def main():
    device = "cuda"
    E, H, I, topk = 64, 2048, 1408, 6
    weights = ExpertWeights.random(E, H, I, device=device)
    cap = 128                                    # 每专家固定容量(对照实验用，固定 shape 才可图)
    x_sorted = torch.randn(E * cap, H, dtype=torch.bfloat16, device=device) * (H ** -0.5)
    offsets = (torch.arange(E + 1, device=device) * cap).long()

    print("[graph_compare] === A: 碎 kernel 计算段(逐专家 E=64 次小 GEMM) ===")
    run_case("碎kernel(torch)", lambda: make_scatter_compute(x_sorted, weights, offsets, E, cap), device)

    # B: 大 kernel(融合) —— 用真实融合 kernel，固定 shape
    from .grouped_gemm_fused import fused_gate_up_silu
    from .grouped_gemm_triton import grouped_gemm
    from . import moe_layer_triton as mt
    aligned_offsets = offsets.clone()            # cap=128 已是 BLOCK_M(64) 倍数
    def build_fused():
        def compute():
            h = fused_gate_up_silu(x_sorted, weights.W_gate, weights.W_up, aligned_offsets,
                                   mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K)
            return grouped_gemm(h, weights.W_down, aligned_offsets, mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K)
        return compute

    print("[graph_compare] === B: 大 kernel 计算段(融合 2 个饱满 GEMM) ===")
    run_case("大kernel(fused)", build_fused, device)

    print("\n[graph_compare] 结论应为: A(碎)graph 提升明显 >1; B(大)graph 提升≈1 甚至略<1。")
    print("[graph_compare] 这就是'graph 只在 kernel 多而碎时有用'的铁证。")


if __name__ == "__main__":
    main()