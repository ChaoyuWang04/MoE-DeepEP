"""
[Phase 1] probe_triton.py — 拆解 triton MoE 的耗时构成，定位大输入下变慢的真因

做什么:
    把 moe_forward_triton 拆成两段分别计时:
      (1) _sort_and_align 预处理(sort/bincount/散射)  —— host+少量 kernel
      (2) 3 次 grouped_gemm kernel 本身
    并扫不同 token 数 + 不同 BLOCK_M，看到底是【预处理】还是【kernel】随 token 爆炸，
    以及 BLOCK 配置的影响。用数据回答"为何大输入下 triton 反慢"，而非猜测。

运行:
    uv run python -m src.phase1_single_gpu_moe.probe_triton
输出: 各 token 数下 预处理ms / 3xGEMM ms / 总ms，以及不同 BLOCK_M 的 GEMM ms
"""
import torch
import torch.nn.functional as F
from .common_moe import ExpertWeights, build_inputs_random
from .grouped_gemm_triton import grouped_gemm
from . import moe_layer_triton as mt


def _time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    device = "cuda"
    E, H, I, topk = 64, 2048, 1408, 6
    weights = ExpertWeights.random(E, H, I, device=device)

    print(f"{'tokens':>7} | {'preprocess':>10} | {'3x gemm':>9} | {'total':>8} | 主导")
    print("-" * 60)
    for tk in [445, 4096, 16384]:
        x, idx, w = build_inputs_random(tk, H, E, topk, device=device, skew=1.5)

        # (1) 预处理单独计时
        def pre():
            return mt._sort_and_align(x, idx, w, E)
        t_pre = _time(pre)

        # (2) 3 次 grouped_gemm 单独计时(用预处理结果)
        x_sorted, m_offsets, valid_pos, gather_token, sort_weight = mt._sort_and_align(x, idx, w, E)
        def gemm3():
            g = grouped_gemm(x_sorted, weights.W_gate, m_offsets, mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K)
            u = grouped_gemm(x_sorted, weights.W_up, m_offsets, mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K)
            h = (F.silu(g.float()) * u.float()).to(x.dtype)
            return grouped_gemm(h, weights.W_down, m_offsets, mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K)
        t_gemm = _time(gemm3)

        total = t_pre + t_gemm
        dom = "预处理" if t_pre > t_gemm else "GEMM kernel"
        print(f"{tk:>7} | {t_pre:>9.3f}m | {t_gemm:>8.3f}m | {total:>7.3f}m | {dom}")

    # --- 扫 BLOCK_M/N，看 tile 大小对大输入 GEMM 的影响 ---
    print("\n[probe] 大输入(16384) 下扫 BLOCK 配置对 3xGEMM 的影响:")
    tk = 16384
    x, idx, w = build_inputs_random(tk, H, E, topk, device=device, skew=1.5)
    x_sorted, m_offsets, *_ = mt._sort_and_align(x, idx, w, E)
    print(f"{'BLOCK_M':>8} {'BLOCK_N':>8} {'BLOCK_K':>8} | {'3x gemm ms':>10}")
    print("-" * 44)
    for bm, bn, bk in [(64, 64, 32), (128, 128, 32), (128, 256, 64), (64, 128, 64), (256, 128, 64)]:
        try:
            def gemm3():
                g = grouped_gemm(x_sorted, weights.W_gate, m_offsets, bm, bn, bk)
                u = grouped_gemm(x_sorted, weights.W_up, m_offsets, bm, bn, bk)
                h = (F.silu(g.float()) * u.float()).to(x.dtype)
                return grouped_gemm(h, weights.W_down, m_offsets, bm, bn, bk)
            t = _time(gemm3)
            print(f"{bm:>8} {bn:>8} {bk:>8} | {t:>9.3f}m")
        except Exception as ex:
            print(f"{bm:>8} {bn:>8} {bk:>8} | 失败: {type(ex).__name__}")


if __name__ == "__main__":
    main()