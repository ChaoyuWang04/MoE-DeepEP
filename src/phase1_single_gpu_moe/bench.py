"""
[Phase 1] bench.py — 三版专家计算的统一 benchmark / 对拍 / profile 脚手架

做什么:
    把 naive / torch向量化 / triton 三版插进同一框架，做到:
      1. 同输入: 优先用 Phase 0 真实 trace(保留 ~10x 倾斜)，否则随机/幂律退化。
      2. 对拍: 以 naive 为基准，校验另两版数值一致(bf16 容差)。
      3. 计时: CUDA event + 预热 + 多次取中位数(避开异步计时坑)。
      4. profile: 每版套 NVTX 区间，Nsight Systems 时间线上能分辨各版 kernel 形态。

为什么先搭它:
    先有量尺再量。三版每写好一版，这里注册一下即可立刻 profile/对拍/看 speedup。

运行:
    # 纯 benchmark + 对拍
    uv run python -m src.phase1_single_gpu_moe.bench
    # 配合 Nsight Systems 抓时间线(看 NVTX 区间区分三版):
    nsys profile -o phase1 --trace=cuda,nvtx \
      uv run python -m src.phase1_single_gpu_moe.bench --profile

输入: 命令行参数(见 argparse) + datas/routing_traces/deepseek_v2_lite.pt(可选)
输出: 控制台 speedup / 对拍结果表；--profile 时配合 nsys 生成时间线
"""
import argparse
import statistics
import torch

from src.common.trace_io import load_trace
from .common_moe import (
    ExpertWeights, build_inputs_from_trace, build_inputs_random,
)
from .moe_layer_naive import moe_forward_naive

# ---- 注册三版实现。未实现的填 None，bench 自动跳过。----
def _try_import_optimized():
    try:
        from .moe_layer_optimized import moe_forward_optimized
        return moe_forward_optimized
    except Exception:
        return None

def _try_import_bmm():
    try:
        from .moe_layer_bmm import moe_forward_bmm
        return moe_forward_bmm
    except Exception:
        return None


def _try_import_fused():
    try:
        from .moe_layer_triton_fused import moe_forward_triton_fused
        return moe_forward_triton_fused
    except Exception as e:
        print('[bench] fused import 失败:', e)
        return None


def _try_import_graph(weights):
    try:
        from .moe_layer_triton_fused import MoEGraphRunner
        return MoEGraphRunner(weights)
    except Exception as e:
        print('[bench] graph import 失败:', e)
        return None


def _try_import_triton():
    try:
        from .moe_layer_triton import moe_forward_triton
        return moe_forward_triton
    except Exception:
        return None


# ------------------------- 计时 -------------------------
def cuda_time_ms(fn, *args, warmup=10, iters=50):
    """CUDA event 计时: 先预热(触发编译/缓存)，再多次测，取中位数。

    关键: GPU 调用是异步的，必须用 event 而非 time.time()，否则测到的是 launch 时间。
    """
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()                              # 关键行: 预热后同步，清空队列
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        end.synchronize()                                 # 关键行: 等本次真正算完再读时间
        times.append(start.elapsed_time(end))             # ms
    return statistics.median(times)


# ------------------------- 对拍 -------------------------
def check_close(name, out, ref, atol=2e-2, rtol=2e-2):
    """与基准对拍。bf16 容差放宽；打印最大绝对误差。"""
    max_abs = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    print(f"  [{name:8s}] 对拍 {'PASS' if ok else 'FAIL'}  max_abs_diff={max_abs:.4e}")
    return ok


# ------------------------- 主流程 -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1, help="用 trace 的哪一层(默认首个 MoE 层)")
    ap.add_argument("--hidden", type=int, default=2048, help="DeepSeek-V2-Lite hidden=2048")
    ap.add_argument("--inter", type=int, default=1408, help="单专家 intermediate (moe_intermediate_size)")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=4096, help="无 trace 时随机输入的 token 数")
    ap.add_argument("--skew", type=float, default=1.5, help="无 trace 时幂律倾斜强度(模拟真实)")
    ap.add_argument("--profile", action="store_true", help="套 NVTX 区间，配合 nsys 使用")
    ap.add_argument("--synthetic", action="store_true",
                    help="强制用幂律随机输入(忽略 trace)，配合 --tokens/--skew 测大输入差距")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA"
    device = "cuda"

    # ---- 构造输入: --synthetic 强制随机；否则优先真实 trace ----
    try:
        if args.synthetic:
            raise RuntimeError("--synthetic: 跳过 trace，走幂律随机输入")
        traces = load_trace()
        layer = args.layer if args.layer in traces else next(iter(traces))
        x, topk_idx, topk_weight = build_inputs_from_trace(
            traces[layer], args.hidden, args.experts, device=device)
        # 真实 trace 的专家数以数据为准
        n_exp = int(topk_idx.max().item()) + 1
        n_exp = max(n_exp, args.experts)
        print(f"[bench] 用真实 trace 层 {layer}: tokens={x.shape[0]}, experts={n_exp}, "
              f"hidden={args.hidden}")
    except Exception as e:
        print(f"[bench] 无 trace({e})，改用幂律随机输入 skew={args.skew}")
        n_exp = args.experts
        x, topk_idx, topk_weight = build_inputs_random(
            args.tokens, args.hidden, n_exp, args.topk, device=device, skew=args.skew)

    weights = ExpertWeights.random(n_exp, args.hidden, args.inter, device=device)

    # ---- 收集要测的实现 ----
    impls = {"naive": moe_forward_naive}
    opt = _try_import_optimized()
    if opt is not None:
        impls["torch_vec"] = opt
    bmm = _try_import_bmm()
    if bmm is not None:
        impls["torch_bmm"] = bmm
    tri = _try_import_triton()
    if tri is not None:
        impls["triton"] = tri
    fused = _try_import_fused()
    if fused is not None:
        impls["fused"] = fused
    graph = _try_import_graph(weights)
    if graph is not None:
        impls["fused+graph"] = graph
    print(f"[bench] 参与对比的实现: {list(impls.keys())}")

    # ---- 基准输出(naive) ----
    ref = moe_forward_naive(x, topk_idx, topk_weight, weights)

    # ---- NVTX(profile 模式) ----
    nvtx = torch.cuda.nvtx

    print("\n[bench] === 正确性对拍 (基准=naive) ===")
    for name, fn in impls.items():
        if name == "naive":
            continue
        try:
            out = fn(x, topk_idx, topk_weight, weights)
            check_close(name, out, ref)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{name:8s}] OOM(跳过对拍) — 该方案显存爆掉，本身就是其缺陷")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [{name:8s}] 报错(跳过): {type(e).__name__}: {e}")

    print("\n[bench] === 延迟 (median over iters) ===")
    base_ms = None
    for name, fn in impls.items():
        try:
            if args.profile:
                nvtx.range_push(f"moe_{name}")           # 关键行: Nsight 时间线按版本分段
            ms = cuda_time_ms(fn, x, topk_idx, topk_weight, weights)
            if args.profile:
                nvtx.range_pop()
        except torch.cuda.OutOfMemoryError:
            print(f"  [{name:8s}]      OOM — 显存爆掉(padding 方案在大输入下的致命缺陷)")
            torch.cuda.empty_cache()
            continue
        if base_ms is None:
            base_ms = ms
        speedup = base_ms / ms
        print(f"  [{name:8s}] {ms:8.3f} ms   speedup x{speedup:5.2f}  (vs naive)")

    print("\n[bench] 提示: 用 nsys 看时间线 —— naive 应是几十段稀疏小 kernel，"
          "向量化/triton 应是 1~2 段饱满 kernel。把现象记进 design_notes。")


if __name__ == "__main__":
    main()