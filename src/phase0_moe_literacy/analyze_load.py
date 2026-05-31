"""
[Phase 0] analyze_load.py — 真实路由的负载分布分析与可视化

做什么:
    读 capture_routing.py 抓到的真实路由 trace，统计每个专家被选中的次数，
    量化"负载倾斜"程度，并画出 (1) 全层聚合负载条形图 (2) 层×专家负载热力图。
    核心结论将驱动 Phase 2: 真实路由是否均匀，直接决定 all-to-all buffer 该开多大、
    要不要做负载均衡。

    关键认知（也是本脚本想用数据替你验证的）:
        - 训练均衡 != 这次推理均衡: 均衡 loss 只保证训练分布上聚合均衡，
          小样本 / 特定领域下，单层单批仍可能很倾斜。
        - 样本太小则看到的"倾斜"是噪声: 先确保 token 总数足够大再下结论。

输入:
    datas/routing_traces/deepseek_v2_lite.pt
        结构: {layer_idx: (topk_idx[T,k], topk_weight[T,k])}
    （可选）models/DeepSeek-V2-Lite 的 config，用来读 n_routed_experts
输出:
    - 控制台: 每层负载不均衡指标 + 全局汇总 + 一句解读提示
    - datas/load_stats/aggregate_load.png   全层聚合每专家负载
    - datas/load_stats/load_heatmap.png      层(行) × 专家(列) 负载热力图

运行:
    python -m src.phase0_moe_literacy.analyze_load
"""
import matplotlib
matplotlib.use("Agg")                       # 关键行: headless 服务器无显示器，用 Agg 后端
import matplotlib.pyplot as plt
import torch

from src.common import config
from src.common.trace_io import load_trace

LOAD_STATS_DIR = config.DATA_DIR / "load_stats"


# ---------------------------------------------------------------------------
# 基础统计
# ---------------------------------------------------------------------------
def expert_load(topk_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """统计每个专家被选中的次数。

    输入: topk_idx (T, k) 每个 token 选中的专家 id
    输出: counts (n_experts,) 每个专家被选中的总次数
    """
    # 关键行: 展平成一维后一次 bincount 即得负载；minlength 保证冷门专家也占一格(=0)
    return torch.bincount(topk_idx.flatten().to(torch.long), minlength=n_experts)


def imbalance_ratio(counts: torch.Tensor) -> float:
    """不均衡比 = 最热专家负载 / 平均负载。1.0=完全均匀，越大越倾斜。"""
    mean = counts.float().mean()
    return (counts.max() / mean).item() if mean > 0 else 0.0


def cv(counts: torch.Tensor) -> float:
    """变异系数 = 标准差 / 均值。对'整体有多散'比 max/mean 更稳健。"""
    c = counts.float()
    return (c.std(unbiased=False) / c.mean()).item() if c.mean() > 0 else 0.0


def cold_expert_frac(counts: torch.Tensor, thresh_ratio: float = 0.5) -> float:
    """冷门专家占比 = 负载低于(均值 × thresh_ratio)的专家数 / 总专家数。"""
    mean = counts.float().mean()
    return (counts.float() < mean * thresh_ratio).float().mean().item()


# ---------------------------------------------------------------------------
# 推断专家数（优先读 config，失败则用 trace 里的最大 id 兜底）
# ---------------------------------------------------------------------------
def infer_n_experts(traces: dict) -> int:
    """优先从模型 config 读 n_routed_experts；读不到就用 trace 最大专家 id + 1。"""
    try:
        from transformers import AutoConfig
        src = config.MODEL_DIR if config.MODEL_DIR.exists() else config.MODEL_NAME
        cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
        n = getattr(cfg, "n_routed_experts", None)
        if n:
            return int(n)
    except Exception as e:
        print(f"[analyze] 读 config 失败({e})，改用 trace 最大 id 兜底")
    max_id = max(idx.max().item() for idx, _ in traces.values())
    return int(max_id) + 1


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    traces = load_trace()
    n_experts = infer_n_experts(traces)
    layers = sorted(traces.keys())

    # --- 样本量体检: token 太少则结论是噪声，先警告 ---
    total_selections = sum(idx.numel() for idx, _ in traces.values())
    total_tokens = sum(idx.shape[0] for idx, _ in traces.values())
    avg_per_expert = total_selections / max(len(layers), 1) / n_experts
    print(f"[analyze] MoE 层数={len(layers)}  专家数={n_experts}")
    print(f"[analyze] 累计 token(含各层)={total_tokens}  平均每层每专家被选 ~{avg_per_expert:.1f} 次")
    if avg_per_expert < 20:
        print("WARN  样本偏小：平均每专家被选不足 ~20 次，下面的'倾斜'很可能是噪声。"
              "\n      建议回 capture_routing.py 把 prompt 换成几大段真实长文本(中英+代码)再跑。")

    # --- 逐层指标 + 构建 层×专家 负载矩阵 ---
    load_matrix = torch.zeros((len(layers), n_experts))   # 行=层, 列=专家, 值=被选次数
    print(f"\n{'layer':>6} | {'max/mean':>9} | {'CV':>6} | {'cold%':>8}")
    print("-" * 40)
    for r, lyr in enumerate(layers):
        topk_idx, _ = traces[lyr]
        counts = expert_load(topk_idx, n_experts)
        load_matrix[r] = counts.float()
        print(f"{lyr:>6} | {imbalance_ratio(counts):>9.2f} | {cv(counts):>6.2f} | "
              f"{cold_expert_frac(counts) * 100:>7.1f}%")

    # --- 全局汇总（沿层求和）---
    agg = load_matrix.sum(dim=0)
    print("-" * 40)
    print(f"[全局聚合] max/mean={imbalance_ratio(agg):.2f}  CV={cv(agg):.2f}  "
          f"冷门占比={cold_expert_frac(agg) * 100:.1f}%")
    print(f"[解读提示] max/mean 越接近 1 越均衡；>1.5 即明显倾斜。"
          f" CV 越大整体越不均。冷门专家越多，combine 时 padding 浪费越严重。")

    # --- 可视化 ---
    LOAD_STATS_DIR.mkdir(parents=True, exist_ok=True)

    # 图1: 全层聚合，每个专家被选总次数
    plt.figure(figsize=(12, 4))
    plt.bar(range(n_experts), agg.numpy())
    plt.axhline(agg.mean().item(), color="r", ls="--", label="mean")  # 关键行: 均线作均衡参照
    plt.xlabel("expert id"); plt.ylabel("selected count (all layers)")
    plt.title("Aggregate expert load - DeepSeek-V2-Lite"); plt.legend()
    p1 = LOAD_STATS_DIR / "aggregate_load.png"
    plt.tight_layout(); plt.savefig(p1, dpi=120); plt.close()

    # 图2: 层×专家 热力图（一眼看出哪层哪专家是热点）
    plt.figure(figsize=(12, max(4, len(layers) * 0.25)))
    plt.imshow(load_matrix.numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="selected count")
    plt.xlabel("expert id"); plt.ylabel("MoE layer (row order)")
    plt.title("Per-layer expert load heatmap")
    p2 = LOAD_STATS_DIR / "load_heatmap.png"
    plt.tight_layout(); plt.savefig(p2, dpi=120); plt.close()

    print(f"\n[analyze] 图已保存:\n  {p1}\n  {p2}")
    print("[analyze] 把全局指标和你看到的现象写进 docs/design_notes.md 的 'Phase 0 观察'。")


if __name__ == "__main__":
    main()