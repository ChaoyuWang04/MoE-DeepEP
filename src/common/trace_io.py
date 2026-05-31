"""
[common] trace_io.py — 路由 trace 的保存与读取

做什么:
    Phase 0 抓到的真实 topk_idx / topk_weight 存盘；Phase 2 读出来当作
    驱动 all-to-all 的真实负载分布（替代老师代码里的 torch.randint 假数据）。

输入/输出:
    save_trace: 输入 dict{layer_idx -> (topk_idx, topk_weight)} -> 输出 .pt 文件路径
    load_trace: 输入 .pt 名 -> 输出同结构 dict
"""
import torch
from pathlib import Path
from .config import TRACE_DIR


def save_trace(traces: dict, name: str = "deepseek_v2_lite.pt") -> Path:
    """把 {layer_idx: (topk_idx, topk_weight)} 存到 datas/routing_traces/。"""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / name
    torch.save(traces, path)                         # 关键行: 整 dict 落盘
    return path


def load_trace(name: str = "deepseek_v2_lite.pt") -> dict:
    """读回路由 trace（默认放 CPU）。"""
    return torch.load(TRACE_DIR / name, map_location="cpu")
