"""
[common] config.py — 全项目共享的配置与超参

做什么:
    集中存放模型名、MoE 结构超参、并行配置、路径常量，避免魔数散落各处。
    各 phase 脚本统一从这里 import，改一处即全局生效。

输入: 无（被其它模块 import）
输出: 路径常量 / dataclass 配置实例
"""
from dataclasses import dataclass
from pathlib import Path

# ---- 路径 ----
ROOT = Path(__file__).resolve().parents[2]          # 仓库根目录
DATA_DIR = ROOT / "data"
TRACE_DIR = DATA_DIR / "routing_traces"
MODEL_DIR = ROOT / "models" / "DeepSeek-V2-Lite"

# ---- 模型 ----
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite"         # 跑不起来时退路: "Qwen/Qwen3-30B-A3B"


@dataclass
class MoEConfig:
    """MoE 结构超参。实际值在 capture_routing 里从 model.config 读取后覆盖。"""
    n_routed_experts: int = 64       # 路由专家数（DeepSeek-V2-Lite）
    n_shared_experts: int = 2        # 共享专家数
    num_experts_per_tok: int = 6     # 每 token 选几个路由专家 (top-k)
    hidden: int = 2048               # 隐藏维（占位，以 model.config 为准）


@dataclass
class EPConfig:
    """Phase 2 专家并行配置。"""
    world_size: int = 2              # 几个 rank / 几张卡（Phase 2 云端）
    use_fp8_dispatch: bool = False   # 通信量优化轴: dispatch 是否用 FP8
    overlap_chunks: int = 1          # 通信-计算重叠: batch 切几块（1=不重叠）
