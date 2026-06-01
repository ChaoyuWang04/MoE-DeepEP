"""
[Phase 1] common_moe.py — 三版专家计算共享的契约、数据构造与专家权重

做什么:
    1. 定义统一函数签名 MoEForward，三版(naive / torch向量化 / triton)都实现它，
       这样 benchmark 脚本能用同一套代码喂同样输入、对拍同样输出。
    2. 提供专家权重容器 ExpertWeights：把 64 个专家的 W1/W2 堆成大张量，
       供向量化/Triton 版做 grouped GEMM（朴素版也能用，按 e 索引切片）。
    3. 提供输入构造：优先用 Phase 0 真实 trace（保留 ~10x 负载倾斜），
       否则退化为随机 topk（仅用于无 trace 时跑通流程）。

为什么需要它:
    三版要"同输入同输出"才能公平对比；倾斜的真实路由是性能差异的来源，必须统一。

输入/输出: 见各函数 docstring
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import torch

# 统一签名: 一个 MoE 专家计算版本 = 接收 (x, topk_idx, topk_weight, weights) 返回 out(T,H)
#   x          : (T, H)        本层输入 token（已是 2D，T = tokens）
#   topk_idx   : (T, k)        每个 token 选中的专家 id
#   topk_weight: (T, k)        对应权重
#   weights    : ExpertWeights 所有专家的 W1/W2
#   返回 out   : (T, H)        加权合成结果
MoEForward = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, "ExpertWeights"], torch.Tensor]


@dataclass
class ExpertWeights:
    """所有专家的权重，堆成 (E, ...) 大张量，便于按专家切片 / grouped GEMM。

    用一个最小但真实的 SwiGLU FFN: y = (silu(x@W_gate) * (x@W_up)) @ W_down
      W_gate, W_up : (E, H, I)   每个专家 hidden->intermediate
      W_down       : (E, I, H)   每个专家 intermediate->hidden
    """
    W_gate: torch.Tensor  # (E, H, I)
    W_up: torch.Tensor    # (E, H, I)
    W_down: torch.Tensor  # (E, I, H)

    @property
    def num_experts(self) -> int:
        return self.W_gate.shape[0]

    @property
    def hidden(self) -> int:
        return self.W_gate.shape[1]

    @property
    def intermediate(self) -> int:
        return self.W_gate.shape[2]

    @staticmethod
    def random(num_experts: int, hidden: int, intermediate: int,
               dtype=torch.bfloat16, device="cuda", seed: int = 0) -> "ExpertWeights":
        """造一组随机但数值稳定的专家权重(缩放过，避免 bf16 溢出)。"""
        g = torch.Generator(device=device).manual_seed(seed)
        scale = (hidden ** -0.5)                       # 关键行: 按 fan-in 缩放，控制激活幅度
        mk = lambda *s: (torch.randn(*s, generator=g, device=device, dtype=torch.float32) * scale).to(dtype)
        return ExpertWeights(
            W_gate=mk(num_experts, hidden, intermediate),
            W_up=mk(num_experts, hidden, intermediate),
            W_down=mk(num_experts, intermediate, hidden),
        )


def single_expert_ffn(x_e: torch.Tensor, w: ExpertWeights, e: int) -> torch.Tensor:
    """单个专家 e 的 SwiGLU 前向，供朴素版逐专家调用，也作为对拍的语义基准。

    输入: x_e (n_e, H) 分给专家 e 的 token；e 专家 id
    输出: (n_e, H)
    """
    gate = torch.nn.functional.silu(x_e @ w.W_gate[e])   # (n_e, I)
    up = x_e @ w.W_up[e]                                  # (n_e, I)
    return (gate * up) @ w.W_down[e]                      # (n_e, H)


def build_inputs_from_trace(trace_layer, hidden: int, num_experts: int,
                            dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """用 Phase 0 真实 trace 的某一层构造输入(保留真实 10x 倾斜)。

    输入: trace_layer = (topk_idx[T,k], topk_weight[T,k]) 来自 datas/routing_traces
    输出: x(T,H), topk_idx(T,k) long, topk_weight(T,k) float —— 全部在 device 上
    注: x 是随机生成的(我们没存隐藏态)，但 topk_idx/weight 是真实的 —— 性能只取决于路由分布，
        所以这足以复现真实的负载倾斜对 kernel 的影响。
    """
    topk_idx, topk_weight = trace_layer
    T, k = topk_idx.shape
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(T, hidden, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x,
            topk_idx.to(device).long(),
            topk_weight.to(device).float())


def build_inputs_random(num_tokens, hidden, num_experts, topk,
                        dtype=torch.bfloat16, device="cuda", seed: int = 0,
                        skew: Optional[float] = None):
    """无 trace 时的退化输入。skew=None 均匀；skew>0 用幂律制造人工倾斜(粗略模拟真实)。"""
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(num_tokens, hidden, generator=g, device=device, dtype=torch.float32)).to(dtype)
    if skew is None:
        topk_idx = torch.randint(0, num_experts, (num_tokens, topk), generator=g, device=device)
    else:
        # 幂律采样: 少数专家高频，模拟真实倾斜
        ranks = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
        prob = (ranks ** (-skew))
        prob = prob / prob.sum()
        topk_idx = torch.multinomial(prob, num_tokens * topk, replacement=True,
                                     generator=g).view(num_tokens, topk)
    topk_weight = torch.full((num_tokens, topk), 1.0 / topk, device=device, dtype=torch.float32)
    return x, topk_idx.long(), topk_weight