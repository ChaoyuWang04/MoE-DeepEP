"""
[Phase 1] moe_layer_triton_fused.py — 融合版 Triton MoE + 可选 CUDA Graph

做什么:
    在 moe_layer_triton 基础上做两层优化:
      1. kernel 融合: gate+up+silu 合成一个 kernel(grouped_gemm_fused)，
         3 次 GEMM -> 2 次(融合gate+up, 再 down)，并消掉独立 silu kernel。
      2. CUDA Graph: 把"排序后固定的一长串 kernel"录制成一张图，replay 时由 GPU 自己
         按图依次发射，彻底消除 kernel 之间的 CPU 调度空泡(host launch 间隙)。

    两个函数:
      moe_forward_triton_fused : 普通调用(融合版，无 graph)，用于对拍与首次构图
      MoEGraphRunner           : CUDA Graph 封装，固定 shape 下 replay，消空泡

签名/输入/输出: 同 common_moe.MoEForward
"""
import torch
import torch.nn.functional as F
from .common_moe import ExpertWeights
from .grouped_gemm_triton import grouped_gemm
from .grouped_gemm_fused import fused_gate_up_silu
from . import moe_layer_triton as mt   # 复用 _sort_and_align / BLOCK / ALIGN

BLOCK_M, BLOCK_N, BLOCK_K = mt.BLOCK_M, mt.BLOCK_N, mt.BLOCK_K


def moe_forward_triton_fused(x, topk_idx, topk_weight, weights: ExpertWeights):
    """融合版(无 CUDA Graph)。gate+up+silu 一个 kernel，down 一个 kernel。"""
    E = weights.num_experts
    x_sorted, m_offsets, valid_pos, gather_token, sort_weight = mt._sort_and_align(
        x, topk_idx, topk_weight, E)

    # 关键行: 2 次 GEMM 取代 3 次 + 1 silu —— gate/up/silu 全在一个 kernel
    h = fused_gate_up_silu(x_sorted, weights.W_gate, weights.W_up, m_offsets,
                           BLOCK_M, BLOCK_N, BLOCK_K)
    y_sorted = grouped_gemm(h, weights.W_down, m_offsets, BLOCK_M, BLOCK_N, BLOCK_K)

    y_valid = y_sorted[valid_pos] * sort_weight.unsqueeze(-1).to(y_sorted.dtype)
    out = torch.zeros_like(x)
    out.index_add_(0, gather_token, y_valid)
    return out


# ============================================================================
# CUDA Graph 封装
# ----------------------------------------------------------------------------
# 原理(手把手):
#   每次 launch 一个 kernel，CPU 都要做一串准备(参数打包、驱动调用)，GPU 算完一个
#   要等 CPU 发下一个 —— 这段等待就是"空泡"。当 kernel 序列【固定不变】(同样的 op、
#   同样的 shape、同样的顺序)时，可以把这串发射"录制"成一张图(capture)，之后 replay
#   时 GPU 自己按图一个接一个发，几乎不再回到 CPU，空泡被消除。
#
# 三个硬性前提(否则结果错或崩):
#   1. shape 必须固定 —— graph 录的是固定地址上的固定 op。token 数变了要重录。
#   2. 输入必须写进【同一块固定显存】(static buffer) —— replay 只认录制时的那块地址；
#      新数据要 copy_ 进去，不能换新张量。
#   3. 不能有依赖 CPU 的动态控制流(if/循环次数随数据变) —— 我们的 MoE 排序后形状固定，满足。
#
# 注意: 我们这条链里有 _sort_and_align(含 torch.sort，输出长度依赖数据)。sort 这步
#   做不进 graph(长度可变)，所以我们【只把固定形状的计算段(两次 GEMM + 散射)录进 graph】，
#   sort 留在 graph 外。这是真实工程里的常见折中: 把可图的部分图掉。
# ============================================================================
class MoEGraphRunner:
    """对【固定 x_sorted 形状】录制 graph 并 replay。形状变则自动重录。"""

    def __init__(self, weights: ExpertWeights):
        self.w = weights
        self.graph = None
        self.static_x_sorted = None
        self.static_h = None
        self.static_y = None
        self.captured_N = None

    def _build(self, x_sorted, m_offsets):
        """在固定 buffer 上录制: fused_gate_up_silu -> down grouped_gemm。"""
        N, H = x_sorted.shape
        # 1. 分配固定输入 buffer，把当前数据拷进去
        self.static_x_sorted = x_sorted.clone()          # 关键行: graph 只认这块地址
        self.static_m_offsets = m_offsets.clone()

        # 2. 预热: 在录制前先正常跑一次，让 Triton JIT 完成、cublas 选好算法
        #    (capture 期间不能触发编译，否则崩)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                h = fused_gate_up_silu(self.static_x_sorted, self.w.W_gate, self.w.W_up,
                                       self.static_m_offsets, BLOCK_M, BLOCK_N, BLOCK_K)
                y = grouped_gemm(h, self.w.W_down, self.static_m_offsets, BLOCK_M, BLOCK_N, BLOCK_K)
        torch.cuda.current_stream().wait_stream(s)

        # 3. 正式录制
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):               # 关键行: 这个 with 块里的 kernel 被录制
            self.static_h = fused_gate_up_silu(self.static_x_sorted, self.w.W_gate, self.w.W_up,
                                               self.static_m_offsets, BLOCK_M, BLOCK_N, BLOCK_K)
            self.static_y = grouped_gemm(self.static_h, self.w.W_down, self.static_m_offsets,
                                         BLOCK_M, BLOCK_N, BLOCK_K)
        self.captured_N = N

    def __call__(self, x, topk_idx, topk_weight, weights=None):
        """完整 forward: sort(图外) -> replay 计算段(图内) -> 散射(图外)。
        weights 形参仅为兼容 bench 的统一签名(x,idx,wgt,weights)，实际用 self.w。
        """
        E = self.w.num_experts
        x_sorted, m_offsets, valid_pos, gather_token, sort_weight = mt._sort_and_align(
            x, topk_idx, topk_weight, E)

        # 形状变了(或首次) -> 重录 graph
        if self.graph is None or x_sorted.shape[0] != self.captured_N:
            self._build(x_sorted, m_offsets)

        # 关键行: 把新数据 copy 进固定 buffer(不能换地址)，再 replay
        self.static_x_sorted.copy_(x_sorted)
        self.static_m_offsets.copy_(m_offsets)
        self.graph.replay()                              # 关键行: GPU 自己按图发射，无 CPU 空泡

        y_valid = self.static_y[valid_pos] * sort_weight.unsqueeze(-1).to(self.static_y.dtype)
        out = torch.zeros_like(x)
        out.index_add_(0, gather_token, y_valid)
        return out