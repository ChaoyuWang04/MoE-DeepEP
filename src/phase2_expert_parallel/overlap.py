"""
[Phase 2] overlap.py — 通信-计算重叠（DeepEP hook 技巧的玩具版）

做什么:
    朴素流程 dispatch(等网络)->专家计算(等GPU)->combine(等网络) 严格串行。
    本模块把 batch 切成多个 chunk，用多 CUDA stream 让"上一块的专家计算"与
    "下一块的 dispatch 通信"并行，把通信延迟藏到计算后面。
    成功标志: Nsight Systems 时间线上能看到通信 kernel 与 GEMM kernel 并行。

输入: x, topk_idx, topk_weight, experts, overlap_chunks
输出: out (T, H)，数值与不重叠版一致（重叠只改时序不改结果）

这是本项目最硬核、也是理解 DeepEP 的钥匙: DeepEP 用 RDMA hook 做到零 SM 占用的重叠，
我们用 CUDA stream 做一个能看见原理的简化版。
"""
import torch


def moe_ep_overlapped(x, topk_idx, topk_weight, experts, overlap_chunks=2):
    # TODO(Phase 2 重点): 分 chunk + 多 stream + event 同步；先把 naive 跑通再来
    raise NotImplementedError("理解串行瓶颈后我们一起写重叠")
