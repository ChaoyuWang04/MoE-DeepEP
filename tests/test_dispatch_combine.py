"""
[tests] test_dispatch_combine.py — dispatch/combine 正确性对拍

做什么:
    沿用老师思路: "专家计算"设为恒等、权重均匀，使整条 pipeline 数学上近似恒等，
    于是任何误差只可能来自 dispatch/combine 的路由/索引错误。
    对拍三方: naive vs optimized （vs DeepEP，仅 Hopper）。

判据: combine 输出与输入的相对误差 <= 1e-2
运行: torchrun --nproc_per_node=2 -m pytest tests/test_dispatch_combine.py
"""
# TODO(Phase 2 实现后填充)
