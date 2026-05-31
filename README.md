# moe-ep-lab — 一个 MoE 推理系统的全栈剖析

> 从 DeepSeek-V2-Lite 的真实路由出发，自底向上拆解 MoE：
> 路由 → 单卡专家计算 → 专家并行 all-to-all → 通信-计算重叠 → 对标 DeepEP。

## 这个项目在解决什么
MoE 把"通信"变成了瓶颈：专家分散在多卡上，每一层都要做一次变长 all-to-all
（dispatch 把 token 寄给专家，combine 把结果寄回并加权求和）。本项目分四阶段，
亲手实现这条链路，并搞清楚 DeepEP 到底替 MoE 解决了什么通信问题。

## 硬件分工
- 本地 RTX 5090 (32GB, sm_120)：Phase 0 / Phase 1（单卡，跑模型 + 单卡 kernel）
- 云端 2×H100（RunPod 等）：Phase 2（真·多卡 all-to-all + 真 DeepEP 对标）
  - 注：DeepEP 依赖 NVSHMEM/RDMA，只在 Hopper 上跑得顺；5090 跑不了真 DeepEP。

## 阶段与目录映射
| 阶段 | 目录 | 产出 |
|------|------|------|
| Phase 0 MoE 扫盲 | src/phase0_moe_literacy/ | 真实路由 trace + 负载分布分析 |
| Phase 1 单卡专家路径 | src/phase1_single_gpu_moe/ | 朴素 vs 向量化 grouped-GEMM，Nsight 测速 |
| Phase 2 专家并行 | src/phase2_expert_parallel/ | dispatch/combine + 重叠 + DeepEP 对标 |
| Phase 3 总结 | docs/01_why_deepep.md | "为什么需要 DeepEP" 技术博客 |

## 快速开始
```bash
pip install -r requirements.txt
# Phase 0 第一步：跑通模型 + 抓一次真实路由
python -m src.phase0_moe_literacy.capture_routing
```