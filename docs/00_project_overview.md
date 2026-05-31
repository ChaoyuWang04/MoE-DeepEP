# 00 — 项目总览

## 一句话
从 DeepSeek-V2-Lite 的真实路由出发，自底向上拆解 MoE，最终搞清 DeepEP 替 MoE
解决了什么通信问题。

## 核心链路（也是 shape 主线）
x (T,H)
  --按 topk 复制 k 份、按专家重排-->  send (T*k, H)
  --all-to-all 寄给专家所在 rank-->   recv (E_local, max, H)
  --专家计算-->                        (E_local, max, H)
  --all-to-all 寄回-->                 (total, H)
  --按权重加权求和-->                  out (T, H)   # 形状闭环回到起点

## 四阶段
- Phase 0：跑真实 MoE，抓路由，量化负载倾斜（本地 5090）
- Phase 1：单卡专家计算路径，朴素 -> 向量化 grouped GEMM，Nsight 测速（本地 5090）
- Phase 2：专家并行 dispatch/combine + 通信-计算重叠 + DeepEP 对标（云端 2×H100）
- Phase 3：总结博客《为什么需要 DeepEP》（docs/01）

## 关键认知
- MoE 把"通信"变成瓶颈；DeepEP 是为这个变长小消息 all-to-all 写的库。
- 真 DeepEP 需要 Hopper + NVSHMEM/RDMA，5090(sm_120) 跑不了 -> 对标放云端。
