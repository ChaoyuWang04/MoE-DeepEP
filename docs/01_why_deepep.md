# 01 — 为什么需要 DeepEP（Phase 3 总结博客 · 待写）

## 提纲
1. MoE = 把通信变成瓶颈：每层一次变长 all-to-all，消息小而多。
2. 通用库(NCCL)为大块 all-reduce 优化，小消息下延迟/同步开销占主导。
3. 我们手写的 all-to-all 能做到什么（5 轴优化 + stream 重叠）。
4. DeepEP 多做了什么我们做不到的：
   - IBGDA / NVSHMEM 单边 RDMA：GPU 直接发小消息，绕开 host。
   - hook-based 零 SM 占用的通信-计算重叠。
   - NVLink + RDMA 双路径，异构域带宽转发。
   - FP8 dispatch / BF16 combine。
5. 诚实的边界：单机看不到多机/多节点收益；这正是 DeepEP 的主战场。

## 结论（待我们用 Phase 2 数据填）
我们的实现 vs DeepEP 的延迟差距 = ___；差距主要来自 ___。
