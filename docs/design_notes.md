# 设计笔记 / 优化轴（持续更新）

## all-to-all 优化的 5 个轴（Phase 2 逐项落地）
1. 通信轮数：dispatch 3 次 all2all -> 2 次。
   recv_splits（每 rank 总数）可由 recv_expert_counts（每专家数）按 rank 分组求和
   本地算出，无需单独再发一次。
       recv_splits = recv_expert_counts.view(world_size, E_local).sum(dim=1)
2. 通信量：dispatch 用 FP8；同一目标 rank 的重复 token 去重，只过一次网络。
3. 重叠：把 batch 切 chunk，多 CUDA stream 让"算上一块"与"传下一块"并行。
4. 执行效率：干掉逐元素 .item()（每次 = 一次 GPU->CPU 同步），路由用
   argsort/bincount/cumsum 向量化，替掉 Python for + append + stack。
5. 显存：packed 紧凑 buffer 取代 (E_local, T*world_size, H) 的最坏情况大 padding。

## Phase 0 观察（待填）
- 负载分布是否倾斜？最热/最冷专家？这决定 buffer 策略与是否需负载均衡。

## 踩坑记录（随做随记）
- token_map 存的是 append 后长度(1-indexed)，取值要 index-1（off-by-one 高发区）。
- gate 返回结构随 modeling 版本不同，先 print 确认再写 hook。
