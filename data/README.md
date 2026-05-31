# datas/ — 路由 trace、benchmark 结果

- routing_traces/  : Phase 0 抓取的真实 topk_idx / topk_weight（每层一份）
- load_stats/      : 负载分布统计与图
- bench/           : Phase 2 三方对标（naive / ours / DeepEP）的延迟、通信量

大文件 (.pt/.npy/.npz) 默认不进 git，见 .gitignore。
