"""
[Phase 2] deepep_baseline.py — 真 DeepEP 对标(仅 Hopper + 已装 deep_ep)

做什么:
    用真 deep_ep.Buffer.low_latency_dispatch / low_latency_combine 跑同一份输入，
    与我们自研的 dispatch_2a2a/combine 对比延迟，作为"性能上界 + 标准答案"。

    对标话术(写进 docs/01_why_deepep.md):
      我们的实现 = PyTorch 层 NCCL all2all + Triton 融合 kernel + 异步流水线重叠。
      DeepEP = RDMA/NVSHMEM 单边通信 + IBGDA + 零 SM 占用 hook 重叠。
      原理一致(减通信轮数、藏延迟)，DeepEP 在网卡/kernel 层更彻底，故延迟更低。

前提: 仅在装好 deep_ep 的 Hopper 节点可用。被 bench_cloud 在 try/except 里调用，
      非 Hopper 自动跳过。
"""
import statistics
import torch
import torch.distributed as dist


def _time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def run_deepep_compare(x, topk_idx, topk_weight, num_experts, world_size, rank, device):
    """用真 DeepEP 跑 dispatch+combine 并计时。x:(T,H), topk_idx:(T,k)。"""
    import deep_ep
    T, H = x.shape
    group = dist.group.WORLD

    # DeepEP low-latency buffer
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(T, H, world_size, num_experts)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // world_size)

    topk_idx_i = topk_idx.to(torch.int64)

    def deepep_dispatch_combine():
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x, topk_idx_i, T, num_experts, use_fp8=False)
        # 这里专家计算用恒等(只测通信路径开销，与自研版的通信部分公平对比)
        combined, *_ = buffer.low_latency_combine(recv_x, topk_idx_i, topk_weight, handle)
        return combined

    t = _time(deepep_dispatch_combine)
    if rank == 0:
        print(f"[rank 0] 真 DeepEP dispatch+combine(恒等专家) {t:.3f} ms")
        print(f"[rank 0] 对标提示: 与自研 dispatch({{disp}})+combine 的通信部分对比，"
              f"DeepEP 应更低(RDMA/IBGDA/零SM重叠)。差距与原因写进 docs/01_why_deepep.md。")
    return t