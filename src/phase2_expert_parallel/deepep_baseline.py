"""
[Phase 2] deepep_baseline.py — 真 DeepEP 对标(按实测 API 签名适配)

做什么:
    用真 deep_ep.Buffer 跑 dispatch+combine 纯通信，与自研 2-all2all 对比延迟。
    对标的是【通信部分】(DeepEP 负责的事)，专家计算用恒等以隔离通信开销。

    按本机实测签名适配(check_deepep_api 输出):
      low_latency_dispatch(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts,
                           use_fp8=True, ..., return_recv_hook=False)
        -> ((recv_x, recv_count), recv_count2, handle, event, hook)   # 注意嵌套 tuple
      low_latency_combine(x, topk_idx, topk_weights, handle, ...) -> (out, event, hook)
    关键: use_fp8 显式关掉(=False)，否则 FP8 量化会让对比不公平/数值不可比。

    公平性说明(写进 docs/01):
      - DeepEP low_latency 用 IBGDA/RDMA，主场是【多机跨节点】；单机 NVLink 上其优势受限。
      - 我们的实现是 PyTorch+NCCL all2all(NVLink)。单机对比 DeepEP 未必碾压，
        这恰好说明 DeepEP 为多机大规模 EP 而生。

前提: Hopper + 已装 deep_ep。被 bench_cloud 在 try/except 调用。
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
    """用真 DeepEP low_latency dispatch+combine 计时(恒等专家，纯通信)。

    x:(T,H) bf16, topk_idx:(T,k), topk_weight:(T,k) float
    """
    import deep_ep
    T, H = x.shape
    group = dist.group.WORLD
    topk_idx_i = topk_idx.to(torch.int64)
    topk_w = topk_weight.to(torch.float32)

    # num_max_dispatch_tokens_per_rank: 每 rank 最多分发多少 token(buffer 上界)。用 T。
    num_max = T
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_max, H, world_size, num_experts)
    buffer = deep_ep.Buffer(group, num_nvl_bytes=0, num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=True, num_qps_per_rank=num_experts // world_size)

    def deepep_roundtrip():
        # 关键: use_fp8=False 关闭量化，保证与自研 bf16 路径可比
        dispatch_out = buffer.low_latency_dispatch(
            x, topk_idx_i, num_max, num_experts,
            use_fp8=False, async_finish=False, return_recv_hook=False)
        # 返回是嵌套 tuple: ((recv_x, recv_count), ..., handle, event, hook)
        recv_pack = dispatch_out[0]                 # (recv_x, recv_count)
        handle = dispatch_out[2]                    # combine 需要的 handle
        recv_x = recv_pack[0] if isinstance(recv_pack, (tuple, list)) else recv_pack
        # 恒等专家: 直接把 recv_x 当作专家输出送回 combine(只测通信)
        combine_out = buffer.low_latency_combine(
            recv_x, topk_idx_i, topk_w, handle,
            async_finish=False, return_recv_hook=False)
        return combine_out[0]

    # 预热(触发 DeepEP 内部 JIT / buffer 初始化)
    try:
        t = _time(deepep_roundtrip, warmup=10, iters=30)
    finally:
        pass

    if rank == 0:
        print(f"[rank 0] === 真 DeepEP low_latency dispatch+combine(恒等专家,bf16): {t:.3f} ms ===")
    return t