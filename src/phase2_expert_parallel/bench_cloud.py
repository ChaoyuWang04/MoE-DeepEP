"""
[Phase 2] bench_cloud.py — 云端总入口: 串行 vs 重叠 vs DeepEP 三方对标(真 NCCL)

做什么(在 2×H100 上一条命令跑完):
    1. 正确性: 串行(fused) / 重叠(overlap) 对拍参考，先确保对再测速。
    2. 延迟对比: 串行 vs 重叠(不同 chunk 数)，套 NVTX 便于 nsys 看时间线重叠。
    3. 通信分解: dispatch / combine 各自耗时(真 NCCL all2all)。
    4. DeepEP 对标: 若环境装了 deep_ep，跑真 DeepEP dispatch/combine 同输入对比；
       未装则跳过并提示(本地/非 Hopper 正常跳过)。

环境前提:
    - 真双卡: torchrun --nproc_per_node=2，每进程独占一张卡(NCCL，不要 SINGLE_GPU)。
    - DeepEP 对标需 Hopper + 已装 deep_ep；否则自动跳过该部分。

运行(云端 2×H100):
    torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.bench_cloud
    # 配合 nsys 看重叠时间线:
    nsys profile -o ep_overlap --trace=cuda,nvtx,nccl --force-overwrite true \
      torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.bench_cloud --profile
"""
import argparse
import os
import statistics
import torch
import torch.distributed as dist

from ..phase1_single_gpu_moe.common_moe import ExpertWeights
from .ep_dist_fused import _ref_local
from .ep_dist_opt import dispatch_2a2a, combine_2a2a
from .expert_compute_fused import expert_compute_fused
from .ep_dist_overlap import ep_overlap_forward


def cuda_time_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def serial_forward(x, idx, wgt, weights_local, E, world_size, rank, comm_on_cpu):
    """串行基线: 整批 dispatch -> expert -> combine(无重叠)。"""
    recv_x, rle, sp = dispatch_2a2a(x, idx, E, world_size, rank, comm_on_cpu)
    eo = expert_compute_fused(recv_x, rle, weights_local)
    return combine_2a2a(eo, wgt, sp, world_size, rank, x.shape[1], comm_on_cpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2048, help="每 rank token 数")
    ap.add_argument("--chunks", type=int, default=4, help="重叠版 chunk 数")
    ap.add_argument("--profile", action="store_true", help="套 NVTX 供 nsys")
    args = ap.parse_args()

    backend = os.environ.get("EP_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank(); world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    single_gpu = os.environ.get("SINGLE_GPU", "0") == "1"
    if torch.cuda.is_available():
        gpu_id = 0 if single_gpu else local_rank
        torch.cuda.set_device(gpu_id); device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    comm_on_cpu = (backend == "gloo")
    nvtx = torch.cuda.nvtx if torch.cuda.is_available() else None

    E, H, I, topk = 64, 2048, 1408, 6
    E_local = E // world_size
    full = ExpertWeights.random(E, H, I, device=device, seed=123)
    weights_local = ExpertWeights(
        full.W_gate[rank*E_local:(rank+1)*E_local].contiguous(),
        full.W_up[rank*E_local:(rank+1)*E_local].contiguous(),
        full.W_down[rank*E_local:(rank+1)*E_local].contiguous())

    T_local = args.tokens
    g = torch.Generator(device=device).manual_seed(rank + 7)
    x = (torch.randn(T_local, H, generator=g, device=device, dtype=torch.float32) * (H**-0.5)).to(torch.bfloat16)
    idx = torch.randint(0, E, (T_local, topk), generator=g, device=device)
    wgt = torch.rand(T_local, topk, generator=g, device=device)
    wgt = wgt / wgt.sum(dim=1, keepdim=True)

    if rank == 0:
        print(f"=== Phase2 云端对标 | world_size={world_size} backend={backend} "
              f"T_local={T_local} E={E} top-{topk} H={H} ===")

    # ---- 1. 正确性 ----
    ref = _ref_local(x, idx, wgt, full, topk)
    out_s = serial_forward(x, idx, wgt, weights_local, E, world_size, rank, comm_on_cpu)
    out_o = ep_overlap_forward(x, idx, wgt, weights_local, E, world_size, rank, args.chunks, comm_on_cpu)
    ds = (out_s.float() - ref.float()).abs().max().item()
    do = (out_o.float() - ref.float()).abs().max().item()
    print(f"[rank {rank}] 正确性: serial diff={ds:.2e} {'PASS' if ds<8e-2 else 'FAIL'} | "
          f"overlap diff={do:.2e} {'PASS' if do<8e-2 else 'FAIL'}")
    dist.barrier()

    # ---- 2. 延迟对比: 串行 vs 重叠 ----
    def run_serial():
        if args.profile and nvtx: nvtx.range_push("serial")
        serial_forward(x, idx, wgt, weights_local, E, world_size, rank, comm_on_cpu)
        if args.profile and nvtx: nvtx.range_pop()
    def run_overlap():
        if args.profile and nvtx: nvtx.range_push("overlap")
        ep_overlap_forward(x, idx, wgt, weights_local, E, world_size, rank, args.chunks, comm_on_cpu)
        if args.profile and nvtx: nvtx.range_pop()

    t_s = cuda_time_ms(run_serial)
    t_o = cuda_time_ms(run_overlap)
    if rank == 0:
        print(f"[rank 0] 串行 {t_s:.3f} ms | 重叠({args.chunks}chunk) {t_o:.3f} ms | "
              f"重叠提升 x{t_s/t_o:.2f}")

    # ---- 3. 通信分解: dispatch / combine 各自耗时 ----
    def only_dispatch():
        dispatch_2a2a(x, idx, E, world_size, rank, comm_on_cpu)
    t_disp = cuda_time_ms(only_dispatch, warmup=5, iters=30)
    if rank == 0:
        print(f"[rank 0] 单次 dispatch(2 all2all) {t_disp:.3f} ms")

    # 自研【纯通信】dispatch+combine(恒等专家)，与 DeepEP 同类项对比
    def ours_comm_only():
        recv_x, rle, sp = dispatch_2a2a(x, idx, E, world_size, rank, comm_on_cpu)
        # 恒等专家: 直接拿 recv_x 当输出送回 combine，只测通信
        combine_2a2a(recv_x, wgt, sp, world_size, rank, x.shape[1], comm_on_cpu)
    t_comm = cuda_time_ms(ours_comm_only, warmup=5, iters=30)
    if rank == 0:
        print(f"[rank 0] 自研纯通信 dispatch+combine(恒等专家) {t_comm:.3f} ms  <- 与 DeepEP 同类项")

    # ---- 4. DeepEP 对标(若可用) ----
    try:
        import deep_ep  # noqa
        if rank == 0:
            print("[rank 0] 检测到 deep_ep，运行真 DeepEP 对标...")
        from .deepep_baseline import run_deepep_compare
        run_deepep_compare(x, idx, wgt, E, world_size, rank, device)
    except ImportError:
        if rank == 0:
            print("[rank 0] 未装 deep_ep(非 Hopper 或未安装)，跳过 DeepEP 对标。")
    except Exception as ex:
        if rank == 0:
            print(f"[rank 0] DeepEP 对标出错(跳过): {type(ex).__name__}: {ex}")

    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()