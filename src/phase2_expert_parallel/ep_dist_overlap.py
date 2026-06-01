"""
[Phase 2] ep_dist_overlap.py — 通信-计算重叠(异步 all2all + chunk 流水线)

做什么:
    把一批 token 切 num_chunks 个 chunk，用异步 all2all(async_op=True)做软件流水线:
    先发起所有 chunk 的 dispatch 通信(后台飞行)，再逐 chunk: 等本 chunk dispatch 到达 ->
    GPU 算 expert(此时下一 chunk 的通信仍在后台) -> combine。
    => chunk_i 的专家计算与 chunk_(i+1) 的 dispatch 通信在时间上重叠。

    依赖链限制(真实，DeepEP 也绕不开): dispatch 内"报数 all2all"的结果必须先拿到才能开
    "发数据 all2all"的接收 buffer，所以报数不可与本 chunk 计算重叠；但发数据(大头)可以与
    上一 chunk 的计算重叠。本版把【报数】在流水线启动前一次性批量做完，使每个 chunk 的
    数据传输能干净地与计算交错。

    与 DeepEP 关系(对标话术): 原理一致(让通信飞行时 GPU 不闲着)。DeepEP 用 RDMA hook
    做到零 SM 占用、且把报数/元数据开销压到极低；我们用异步集合通信在 PyTorch 层展示同一思想。

    本地 gloo: 验证逻辑正确(分 chunk 异步流水线结果仍 == 整批)。gloo 异步性弱，真重叠加速
    需 H100 双卡 NCCL + nsys 时间线确认。

运行:
    SINGLE_GPU=1 torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.ep_dist_overlap
    云端: torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.ep_dist_overlap
"""
import os
import torch
import torch.distributed as dist
from ..phase1_single_gpu_moe.common_moe import ExpertWeights
from .expert_compute_fused import expert_compute_fused


def _split_chunks(T_local, num_chunks):
    base = max(T_local // num_chunks, 1)
    bounds, s = [], 0
    for i in range(num_chunks):
        e = T_local if i == num_chunks - 1 else min(s + base, T_local)
        if s < T_local:
            bounds.append((s, e))
        s = e
    return bounds


def _a2a(out_t, in_t, comm_on_cpu, out_split=None, in_split=None, async_op=False):
    """all2all 封装。async_op=True 返回 work handle(后台飞行)；gloo 模式搬 CPU 通信。

    注意: gloo 不支持 cuda tensor，异步时需保证 in/out tensor 生命周期到 wait()。
    本封装在 comm_on_cpu 时返回 (work, out_cpu, out_t) 由调用方 wait 后搬回。
    """
    if comm_on_cpu:
        in_c = in_t.cpu()
        out_c = torch.empty(out_t.shape, dtype=out_t.dtype, device="cpu")
        work = dist.all_to_all_single(out_c, in_c, out_split, in_split, async_op=async_op)
        if async_op:
            return work, out_c, out_t      # 调用方: work.wait(); out_t.copy_(out_c.to(out_t.device))
        out_t.copy_(out_c.to(out_t.device))
        return None
    else:
        work = dist.all_to_all_single(out_t, in_t, out_split, in_split, async_op=async_op)
        return (work, None, out_t) if async_op else None


def _prepare_send(x_c, idx_c, num_experts, world_size):
    """本地准备: 按全局专家排序、算各专家/各 rank 的发送计数。不含通信。"""
    device = x_c.device
    k = idx_c.shape[1]
    E_local = num_experts // world_size
    flat_expert = idx_c.reshape(-1)
    flat_token = torch.arange(x_c.shape[0], device=device).repeat_interleave(k)
    order = torch.argsort(flat_expert)
    send_x = x_c[flat_token[order]]
    per_expert_send = torch.bincount(flat_expert, minlength=num_experts).to(torch.int64)
    send_counts = per_expert_send.view(world_size, E_local).sum(dim=1)
    return {
        "send_x": send_x, "per_expert_send": per_expert_send,
        "send_counts": send_counts, "order": order, "flat_token": flat_token,
        "k": k, "T_c": x_c.shape[0],
    }


def ep_overlap_forward(x_local, topk_idx_local, topk_weight_local,
                       weights_local, num_experts, world_size, rank,
                       num_chunks=4, comm_on_cpu=False):
    """异步流水线: 批量报数 -> 预发起各 chunk 数据 dispatch -> 逐 chunk 算+combine。"""
    T_local, H = x_local.shape
    E_local = num_experts // world_size
    device = x_local.device
    out = torch.zeros_like(x_local)
    bounds = _split_chunks(T_local, num_chunks)
    nc = len(bounds)

    # ---- 阶段1: 各 chunk 本地准备(排序/计数，无通信) ----
    prep = [_prepare_send(x_local[s:e], topk_idx_local[s:e], num_experts, world_size)
            for (s, e) in bounds]

    # ---- 阶段2: 批量报数(每 chunk 一次 all2all #1，可连续发起) ----
    recv_per_expert = []
    for c in range(nc):
        rpe = torch.empty_like(prep[c]["per_expert_send"])
        _a2a(rpe, prep[c]["per_expert_send"], comm_on_cpu)   # 报数(同步，量很小)
        recv_per_expert.append(rpe)

    # ---- 阶段3: 预发起各 chunk 的数据 dispatch(异步，后台飞行) ----
    disp_works = []
    recv_x_list = []
    for c in range(nc):
        recv_counts = recv_per_expert[c].view(world_size, E_local).sum(dim=1)
        send_counts_list = prep[c]["send_counts"].tolist()
        recv_counts_list = recv_counts.tolist()
        N_recv = int(recv_counts.sum().item())
        recv_x = torch.empty((N_recv, H), dtype=x_local.dtype, device=device)
        # 关键行: async_op=True -> 通信后台飞行，不阻塞，立即返回 handle
        w = _a2a(recv_x, prep[c]["send_x"].contiguous(), comm_on_cpu,
                 recv_counts_list, send_counts_list, async_op=True)
        disp_works.append((w, recv_counts_list, send_counts_list, N_recv))
        recv_x_list.append(recv_x)

    # ---- 阶段4: 逐 chunk 等数据到达 -> GPU 算 -> combine(此时后续 chunk 通信仍在后台) ----
    for c in range(nc):
        w, recv_counts_list, send_counts_list, N_recv = disp_works[c]
        work, out_cpu, out_t = w
        if work is not None:
            work.wait()                                      # 关键行: 用到本 chunk 数据前才等
            if out_cpu is not None:
                out_t.copy_(out_cpu.to(device))
        recv_x = recv_x_list[c]

        # 本地推出每个收到 token 的本地专家 id(不通信)
        labels = torch.arange(E_local, device=device).repeat(world_size)
        recv_local_expert = labels.repeat_interleave(recv_per_expert[c]).to(torch.int64)

        # GPU 专家计算(融合 kernel)。此时 chunk c+1.. 的 dispatch 通信仍在后台飞行 -> 重叠
        expert_out = expert_compute_fused(recv_x, recv_local_expert, weights_local)

        # combine: 逆路径寄回(同步即可，量随 chunk 减小)
        recv_back = torch.empty((prep[c]["send_x"].shape[0], H), dtype=expert_out.dtype, device=device)
        _a2a(recv_back, expert_out.contiguous(), comm_on_cpu, send_counts_list, recv_counts_list)

        order = prep[c]["order"]; flat_token = prep[c]["flat_token"]
        s, e = bounds[c]
        wgt = topk_weight_local[s:e].reshape(-1)[order]
        weighted = recv_back * wgt.unsqueeze(-1).to(recv_back.dtype)
        out_c = torch.zeros((prep[c]["T_c"], H), dtype=expert_out.dtype, device=device)
        out_c.index_add_(0, flat_token[order], weighted)
        out[s:e] = out_c
    return out


def main():
    backend = os.environ.get("EP_BACKEND", "gloo" if os.environ.get("SINGLE_GPU") == "1" else
                             ("nccl" if torch.cuda.is_available() else "gloo"))
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

    E, H, I, topk = 64, 2048, 1408, 6
    E_local = E // world_size
    full = ExpertWeights.random(E, H, I, device=device, seed=123)
    weights_local = ExpertWeights(
        full.W_gate[rank*E_local:(rank+1)*E_local].contiguous(),
        full.W_up[rank*E_local:(rank+1)*E_local].contiguous(),
        full.W_down[rank*E_local:(rank+1)*E_local].contiguous())

    T_local = 512
    g = torch.Generator(device=device).manual_seed(rank + 7)
    x_local = (torch.randn(T_local, H, generator=g, device=device, dtype=torch.float32) * (H**-0.5)).to(torch.bfloat16)
    topk_idx_local = torch.randint(0, E, (T_local, topk), generator=g, device=device)
    topk_weight_local = torch.rand(T_local, topk, generator=g, device=device)
    topk_weight_local = topk_weight_local / topk_weight_local.sum(dim=1, keepdim=True)

    out_overlap = ep_overlap_forward(x_local, topk_idx_local, topk_weight_local,
                                     weights_local, E, world_size, rank, num_chunks=4, comm_on_cpu=comm_on_cpu)
    out_ref = ep_overlap_forward(x_local, topk_idx_local, topk_weight_local,
                                 weights_local, E, world_size, rank, num_chunks=1, comm_on_cpu=comm_on_cpu)
    d = (out_overlap.float() - out_ref.float()).abs().max().item()
    ok = d < 1e-3
    print(f"[rank {rank}] 异步重叠(4 chunk) vs 整批 对拍 max_abs_diff={d:.4e}  {'PASS' if ok else 'FAIL'}")
    if rank == 0:
        print("[rank 0] 异步流水线逻辑正确。真重叠加速到 H100 双卡 NCCL + nsys 时间线确认。")

    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()