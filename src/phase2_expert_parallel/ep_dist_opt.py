"""
[Phase 2] ep_dist_opt.py — 专家并行(2 次 all2all 优化版)

做什么:
    在 ep_dist_naive 基础上把 dispatch 的 3 次 all2all 砍成 2 次，对拍朴素版/参考语义。

    朴素版 dispatch 的 3 次 all2all:
      #1 报数(每 rank 收多少行)  #2 发 token 数据  #3 逐 token 发 expert_id(N_recv 个 int)
    优化洞察(用户最早推导，落到本结构):
      接收方不需要【逐 token 的 expert_id】。因为 token 是按"目标(rank, 专家)"排序后发的，
      接收方只要知道【每个本地专家收了多少行】(E_local 个 int) 就能推出收到的每段属于哪个
      本地专家 —— 用 cumsum 还原段边界即可。
      于是把 #1 的"按 rank 报数"升级为"按(rank×本地专家)报数"(更细但仍是小向量)，
      #3 的逐 token expert_id(O(N_recv)) 被彻底省掉。dispatch: 3 次 -> 2 次。

    通信量对比(本版会打印):
      朴素 #3 发 N_recv 个 int64；优化版省掉它，报数向量仅 world_size*E_local 个 int。
      token 越多，省得越多 —— 这正是"能本地算的绝不上网问"。

运行:
    SINGLE_GPU=1 torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.ep_dist_opt
"""
import os
import torch
import torch.distributed as dist
from ..phase1_single_gpu_moe.common_moe import ExpertWeights, single_expert_ffn
from .ep_dist_naive import expert_compute  # 复用专家计算


def _a2a(out_t, in_t, comm_on_cpu, out_split=None, in_split=None):
    """all2all 封装: gloo 模式搬 CPU 通信再搬回。"""
    if comm_on_cpu:
        in_c = in_t.cpu()
        out_c = torch.empty(out_t.shape, dtype=out_t.dtype, device="cpu")
        dist.all_to_all_single(out_c, in_c, out_split, in_split)
        out_t.copy_(out_c.to(out_t.device))
    else:
        dist.all_to_all_single(out_t, in_t, out_split, in_split)


def dispatch_2a2a(x_local, topk_idx_local, num_experts, world_size, rank, comm_on_cpu=False):
    """2 次 all2all 的 dispatch。

    返回:
        recv_x            : (N_recv, H)
        recv_local_expert : (N_recv,)  每个收到 token 的本地专家 id(由 counts 本地推出，非通信得到)
        send_plan         : combine 所需元数据
    """
    device = x_local.device
    H = x_local.shape[1]
    k = topk_idx_local.shape[1]
    E_local = num_experts // world_size

    flat_expert = topk_idx_local.reshape(-1)
    flat_token = torch.arange(x_local.shape[0], device=device).repeat_interleave(k)

    # 关键: 按【全局专家 id】排序(等价于先按 rank、rank 内再按本地专家)，
    # 使发往同一 (rank, 专家) 的 token 连续。
    order = torch.argsort(flat_expert)
    sorted_expert = flat_expert[order]
    send_x = x_local[flat_token[order]]

    # 细粒度报数: 每个【全局专家】发多少行(长度 = num_experts = world_size*E_local)
    per_expert_send = torch.bincount(flat_expert, minlength=num_experts).to(torch.int64)  # (E,)
    # 按 rank 聚合成 send_counts(每 rank 发多少行) —— 本地求和，无需通信
    send_counts = per_expert_send.view(world_size, E_local).sum(dim=1)  # (world_size,)

    # all2all #1(报数, 细粒度): 把"我发给每个全局专家多少"交换。
    # 注意: all_to_all_single 按 rank 均分输入；per_expert_send 已按全局专家(=rank分块)排列，
    # 每个 rank 收到的是"各 rank 发给我这 E_local 个本地专家的行数"。
    recv_per_expert = torch.empty_like(per_expert_send)
    _a2a(recv_per_expert, per_expert_send, comm_on_cpu)        # 关键行: 唯一的报数(细粒度)
    # recv_per_expert[i] = 第 (i//E_local) 个 rank 发给我第 (i%E_local) 个本地专家的行数

    # 本地推出 recv_counts(每 rank 发我多少行) —— 关键: 由细粒度 group-sum 本地算出，不再通信!
    recv_counts = recv_per_expert.view(world_size, E_local).sum(dim=1)  # (world_size,)
    send_counts_list = send_counts.tolist()
    recv_counts_list = recv_counts.tolist()
    N_recv = int(recv_counts.sum().item())

    # all2all #2(数据): 发 token
    recv_x = torch.empty((N_recv, H), dtype=x_local.dtype, device=device)
    _a2a(recv_x, send_x.contiguous(), comm_on_cpu, recv_counts_list, send_counts_list)

    # 本地推出每个收到 token 的本地专家 id —— 不再通信 expert_id(省掉朴素版的第 3 次 all2all)!
    # recv_x 的排列: 先按源 rank(0..ws-1)，每个源 rank 内按本地专家(0..E_local-1)。
    # recv_per_expert 给出每段长度，cumsum 还原边界，repeat_interleave 生成本地专家标签。
    local_expert_labels = torch.arange(E_local, device=device).repeat(world_size)  # (E,) 每段对应的本地专家
    recv_local_expert = local_expert_labels.repeat_interleave(recv_per_expert)      # (N_recv,) 关键行
    recv_local_expert = recv_local_expert.to(torch.int64)

    send_plan = {
        "send_counts": send_counts_list, "recv_counts": recv_counts_list,
        "order": order, "flat_token": flat_token,
        "T_local": x_local.shape[0], "k": k, "N_recv": N_recv,
        # 通信量统计(用于和朴素版对比)
        "comm_ints_meta": per_expert_send.numel(),   # 报数向量大小
        "comm_rows_data": int(send_x.shape[0]),       # 发的数据行数
        "naive_extra_expert_ids": N_recv,             # 朴素版第3次额外发的 int 数
    }
    return recv_x, recv_local_expert, send_plan


def combine_2a2a(expert_out, topk_weight_local, send_plan, world_size, rank, H, comm_on_cpu=False):
    """combine: dispatch 逆路径寄回 + 加权。结构同朴素版(combine 本身就是 2 次:1报数隐含+1数据)。"""
    device = expert_out.device
    sp = send_plan
    send_back_counts = sp["recv_counts"]
    recv_back_counts = sp["send_counts"]
    N_back = sum(recv_back_counts)
    recv_back = torch.empty((N_back, H), dtype=expert_out.dtype, device=device)
    _a2a(recv_back, expert_out.contiguous(), comm_on_cpu, recv_back_counts, send_back_counts)

    order = sp["order"]; flat_token = sp["flat_token"]
    flat_weight = topk_weight_local.reshape(-1)[order]
    weighted = recv_back * flat_weight.unsqueeze(-1).to(recv_back.dtype)
    out_local = torch.zeros((sp["T_local"], H), dtype=expert_out.dtype, device=device)
    out_local.index_add_(0, flat_token[order], weighted)
    return out_local


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

    E, H, I, topk = 8, 256, 512, 2
    T_local = 32
    E_local = E // world_size

    full = ExpertWeights.random(E, H, I, device=device, seed=123)
    weights_local = ExpertWeights(
        full.W_gate[rank * E_local:(rank + 1) * E_local].contiguous(),
        full.W_up[rank * E_local:(rank + 1) * E_local].contiguous(),
        full.W_down[rank * E_local:(rank + 1) * E_local].contiguous())

    g2 = torch.Generator(device=device).manual_seed(rank + 1)
    x_local = (torch.randn(T_local, H, generator=g2, device=device, dtype=torch.float32) * (H ** -0.5)).to(torch.bfloat16)
    topk_idx_local = torch.randint(0, E, (T_local, topk), generator=g2, device=device)
    topk_weight_local = torch.full((T_local, topk), 1.0 / topk, device=device, dtype=torch.float32)

    recv_x, recv_local_expert, sp = dispatch_2a2a(x_local, topk_idx_local, E, world_size, rank, comm_on_cpu)
    expert_out = expert_compute(recv_x, recv_local_expert, weights_local)
    out_local = combine_2a2a(expert_out, topk_weight_local, sp, world_size, rank, H, comm_on_cpu)

    # 对拍: 本地用全局专家直接算
    ref = torch.zeros_like(out_local)
    for t in range(T_local):
        for j in range(topk):
            e = int(topk_idx_local[t, j].item())
            y = single_expert_ffn(x_local[t:t+1], full, e)
            ref[t] += (y[0] * float(topk_weight_local[t, j].item())).to(ref.dtype)
    d = (out_local.float() - ref.float()).abs().max().item()
    ok = d < 5e-2
    print(f"[rank {rank}] 2-all2all 对拍 max_abs_diff={d:.4e}  {'PASS' if ok else 'FAIL'}")
    if rank == 0:
        print(f"[rank {rank}] 通信量: dispatch 报数向量 {sp['comm_ints_meta']} ints + 数据 {sp['comm_rows_data']} 行")
        print(f"[rank {rank}] 对比朴素版: 省掉了第3次 all2all 的 {sp['naive_extra_expert_ids']} 个逐-token expert_id")

    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()