"""
[Phase 2] ep_dist_naive.py — 真·多进程 NCCL 专家并行(朴素正确版)

做什么:
    用 torchrun 起 world_size 个进程(每进程一张卡)，每进程只持有自己的 token 和本地专家，
    跨进程数据用真实 dist.all_to_all_single 搬运。实现一层 MoE 的 dispatch→expert→combine，
    并与单进程参考 ep_reference 对拍(每个 rank 对自己那份 token 的输出做校验)。

    与 ep_reference 的区别:
      ep_reference  : 单进程"假装"分卡，append 到 inbox 模拟搬运。
      本文件        : 真起 N 进程，inbox 之间的搬运 = 真 all_to_all_single(NCCL/gloo)。
    与老师 reference 的关系: 同一套 dispatch/combine 逻辑，但用真实 FFN 专家 + 真实路由结构。

    本版【故意朴素】: dispatch 用 3 次 all2all(2 报数 + 1 数据)，combine 2 次。
    不追求快，先求正确可对拍。优化(2次all2all/融合kernel/重叠)在后续版本。

数据划分:
    全局 T_global 个 token 均分到各 rank，每 rank T_local 个。
    全局 E 个专家连续切分: rank r 持有专家 [r*E_local, (r+1)*E_local)。

运行:
    torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.ep_dist_naive
    # 单机无 GPU 调试可改 backend=gloo（见 main）
"""
import os
import torch
import torch.distributed as dist
from ..phase1_single_gpu_moe.common_moe import ExpertWeights, single_expert_ffn


# ---------------------------------------------------------------------------
# dispatch: 把本 rank 的 (token,k) 按专家所在 rank 分组，all2all 发送
# ---------------------------------------------------------------------------
def dispatch(x_local, topk_idx_local, num_experts, world_size, rank, comm_on_cpu=False):
    """把本地 token 按目标 rank 分组并 all2all 发送。

    输入:
        x_local        : (T_local, H)        本 rank 的 token
        topk_idx_local : (T_local, k)        本 rank token 选的全局专家 id
    输出:
        recv_x         : (N_recv, H)         收到的、命中本 rank 专家的 token
        recv_expert    : (N_recv,)           每个收到 token 对应的【本地】专家 id
        send_plan      : dict                combine 时按相同路径寄回所需的元数据
    """
    device = x_local.device
    H = x_local.shape[1]
    k = topk_idx_local.shape[1]
    E_local = num_experts // world_size

    # 1. 展平 (token,k)，算每个选择的目标 rank
    flat_expert = topk_idx_local.reshape(-1)                  # (T_local*k,)
    flat_token = torch.arange(x_local.shape[0], device=device).repeat_interleave(k)
    target_rank = flat_expert // E_local                      # 关键行: 专家 id -> 目标 rank

    # 2. 按目标 rank 排序，使发往同一 rank 的数据连续(all2all 要求按 rank 分块)
    order = torch.argsort(target_rank)
    sorted_rank = target_rank[order]
    send_x = x_local[flat_token[order]]                       # 按目标 rank 排好的待发 token
    send_expert = flat_expert[order]                          # 对应的全局专家 id(随数据一起发)

    # 3. 每个目标 rank 发多少行
    send_counts = torch.bincount(sorted_rank, minlength=world_size)  # (world_size,)

    # 4. all2all #1(报数): 我发给各 rank 多少 -> 我从各 rank 收多少
    # 关键行: gloo 后端不支持 cuda tensor 通信，须搬到 CPU 做 all2all 再搬回。
    def _a2a(out_t, in_t, out_split=None, in_split=None):
        if comm_on_cpu:
            in_c = in_t.cpu()
            out_c = torch.empty(out_t.shape, dtype=out_t.dtype, device="cpu")
            dist.all_to_all_single(out_c, in_c, out_split, in_split)
            out_t.copy_(out_c.to(out_t.device))
        else:
            dist.all_to_all_single(out_t, in_t, out_split, in_split)
    recv_counts = torch.empty_like(send_counts)
    _a2a(recv_counts, send_counts)                            # 先报数
    send_counts_list = send_counts.tolist()
    recv_counts_list = recv_counts.tolist()
    N_recv = int(recv_counts.sum().item())

    # 5. all2all #2(数据): 真正发送 token
    recv_x = torch.empty((N_recv, H), dtype=x_local.dtype, device=device)
    _a2a(recv_x, send_x.contiguous(), recv_counts_list, send_counts_list)

    # 6. all2all #3(数据): 同步发送每个 token 对应的专家 id(int)，combine 用不到专家但
    #    expert 计算要知道用哪个本地专家。这里随数据发 expert id。
    send_expert_i = send_expert.to(torch.int64)
    recv_expert = torch.empty((N_recv,), dtype=torch.int64, device=device)
    _a2a(recv_expert, send_expert_i.contiguous(), recv_counts_list, send_counts_list)
    recv_local_expert = recv_expert % E_local                 # 全局专家 id -> 本地专家 id

    send_plan = {
        "send_counts": send_counts_list, "recv_counts": recv_counts_list,
        "order": order, "flat_token": flat_token,
        "T_local": x_local.shape[0], "k": k, "N_recv": N_recv,
    }
    return recv_x, recv_local_expert, send_plan


def expert_compute(recv_x, recv_local_expert, weights_local: ExpertWeights):
    """本 rank 用本地专家算收到的 token。weights_local 只含本 rank 的 E_local 个专家。"""
    out = torch.empty_like(recv_x)
    E_local = weights_local.num_experts
    for le in range(E_local):
        mask = (recv_local_expert == le)
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0]
            out[idx] = single_expert_ffn(recv_x[idx], weights_local, le).to(out.dtype)
    return out


def combine(expert_out, topk_weight_local, send_plan, world_size, rank, H, comm_on_cpu=False):
    """把专家结果按 dispatch 的逆路径寄回原 rank，并按 topk_weight 加权求和回原 token。

    输入:
        expert_out       : (N_recv, H)   本 rank 算完的结果(顺序同 recv_x)
        topk_weight_local: (T_local, k)  本 rank token 的权重
    输出:
        out_local        : (T_local, H)  本 rank 每个 token 的最终加权结果
    """
    device = expert_out.device
    sp = send_plan
    # combine 是 dispatch 的逆: 把 expert_out 按【当初谁发来的】寄回去。
    # 当初我从各 rank 收了 recv_counts 行；现在原样按 recv_counts 发回、send_counts 收回。
    send_back_counts = sp["recv_counts"]                      # 现在发回的 = 当初收到的
    recv_back_counts = sp["send_counts"]                      # 现在收回的 = 当初发出的
    N_back = sum(recv_back_counts)
    recv_back = torch.empty((N_back, H), dtype=expert_out.dtype, device=device)
    if comm_on_cpu:
        ec = expert_out.contiguous().cpu()
        rb = torch.empty((N_back, H), dtype=expert_out.dtype, device="cpu")
        dist.all_to_all_single(rb, ec, recv_back_counts, send_back_counts)
        recv_back.copy_(rb.to(device))
    else:
        dist.all_to_all_single(recv_back, expert_out.contiguous(), recv_back_counts, send_back_counts)

    # recv_back 的顺序对应 dispatch 时"按目标 rank 排序后的 send 顺序"(order)。
    # 用 order 逆置换 + 加权，累加回原 token。
    order = sp["order"]
    flat_token = sp["flat_token"]
    k = sp["k"]
    flat_weight = topk_weight_local.reshape(-1)[order]        # 按发送顺序排好的权重
    weighted = recv_back * flat_weight.unsqueeze(-1).to(recv_back.dtype)
    out_local = torch.zeros((sp["T_local"], H), dtype=expert_out.dtype, device=device)
    out_local.index_add_(0, flat_token[order], weighted)      # 关键行: 加权回原 token
    return out_local


def main():
    # 后端选择(关键认知):
    #   nccl: GPU 通信库，要求【一卡一 rank】，禁止两 rank 共卡(报 Duplicate GPU)。真双卡用它。
    #   gloo: CPU 通信库，【允许多 rank 共用一张 GPU】，通信走 CPU。本地单卡验证通信逻辑用它。
    # EP_BACKEND 显式指定；默认: 有多卡用 nccl，本地单卡调试用 gloo。
    backend = os.environ.get("EP_BACKEND", "gloo" if os.environ.get("SINGLE_GPU") == "1" else
                             ("nccl" if torch.cuda.is_available() else "gloo"))
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    single_gpu = os.environ.get("SINGLE_GPU", "0") == "1"

    # 计算设备: 单卡模拟时所有 rank 都用 cuda:0；真双卡用各自 local_rank。
    if torch.cuda.is_available():
        gpu_id = 0 if single_gpu else local_rank
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    # 通信设备: gloo 不支持 cuda tensor 的 all2all，须在 CPU 通信；nccl 在 GPU 通信。
    comm_on_cpu = (backend == "gloo")
    print(f"[rank {rank}] backend={backend}, compute_device={device}, comm_on_cpu={comm_on_cpu}")

    # 全局配置(小规模便于对拍)
    E, H, I, topk = 8, 256, 512, 2
    T_local = 32
    E_local = E // world_size

    # 各 rank 用相同 seed 生成【全局一致】的专家权重，再各取本地切片(保证对拍时权重一致)
    g = torch.Generator(device=device).manual_seed(123)
    full = ExpertWeights.random(E, H, I, device=device, seed=123)
    Wg_local = full.W_gate[rank * E_local:(rank + 1) * E_local].contiguous()
    Wu_local = full.W_up[rank * E_local:(rank + 1) * E_local].contiguous()
    Wd_local = full.W_down[rank * E_local:(rank + 1) * E_local].contiguous()
    weights_local = ExpertWeights(Wg_local, Wu_local, Wd_local)

    # 本 rank 的 token 与路由(各 rank 用不同 seed，模拟不同数据)
    g2 = torch.Generator(device=device).manual_seed(rank + 1)
    x_local = (torch.randn(T_local, H, generator=g2, device=device, dtype=torch.float32) * (H ** -0.5)).to(torch.bfloat16)
    topk_idx_local = torch.randint(0, E, (T_local, topk), generator=g2, device=device)
    topk_weight_local = torch.full((T_local, topk), 1.0 / topk, device=device, dtype=torch.float32)

    # ---- 真·多进程 dispatch → expert → combine ----
    recv_x, recv_local_expert, send_plan = dispatch(x_local, topk_idx_local, E, world_size, rank, comm_on_cpu)
    expert_out = expert_compute(recv_x, recv_local_expert, weights_local)
    out_local = combine(expert_out, topk_weight_local, send_plan, world_size, rank, H, comm_on_cpu)

    # ---- 对拍: 本 rank 用全局专家在本地直接算自己 token 的结果 ----
    ref = torch.zeros_like(out_local)
    for t in range(T_local):
        for j in range(topk):
            e = int(topk_idx_local[t, j].item())
            y = single_expert_ffn(x_local[t:t+1], full, e)
            ref[t] += (y[0] * float(topk_weight_local[t, j].item())).to(ref.dtype)
    d = (out_local.float() - ref.float()).abs().max().item()
    print(f"[rank {rank}] dispatch/combine 对拍 max_abs_diff={d:.4e}  {'PASS' if d < 5e-2 else 'FAIL'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()