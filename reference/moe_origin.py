import torch
import torch.distributed as dist
import deep_ep
import random

# 1. DeepEP 的 dispatch (不执行 combine)
def dispatch_only_deep_ep(buffer, x, topk_idx, num_tokens, num_experts):
    recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x, topk_idx, num_tokens, num_experts, use_fp8=False
    )
    return recv_x, recv_count, handle

# 2. All-to-all 的 dispatch (不执行 combine)
def dispatch_only_all2all(x, topk_idx, num_tokens, num_experts, world_size, rank):
    num_local_experts = num_experts // world_size
    start_exp = rank * num_local_experts
    end_exp   = start_exp + num_local_experts

    # 1️⃣ 按 (token, k) 粒度统计
    expert_tokens = [[] for _ in range(num_experts)]
    # expert_token_map: 记录每个token分发到的(专家id, k)列表
    expert_token_map = [[] for _ in range(num_tokens)]
    for t in range(num_tokens):
        for k in range(topk_idx.shape[1]):
            exp = int(topk_idx[t, k].item())
            if exp < 0:
                continue
            expert_tokens[exp].append(x[t])     # 每选一次加一次
            expert_token_map[t].append((exp, len(expert_tokens[exp])))  # 记录分发映射

    # 2️⃣ 构建发送顺序 & 拆分
    send_order = []
    send_splits = [0] * world_size
    for expert_id in range(num_experts):
        target_rank = expert_id // num_local_experts
        send_splits[target_rank] += len(expert_tokens[expert_id])
        send_order.extend(expert_tokens[expert_id])

    # 3️⃣ 第一次 All-to-All: 传递每个rank的总token数量
    send_splits = torch.tensor(send_splits, dtype=torch.int32, device=x.device)
    recv_splits = torch.empty_like(send_splits)
    dist.all_to_all_single(recv_splits, send_splits)

    # 4️⃣ 构建每个专家的token数量信息
    expert_counts = [len(expert_tokens[exp]) for exp in range(num_experts)]
    expert_counts_tensor = torch.tensor(expert_counts, dtype=torch.int32, device=x.device)

    # 5️⃣ 第二次 All-to-All: 传递每个专家的token数量
    # 每个rank发送所有专家的token数量, 接收所有专家的token数量
    recv_expert_counts = torch.empty_like(expert_counts_tensor)
    dist.all_to_all_single(recv_expert_counts, expert_counts_tensor)

    # 6️⃣ 第三次 All-to-All: 传递实际的token数据
    send_x = torch.stack(send_order, dim=0) if send_order else torch.empty((0, x.shape[1]), dtype=x.dtype, device=x.device)
    total_recv = int(recv_splits.sum().item())
    recv_x_raw = torch.empty((total_recv, x.shape[1]), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(recv_x_raw, send_x, recv_splits.tolist(), send_splits.tolist())

    # 7️⃣ 重建本地 expert buffer (使用专家token数量信息)
    local_expert_tokens = [[] for _ in range(num_local_experts)]
    ptr = 0
    for i in range(recv_expert_counts.shape[0]):
        cnt = int(recv_expert_counts[i].item())
        expert_id = i % num_local_experts
        local_expert_tokens[expert_id].append(recv_x_raw[ptr:ptr + cnt])
        ptr += cnt
    # 把tensor cat起来
    for i in range(len(local_expert_tokens)):
        local_expert_tokens[i] = torch.cat(local_expert_tokens[i], dim=0)

    # 8️⃣ 构造输出
    max_len = num_tokens * world_size
    recv_x = torch.zeros((num_local_experts, max_len, x.shape[1]), dtype=x.dtype, device=x.device)
    recv_count = torch.tensor([lst.shape[0] for lst in local_expert_tokens], dtype=torch.int32, device=x.device)

    for i, lst in enumerate(local_expert_tokens):
        if len(lst) > 0:
            recv_x[i, :len(lst)] = lst
    # 返回本地 expert buffer 及其分发映射
    return recv_x, recv_count, expert_token_map, recv_expert_counts


# 对齐维度: 只比较有效 token
def compare_dispatch(deep_ep_recv_x, deep_ep_recv_count, all2all_recv_x, all2all_recv_count):
    if dist.get_rank() == 1 and 1:
        print(f"deepep=== rank={dist.get_rank()} ===")
        recv_count = deep_ep_recv_count
        for exp_id in range(recv_count.numel()):
            cnt = int(recv_count[exp_id].item())
            print(f"Expert {exp_id} (recv_count={cnt}):")
            for tok_id in range(cnt):
                # 取第 tok_id 行, 列用省略号
                row = deep_ep_recv_x[exp_id, tok_id]
                print(f"deepep  token {tok_id:02d}: {row}")
                row = all2all_recv_x[exp_id, tok_id]
                print(f"all2all token {tok_id:02d}: {row}")

    diff = 0.0
    for i in range(deep_ep_recv_x.shape[0]):
        cnt = int(deep_ep_recv_count[i].item())
        a = deep_ep_recv_x[i, :cnt]
        b = all2all_recv_x[i, :cnt]
        if a.shape != b.shape:
            print(f"Shape mismatch at expert {i}: deep_ep {a.shape}, all2all {b.shape}")
            continue
        print("ljl--shape--same!!!")
        d = (a - b).abs().max().item()
        print(f"Expert {i} max diff: {d}", "  rank=", dist.get_rank())
        diff = max(diff, d)
    return diff


# =========== Expert Forward Functions ===========

def expert_forward_all2all(recv_x, recv_count, expert_fn):
    """
    对 all2all 分发结果做专家计算。
    expert_fn: (tokens: Tensor) -> Tensor
    返回: (num_experts, max_len, hidden)
    """
    num_experts, max_len, hidden = recv_x.shape
    out = torch.zeros_like(recv_x)
    for i in range(num_experts):
        cnt = int(recv_count[i].item())
        if cnt > 0:
            out[i, :cnt] = expert_fn(recv_x[i, :cnt])
    return out

def expert_forward_deepep(recv_x, recv_count, expert_fn):
    """
    对 deepep 分发结果做专家计算。
    expert_fn: (tokens: Tensor) -> Tensor
    返回: (num_experts, max_len, hidden)
    """
    num_experts, max_len, hidden = recv_x.shape
    out = torch.zeros_like(recv_x)
    for i in range(num_experts):
        cnt = int(recv_count[i].item())
        if cnt > 0:
            out[i, :cnt] = expert_fn(recv_x[i, :cnt])
    return out


def all2all_combine(all2all_out, all2all_recv_count, recv_expert_counts, expert_token_map, topk_weights, num_tokens, num_experts, num_topk, world_size, rank):
    """
    all2all combine: 先将专家计算结果通过 all2all 分发回原始 token 所在 rank, 然后本地根据分发映射和 topk_weights 对每个 token 做加权聚合。
    """
    device = all2all_out.device
    hidden = all2all_out.shape[2]
    num_local_experts = num_experts // world_size

    # 1. 收集本地 expert buffer 的输出和映射
    send_tokens = []
    for local_exp in range(num_local_experts):
        cnt = int(all2all_recv_count[local_exp].item())
        send_tokens.append(all2all_out[local_exp, : cnt])

    send_ranks = [[] for _ in range(world_size)]
    index_flag = [0] * num_local_experts
    for i in range(len(recv_expert_counts)):
        local_experts = i % num_local_experts
        rank_id = i // num_local_experts
        begin_index = index_flag[local_experts]
        end_index = index_flag[local_experts] + recv_expert_counts[i]
        send_ranks[rank_id].append(send_tokens[local_experts][begin_index : end_index])
        index_flag[local_experts] = index_flag[local_experts] + recv_expert_counts[i]

    # send_ranks: 每个目标rank一个list, list里是本地要发给该rank的token tensor
    # 1. 统计每个rank要发多少token
    send_counts = [sum([t.shape[0] for t in send_ranks[r]]) for r in range(world_size)]

    # 2. 拼接所有要发给每个rank的token
    send_tokens_cat = torch.cat([t for sublist in send_ranks for t in sublist], dim=0) if any(send_ranks) else torch.empty((0, all2all_out.shape[2]), dtype=all2all_out.dtype, device=device)

    # 3. all2all_single统计每个rank接收多少token
    send_counts_tensor = torch.tensor(send_counts, dtype=torch.int32, device=device)
    recv_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    total_recv = int(recv_counts_tensor.sum().item())

    # 4. all2all_single发送token
    recv_tokens_cat = torch.empty((total_recv, all2all_out.shape[2]), dtype=all2all_out.dtype, device=device)
    dist.all_to_all_single(recv_tokens_cat, send_tokens_cat, recv_counts_tensor.tolist(), send_counts_tensor.tolist())

    # 5. 本地加权聚合 (只用本地数据, 不再做all2all)
    out = torch.zeros((num_tokens, all2all_out.shape[2]), dtype=all2all_out.dtype, device=device)
    # 统计每个专家buffer当前已用到的位置
    expert_tokens_num = [0] * num_experts
    for t in range(len(expert_token_map)):
        topk = len(expert_token_map[t])
        for k in range(topk):
            expert_id = expert_token_map[t][k][0]
            expert_tokens_num[expert_id] += 1

    expert_tokens_tensor = []
    cnt = 0
    for i in range(num_experts):
        length = expert_tokens_num[i]
        expert_tokens_tensor.append(recv_tokens_cat[cnt : cnt + length][:])
        cnt = cnt + length

    for t in range(num_tokens):
        topk = len(expert_token_map[t])
        for k in range(topk):
            w = topk_weights[t, k].item()
            expert_id, index = expert_token_map[t][k]
            out[t] += expert_tokens_tensor[expert_id][index - 1][:] * w
    return out

# =========== Dummy Expert Function ===========
def dummy_expert_fn(tokens):
    # 这里可以替换为任意专家前向逻辑, 比如线性层、激活等
    return tokens * 1  # 简单起见, 乘2

def compare_expert_forward(all2all_out, all2all_recv_count, deepep_out, deepep_recv_count):
    """
    比对 all2all 和 deepep 的专家前向输出。
    """
    expert_diff = 0.0
    for i in range(all2all_out.shape[0]):
        cnt = int(all2all_recv_count[i].item())
        a = all2all_out[i, :cnt]
        b = deepep_out[i, :cnt]
        if a.shape != b.shape:
            print(f"Expert {i} shape mismatch after forward: all2all {a.shape}, deepep {b.shape}")
            continue
        d = (a - b).abs().max().item()
        print(f"Expert {i} forward max diff: {d}")
        expert_diff = max(expert_diff, d)
    return expert_diff


# 主脚本
if __name__ == '__main__':
    import os
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    seed = 0
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    random.seed(seed + rank)

    num_tokens = 256
    hidden = 7168
    num_experts = 8
    num_topk = 2

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device='cuda')

    # 均值概率生成weights
    topk_weights = torch.full((num_tokens, num_topk), 1.0 / num_topk,
                              dtype=torch.float32, device=topk_idx.device)
    # 随机概率生成weights
    # topk_weights = torch.rand_like(topk_idx, dtype=torch.float32, device=topk_idx.device)
    # topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)  # 归一化

    group = dist.group.WORLD
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, world_size, num_experts)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // world_size)

    # DeepEP dispatch
    deep_ep_recv_x, deep_ep_recv_count, deep_ep_handle = dispatch_only_deep_ep(buffer, x, topk_idx, num_tokens, num_experts)
    # DeepEP expert forward
    deepep_out = expert_forward_deepep(deep_ep_recv_x, deep_ep_recv_count, dummy_expert_fn)
    # deepep combine (修正: 直接传 handle)
    deepep_combined, *_ = buffer.low_latency_combine(
        deepep_out, topk_idx, topk_weights, deep_ep_handle
    )

    # All2All dispatch
    all2all_recv_x, all2all_recv_count, all2all_token_map, recv_expert_counts = dispatch_only_all2all(x, topk_idx, num_tokens, num_experts, world_size, rank)
    # All2All expert forward
    all2all_out = expert_forward_all2all(all2all_recv_x, all2all_recv_count, dummy_expert_fn)
    # all2all combine (修正: 传 expert_token_map, topk_weights, world_size, rank)
    all2all_combined = all2all_combine(all2all_out, all2all_recv_count, recv_expert_counts,
        all2all_token_map, topk_weights, num_tokens, num_experts, num_topk, world_size, rank)

    # 对齐维度对比
    if rank == 0 or 1:
        diff = compare_dispatch(deep_ep_recv_x, deep_ep_recv_count,
                                all2all_recv_x, all2all_recv_count)
        print("dispatch max diff:", diff, "rank=", rank)

        # =========== Expert Forward & Compare ===========
        expert_diff = compare_expert_forward(all2all_out, all2all_recv_count, deepep_out, deep_ep_recv_count)
        print("expert forward max diff:", expert_diff, "rank=", rank)

        # 比较 combine 输出
        if rank == 1 or 1:
            print("deepep_combined (rank=", rank, "):\n", deepep_combined)
            print("all2all_combined (rank=", rank, "):\n", all2all_combined)
            combine_diff = (all2all_combined - deepep_combined).abs().max().item()
            # 计算相对误差
            deepep_norm = deepep_combined.abs().max().item() if deepep_combined.abs().max().item() != 0 else 1.0
            combine_rel_diff = combine_diff / deepep_norm
            print("combine max diff (abs):", combine_diff, "rank=", rank)
            print("combine max diff (rel):", combine_rel_diff, "rank=", rank)
            assert combine_rel_diff <= 1e-2, "error, test fail!"
