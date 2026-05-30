class AlltoAll:
    def __init__(self, num_experts, world_size, rank):
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.num_local_experts = num_experts // world_size

        # 保存用于恢复顺序的关键索引
        self.sorted_idx = None        # [Step 2] 用于恢复 (Batch*TopK) 的原始顺序
        self.sorted_eidx = None       # [Step 4] 用于恢复 Rank-Major 的接收顺序
        self.send_split = None
        self.recv_split = None

    def dispatch(self, x, topk_idx):
        # x: [num_tokens, hidden_dim]
        # topk_idx: [num_tokens, topk]
        num_tokens, hidden = x.shape
        _, topk = topk_idx.shape

        # --- Step 1: 计算并交换计数 ---
        topk_idx_flat = topk_idx.view(-1)
        # 本地发出的每个 Global Expert 的计数
        num_tokens_per_expert = torch.bincount(topk_idx_flat, minlength=self.num_experts)
        # 接收到的每个 Global Expert (来自所有 Rank) 的计数
        # 注意: 这里接收到的布局是 [Rank0_E0, Rank0_E1... | Rank1_E0...] (假设 num_tokens_per_expert 本身按专家ID排好序)
        num_tokens_per_expert_group = torch.empty_like(num_tokens_per_expert)
        dist.all_to_all_single(num_tokens_per_expert_group, num_tokens_per_expert)

        # --- Step 2: 构建 Split 和 Buffer ---
        # 2.1 计算 Rank 级别的通信量
        # 假设 Expert ID 是连续的, 则 view(world_size, -1) 正确将 Expert 分组归属到 Rank
        send_count = num_tokens_per_expert.view(self.world_size, -1).sum(dim=1)
        recv_count = num_tokens_per_expert_group.view(self.world_size, -1).sum(dim=1)
        self.send_split = send_count.tolist()
        self.recv_split = recv_count.tolist()

        # 2.2 排序并构建发送 Buffer
        # 关键: 我们需要保存 sorted_idx (排序索引), 而不是排序后的值
        sorted_idx = torch.argsort(topk_idx_flat)
        self.sorted_idx = sorted_idx  # <--- 保存这个用于 Combine 恢复

        # 扩展 x 到 (Batch * TopK), 然后按目标 Expert 排序
        # 优化: 只取需要的行, 不需要 repeat_interleave 整个 x 再索引
        src_token_idx = torch.arange(num_tokens, device=x.device).repeat_interleave(topk)
        sorted_src_token_idx = src_token_idx[sorted_idx]
        send_buff = x[sorted_src_token_idx]

        total_recv = self.recv_split_sum = sum(self.recv_split)
        recv_buff = torch.empty((total_recv, hidden), dtype=x.dtype, device=x.device)

        # --- Step 3: 交换 Token 数据 ---
        dist.all_to_all_single(recv_buff, send_buff, self.recv_split, self.send_split)

        # --- Step 4: 本地重排 (Rank-Major -> Expert-Major) ---
        # 目前 recv_buff 是按发送 Rank 排序的。我们需要按 Local Expert ID 排序。
        # 生成对应的 Local Expert ID 标签
        # 这里的顺序对应 num_tokens_per_expert_group 的扁平化顺序
        local_expert_ids_flat = torch.arange(self.world_size * self.num_local_experts, device=x.device) % self.num_local_experts
        # 扩展出每个 Token 对应的 Local Expert ID
        recv_buff_eid = torch.repeat_interleave(local_expert_ids_flat, num_tokens_per_expert_group)

        sorted_eidx = torch.argsort(recv_buff_eid)
        self.sorted_eidx = sorted_eidx   # <--- 保存这个用于 Combine 逆操作

        recv_x_permuted = recv_buff[sorted_eidx]

        # 计算每个本地专家分到的数据量, 返回给计算层切分使用
        # num_tokens_per_expert_group 形状: [World_Size * Num_Local]
        # 我们需要算出 [Local_Expert_0_Total, Local_Expert_1_Total ...]
        local_expert_count = num_tokens_per_expert_group.view(self.world_size, -1).sum(dim=0)

        return recv_x_permuted, local_expert_count

    def combine(self, expert_out, topk_weights):
        # expert_out: [Total_Recv, Hidden] (Expert-Major)
        num_tokens, topk = topk_weights.shape

        # --- Step 1: 本地逆重排 (Expert-Major -> Rank-Major) ---
        # 错误修正: 不能用 expert_out[self.sorted_eidx]
        # 必须用 Scatter 操作:  out[idx] = val
        recv_buff_restored = torch.empty_like(expert_out)
        recv_buff_restored[self.sorted_eidx] = expert_out

        # --- Step 2: 数据发回 (All-to-All) ---
        # 这里的 send/recv 是相对于 Dispatch 的逆向
        send_buff = recv_buff_restored
        recv_buff = torch.empty((sum(self.send_split), expert_out.shape[1]), dtype=expert_out.dtype, device=expert_out.device)

        # 注意 splits 互换
        dist.all_to_all_single(recv_buff, send_buff, output_split_sizes=self.send_split, input_split_sizes=self.recv_split)

        # --- Step 3: 恢复原始序列 (Original Batch Order) ---
        # recv_buff 目前是按 (Target_Expert_ID) 排序的 (因为 dispatch 发送时排过序)
        # 我们需要把它放回原来的 (Batch, TopK) 位置

        # 创建一个能容纳所有 results 的 buffer (Flat)
        out_flat = torch.empty((num_tokens * topk, expert_out.shape[1]), dtype=expert_out.dtype, device=expert_out.device)

        # 错误修正: 不能用 recv_buff[self.sorted_idx]
        # 必须用 Scatter:  out[idx] = val
        out_flat[self.sorted_idx] = recv_buff

        # --- Step 4: 加权求和 ---
        out_reshaped = out_flat.view(num_tokens, topk, -1)
        out = (out_reshaped * topk_weights.unsqueeze(-1)).sum(dim=1)

        return out
