"""
================================================================================
 MoE 专家并行 (Expert Parallelism) 通信对照脚本  —— 初学者精读版
================================================================================

【这份代码到底在做什么?】
在 MoE (Mixture of Experts, 混合专家) 模型里:
  - 模型有很多个「专家」(expert), 本质就是一组独立的小型前馈网络 (FFN)。
  - 每个 token 不会经过所有专家, 而是由一个 router 选出 top-k 个专家 (这里 k=2)。
  - 当专家数量很多时, 我们把专家「切开」放到不同的 GPU (这里叫 rank) 上,
    这就是「专家并行 / Expert Parallelism」。

  问题来了: token 在 GPU-A 上产生, 但它选中的专家可能住在 GPU-B 上。
  于是需要三步通信:
    1) Dispatch (分发): 把每个 token 送到「它选中的专家所在的那张 GPU」。
    2) Expert Forward (专家计算): 每张 GPU 用本地专家处理收到的 token。
    3) Combine (聚合): 把计算结果送回 token 原来所在的 GPU,
       并按 router 给的权重 (topk_weights) 做加权求和, 得到每个 token 的最终输出。

  这三步里最难、最影响性能的就是「跨 GPU 搬数据」(all-to-all 通信)。

【这份脚本的目的】
  对比两种实现是否「数值等价」:
    路径 A: DeepEP 库   —— 工业级高性能 kernel (low_latency_dispatch/combine)。
    路径 B: 手写 all2all —— 用 PyTorch 的 dist.all_to_all_single 一步步拼出来。
  如果两条路径输出一致, 就说明我们「真的理解了」DeepEP 在底层做了什么。
  这是一种非常好的学习方法: 用一个看得懂的「参考实现」去解剖一个黑盒库。

【关键名词速查】
  - rank        : 一个进程 / 一张 GPU 的编号 (0, 1, 2, ...)。
  - world_size  : 总共有几张 GPU / 几个进程。
  - num_experts : 专家总数 (跨所有 GPU)。
  - num_local_experts = num_experts // world_size : 每张 GPU 上有几个专家。
  - topk_idx    : 形状 (num_tokens, k), 每个 token 选中的 k 个专家的 id。
  - topk_weights: 形状 (num_tokens, k), 每个选择对应的聚合权重 (router softmax 概率)。
  - all_to_all  : 集合通信原语。每个 rank 都把「一份不同的数据」发给「每一个其他 rank」,
                  同时从每一个 rank 收一份。可以想象成 N×N 的「数据洗牌矩阵」。

【运行方式】
  这是一个多进程分布式程序, 通常用:
      torchrun --nproc_per_node=<GPU数> moe_origin.py
================================================================================
"""

import torch
import torch.distributed as dist   # PyTorch 分布式通信 (集合通信原语: all_to_all 等)
import deep_ep                     # DeepEP: 高性能 MoE 通信库 (待对照的「黑盒」)
import random


# ============================================================================
# 1. DeepEP 的 dispatch (只做分发, 不做 combine)
# ============================================================================
def dispatch_only_deep_ep(buffer, x, topk_idx, num_tokens, num_experts):
    """
    用 DeepEP 库把 token 分发给各 GPU 上的专家。
    几乎所有重活都被 buffer.low_latency_dispatch 这一个调用包掉了 —— 这正是库的价值。

    参数:
      buffer      : DeepEP 预分配好的通信缓冲区对象 (持有 RDMA/NVLink 资源)。
      x           : 本 rank 的输入 token, 形状 (num_tokens, hidden)。
      topk_idx    : 每个 token 选中的专家 id, 形状 (num_tokens, k)。
      num_tokens  : 本 rank 的 token 数。
      num_experts : 专家总数。

    返回:
      recv_x     : 本 rank 上的专家收到的 token, 形状大致 (num_local_experts, max_len, hidden)。
      recv_count : 每个本地专家实际收到多少 token。
      handle     : 一个「通信句柄」, 记录了这次 dispatch 的路由信息;
                   combine 阶段需要它来「按原路把结果送回去」。
    """
    recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x, topk_idx, num_tokens, num_experts, use_fp8=False  # use_fp8=False: 用原始精度传输, 不做 FP8 量化
    )
    return recv_x, recv_count, handle


# ============================================================================
# 2. 手写 All-to-all 的 dispatch (只做分发, 不做 combine)
#    —— 这是 DeepEP dispatch 的「白盒参考实现」, 一步步拆给你看
# ============================================================================
def dispatch_only_all2all(x, topk_idx, num_tokens, num_experts, world_size, rank):
    # 每张 GPU 负责的专家是「连续的一段」: rank r 负责 [start_exp, end_exp)
    num_local_experts = num_experts // world_size   # 本 rank 上的专家个数
    start_exp = rank * num_local_experts            # 本 rank 第一个专家的全局 id
    end_exp   = start_exp + num_local_experts        # (这里只是示意, 下面没直接用到)

    # 1️⃣ 按 (token, k) 粒度统计: 把每个 token「按它选中的专家」分桶
    #    注意: 一个 token 选了 k 个专家, 就会被复制 k 份, 分别放进 k 个专家的桶里。
    expert_tokens = [[] for _ in range(num_experts)]   # expert_tokens[e] = 选中专家 e 的所有 token 行
    # expert_token_map: 记录每个 token 分发到的 (专家id, 在该专家桶里的序号) 列表
    #   —— 这是「分发地图」, combine 阶段要靠它把结果再装回原 token。
    expert_token_map = [[] for _ in range(num_tokens)]
    for t in range(num_tokens):                        # 遍历每个 token
        for k in range(topk_idx.shape[1]):             # 遍历它选的 k 个专家
            exp = int(topk_idx[t, k].item())           # 第 k 个选择的专家 id
            if exp < 0:                                 # 负数表示「无效/未选择」, 跳过
                continue
            expert_tokens[exp].append(x[t])            # 把这份 token 复制进专家 exp 的桶 (每选一次加一次)
            # 记录 (专家id, 桶内位置)。位置 = append 之后的长度 = 1-based 序号。
            expert_token_map[t].append((exp, len(expert_tokens[exp])))

    # 2️⃣ 构建「发送顺序」并按目标 rank 拆分
    #    我们要把数据打平成一条发送队列 send_order, 顺序 = 按专家 id 从小到大。
    #    同时统计「发给每个 rank 一共多少 token」(send_splits)。
    send_order = []                       # 最终要发出去的 token, 排列顺序 = 专家 id 升序
    send_splits = [0] * world_size        # send_splits[r] = 本 rank 要发给 rank r 的 token 总数
    for expert_id in range(num_experts):
        target_rank = expert_id // num_local_experts   # 专家 expert_id 住在哪个 rank
        send_splits[target_rank] += len(expert_tokens[expert_id])
        send_order.extend(expert_tokens[expert_id])    # 按专家顺序把 token 接到发送队列尾部

    # 3️⃣ 第一次 All-to-All: 交换「每个 rank 要给我发多少 token」
    #    每个 rank 把自己的 send_splits 发出去, 收到的 recv_splits[r] = rank r 要发给我多少 token。
    #    (这一步是「先对账数量」, 这样接收方才知道该开多大的接收 buffer。)
    send_splits = torch.tensor(send_splits, dtype=torch.int32, device=x.device)
    recv_splits = torch.empty_like(send_splits)
    dist.all_to_all_single(recv_splits, send_splits)

    # 4️⃣ 构建「每个专家收到多少 token」的信息 (粒度比 rank 更细, 细到单个专家)
    expert_counts = [len(expert_tokens[exp]) for exp in range(num_experts)]
    expert_counts_tensor = torch.tensor(expert_counts, dtype=torch.int32, device=x.device)

    # 5️⃣ 第二次 All-to-All: 交换「每个专家的 token 数量」
    #    长度为 num_experts 的数组被平均切成 world_size 段, 每段 num_local_experts 个,
    #    分别发给对应 rank。所以本 rank 收到的 recv_expert_counts 排布为:
    #       [来自 rank0 的本地专家计数..., 来自 rank1 的..., ...]
    #    这告诉我「我的每个本地专家, 分别从哪个源 rank 收到了多少 token」, 第 7 步要用。
    recv_expert_counts = torch.empty_like(expert_counts_tensor)
    dist.all_to_all_single(recv_expert_counts, expert_counts_tensor)

    # 6️⃣ 第三次 All-to-All: 真正搬运 token 数据本身
    #    用第 3 步对好的数量 (send_splits / recv_splits) 做「不等长」的 all_to_all。
    send_x = torch.stack(send_order, dim=0) if send_order else torch.empty((0, x.shape[1]), dtype=x.dtype, device=x.device)
    total_recv = int(recv_splits.sum().item())          # 本 rank 一共会收到多少 token
    recv_x_raw = torch.empty((total_recv, x.shape[1]), dtype=x.dtype, device=x.device)
    # 参数: (接收buffer, 发送buffer, 每段接收长度列表, 每段发送长度列表)
    dist.all_to_all_single(recv_x_raw, send_x, recv_splits.tolist(), send_splits.tolist())

    # 7️⃣ 把收到的「一长条」token 重新切回「按本地专家分组」
    #    recv_x_raw 里的 token 是按 (源rank, 本地专家) 顺序连续排布的;
    #    用 recv_expert_counts 当「分段尺子」, 一段一段切出来, 归到对应本地专家。
    local_expert_tokens = [[] for _ in range(num_local_experts)]
    ptr = 0                                              # 在 recv_x_raw 上滑动的读指针
    for i in range(recv_expert_counts.shape[0]):
        cnt = int(recv_expert_counts[i].item())         # 这一段有多少 token
        expert_id = i % num_local_experts               # 这一段属于哪个本地专家 (跨源 rank 会循环)
        local_expert_tokens[expert_id].append(recv_x_raw[ptr:ptr + cnt])
        ptr += cnt
    # 把每个本地专家的多段 (来自不同源 rank) 拼成一整块
    for i in range(len(local_expert_tokens)):
        local_expert_tokens[i] = torch.cat(local_expert_tokens[i], dim=0)

    # 8️⃣ 构造规整的输出张量 (固定形状, 方便后续专家计算与对比)
    #    recv_x 形状: (本地专家数, 最大可能 token 数, hidden)。多余位置补 0。
    max_len = num_tokens * world_size                   # 单个专家最多可能收到的 token 数上界
    recv_x = torch.zeros((num_local_experts, max_len, x.shape[1]), dtype=x.dtype, device=x.device)
    recv_count = torch.tensor([lst.shape[0] for lst in local_expert_tokens], dtype=torch.int32, device=x.device)

    for i, lst in enumerate(local_expert_tokens):
        if len(lst) > 0:
            recv_x[i, :len(lst)] = lst                  # 把有效 token 填进前 len(lst) 行
    # 返回: 本地专家 buffer、每专家计数、分发地图、各专家收到计数
    #   后两个 (expert_token_map / recv_expert_counts) 是 combine 阶段「原路返回」的关键。
    return recv_x, recv_count, expert_token_map, recv_expert_counts


# ============================================================================
# 对齐维度: 只比较「有效 token」, 检查 DeepEP 与 all2all 的 dispatch 是否一致
# ============================================================================
def compare_dispatch(deep_ep_recv_x, deep_ep_recv_count, all2all_recv_x, all2all_recv_count):
    # 仅在 rank==1 打印详细内容 (避免多进程刷屏)。`and 1` 是开发期的临时开关 (恒为真)。
    if dist.get_rank() == 1 and 1:
        print(f"deepep=== rank={dist.get_rank()} ===")
        recv_count = deep_ep_recv_count
        for exp_id in range(recv_count.numel()):                 # 遍历每个本地专家
            cnt = int(recv_count[exp_id].item())                 # 该专家收到的 token 数
            print(f"Expert {exp_id} (recv_count={cnt}):")
            for tok_id in range(cnt):                            # 逐 token 打印两条路径的内容
                row = deep_ep_recv_x[exp_id, tok_id]
                print(f"deepep  token {tok_id:02d}: {row}")
                row = all2all_recv_x[exp_id, tok_id]
                print(f"all2all token {tok_id:02d}: {row}")

    # 逐专家计算两条路径的最大逐元素绝对误差 (理想情况应为 0 或极小)
    diff = 0.0
    for i in range(deep_ep_recv_x.shape[0]):
        cnt = int(deep_ep_recv_count[i].item())
        a = deep_ep_recv_x[i, :cnt]     # 只取有效 token (前 cnt 行)
        b = all2all_recv_x[i, :cnt]
        if a.shape != b.shape:           # 形状不一致 = 路由结果对不上, 报错并跳过
            print(f"Shape mismatch at expert {i}: deep_ep {a.shape}, all2all {b.shape}")
            continue
        print("ljl--shape--same!!!")     # 调试痕迹: 确认形状一致
        d = (a - b).abs().max().item()   # 该专家的最大绝对误差
        print(f"Expert {i} max diff: {d}", "  rank=", dist.get_rank())
        diff = max(diff, d)              # 取所有专家里的最大值作为总误差
    return diff


# ============================================================================
# =========== Expert Forward Functions (专家前向计算) ===========
# ============================================================================

def expert_forward_all2all(recv_x, recv_count, expert_fn):
    """
    对 all2all 分发结果做专家计算。
    expert_fn: (tokens: Tensor) -> Tensor   表示一个专家的前向逻辑。
    返回: (num_experts, max_len, hidden)     与输入同形状, 无效位置保持 0。
    """
    num_experts, max_len, hidden = recv_x.shape
    out = torch.zeros_like(recv_x)
    for i in range(num_experts):                 # 对每个本地专家
        cnt = int(recv_count[i].item())
        if cnt > 0:
            out[i, :cnt] = expert_fn(recv_x[i, :cnt])   # 只对有效 token 跑前向
    return out

def expert_forward_deepep(recv_x, recv_count, expert_fn):
    """
    对 deepep 分发结果做专家计算。逻辑与上面完全一致 ——
    因为只要 dispatch 对齐了, 后续专家计算就是同一套, 这样才能公平对比。
    返回: (num_experts, max_len, hidden)
    """
    num_experts, max_len, hidden = recv_x.shape
    out = torch.zeros_like(recv_x)
    for i in range(num_experts):
        cnt = int(recv_count[i].item())
        if cnt > 0:
            out[i, :cnt] = expert_fn(recv_x[i, :cnt])
    return out


# ============================================================================
# 手写 all2all 的 combine (聚合) —— DeepEP combine 的「白盒参考实现」
# ============================================================================
def all2all_combine(all2all_out, all2all_recv_count, recv_expert_counts, expert_token_map, topk_weights, num_tokens, num_experts, num_topk, world_size, rank):
    """
    combine 分两步:
      (A) 把专家「计算结果」通过 all2all 原路送回 token 最初所在的 rank;
      (B) 在本地, 按 expert_token_map (分发地图) 和 topk_weights (权重),
          对每个 token 的 k 份专家输出做「加权求和」, 还原出每个 token 的最终结果。
    它是 dispatch 的逆过程: dispatch 是「散」, combine 是「聚」。
    """
    device = all2all_out.device
    hidden = all2all_out.shape[2]
    num_local_experts = num_experts // world_size

    # 1. 取出本地每个专家 buffer 里「有效」的输出 (去掉补 0 的尾巴)
    send_tokens = []
    for local_exp in range(num_local_experts):
        cnt = int(all2all_recv_count[local_exp].item())
        send_tokens.append(all2all_out[local_exp, : cnt])

    # 按「源 rank」把本地专家的输出重新分组, 准备发回去。
    #   回忆: dispatch 时 recv_expert_counts 告诉我们「本地专家的 token 分别来自哪个源 rank、各多少」。
    #   现在反过来用同一份信息, 把结果切回去、发回对应的源 rank。
    send_ranks = [[] for _ in range(world_size)]   # send_ranks[r] = 要发回 rank r 的若干段 token
    index_flag = [0] * num_local_experts           # 每个本地专家 buffer 上的滑动读指针
    for i in range(len(recv_expert_counts)):
        local_experts = i % num_local_experts      # 这一段属于哪个本地专家
        rank_id = i // num_local_experts            # 这一段当初来自哪个源 rank (现在就发回它)
        begin_index = index_flag[local_experts]
        end_index = index_flag[local_experts] + recv_expert_counts[i]
        send_ranks[rank_id].append(send_tokens[local_experts][begin_index : end_index])
        index_flag[local_experts] = index_flag[local_experts] + recv_expert_counts[i]

    # send_ranks: 每个目标 rank 一个 list, list 里是本地要发给该 rank 的 token tensor
    # 1. 统计每个 rank 要发多少 token
    send_counts = [sum([t.shape[0] for t in send_ranks[r]]) for r in range(world_size)]

    # 2. 把要发给各 rank 的 token 全部拼成一长条 (顺序: 目标 rank 升序)
    send_tokens_cat = torch.cat([t for sublist in send_ranks for t in sublist], dim=0) if any(send_ranks) else torch.empty((0, all2all_out.shape[2]), dtype=all2all_out.dtype, device=device)

    # 3. all_to_all 先对账「每个 rank 会收到多少 token」(同 dispatch 第 3 步的套路)
    send_counts_tensor = torch.tensor(send_counts, dtype=torch.int32, device=device)
    recv_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    total_recv = int(recv_counts_tensor.sum().item())

    # 4. all_to_all 真正把专家结果发回各源 rank
    recv_tokens_cat = torch.empty((total_recv, all2all_out.shape[2]), dtype=all2all_out.dtype, device=device)
    dist.all_to_all_single(recv_tokens_cat, send_tokens_cat, recv_counts_tensor.tolist(), send_counts_tensor.tolist())

    # 5. 本地加权聚合 (此时所有需要的数据都已回到本 rank, 不再通信)
    out = torch.zeros((num_tokens, all2all_out.shape[2]), dtype=all2all_out.dtype, device=device)

    # 5a. 先算出「本 rank 的每个专家, 一共发回来了多少 token」
    #     (即本 rank 的 token 里, 选中专家 e 的次数总和)
    expert_tokens_num = [0] * num_experts
    for t in range(len(expert_token_map)):
        topk = len(expert_token_map[t])
        for k in range(topk):
            expert_id = expert_token_map[t][k][0]
            expert_tokens_num[expert_id] += 1

    # 5b. 把收回来的一长条 recv_tokens_cat 按专家切成段, 方便按 (专家id, 序号) 索引。
    #     注意: recv_tokens_cat 的排布顺序与 dispatch 时 send_order 的「专家升序」一致,
    #     所以这里也按专家 id 升序切分, 两者就能对上。
    expert_tokens_tensor = []
    cnt = 0
    for i in range(num_experts):
        length = expert_tokens_num[i]
        expert_tokens_tensor.append(recv_tokens_cat[cnt : cnt + length][:])
        cnt = cnt + length

    # 5c. 对每个 token, 把它选中的 k 份专家结果按权重加起来 ——
    #     这正是 MoE 的核心公式:  y_t = Σ_k  w_{t,k} · Expert_{e_{t,k}}(x_t)
    for t in range(num_tokens):
        topk = len(expert_token_map[t])
        for k in range(topk):
            w = topk_weights[t, k].item()                   # 第 k 个专家的聚合权重
            expert_id, index = expert_token_map[t][k]       # 当初存的 (专家id, 1-based 序号)
            # index-1 把 1-based 序号转成 0-based 下标, 取出对应那一份专家输出
            out[t] += expert_tokens_tensor[expert_id][index - 1][:] * w
    return out


# ============================================================================
# =========== Dummy Expert Function (占位专家函数) ===========
# ============================================================================
def dummy_expert_fn(tokens):
    # 为了「只验证通信、屏蔽计算差异」, 这里用最简单的恒等变换 (×1) 当专家。
    # 真实场景下这里会是线性层 + 激活 (如 GLU/SwiGLU FFN) 等。
    # 注: 末尾注释写「乘2」但代码是「乘1」, 以代码为准 (恒等映射)。
    return tokens * 1  # 简单起见, 乘2


def compare_expert_forward(all2all_out, all2all_recv_count, deepep_out, deepep_recv_count):
    """
    比对 all2all 和 deepep 的专家前向输出是否一致 (逐专家取最大绝对误差)。
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


# ============================================================================
# 主脚本: 完整跑一遍 dispatch → forward → combine, 并三处对比两条路径
# ============================================================================
if __name__ == '__main__':
    import os
    # --- 分布式初始化 ---
    dist.init_process_group(backend='nccl')              # 用 NCCL 后端 (NVIDIA GPU 集合通信)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))    # 本机内的 GPU 序号 (由 torchrun 注入)
    torch.cuda.set_device(local_rank)                    # 绑定本进程到对应 GPU
    rank = dist.get_rank()                               # 全局 rank
    world_size = dist.get_world_size()                   # 总进程/GPU 数

    # --- 随机种子: 每个 rank 用不同种子, 模拟「不同 GPU 上的不同输入」---
    seed = 0
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    random.seed(seed + rank)

    # --- 问题规模 (取自 DeepSeek 风格的配置) ---
    num_tokens = 256        # 本 rank 的 token 数
    hidden = 7168           # 每个 token 的特征维度 (hidden size)
    num_experts = 8         # 专家总数
    num_topk = 2            # 每个 token 选 2 个专家

    # --- 造输入数据 ---
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')        # 随机 token 特征
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device='cuda')   # 随机路由 (每 token 选 2 个专家)

    # --- 造聚合权重 ---
    # 这里用「均匀权重」(每个被选专家权重都是 1/k), 便于验证。
    topk_weights = torch.full((num_tokens, num_topk), 1.0 / num_topk,
                              dtype=torch.float32, device=topk_idx.device)
    # 下面是「随机归一化权重」的备选方案 (更接近真实 router 输出), 当前注释掉:
    # topk_weights = torch.rand_like(topk_idx, dtype=torch.float32, device=topk_idx.device)
    # topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)  # 归一化

    # --- 创建 DeepEP 通信缓冲区 ---
    group = dist.group.WORLD
    # 先问 DeepEP「这种规模下需要多大的 RDMA 缓冲区」, 再据此分配。
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, world_size, num_experts)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // world_size)   # 每个本地专家一个 QP (RDMA 队列对)

    # ===================== 路径 A: DeepEP =====================
    # A-1 分发
    deep_ep_recv_x, deep_ep_recv_count, deep_ep_handle = dispatch_only_deep_ep(buffer, x, topk_idx, num_tokens, num_experts)
    # A-2 专家前向
    deepep_out = expert_forward_deepep(deep_ep_recv_x, deep_ep_recv_count, dummy_expert_fn)
    # A-3 聚合 (直接把 handle 传回去, DeepEP 自己按原路送回并加权)
    deepep_combined, *_ = buffer.low_latency_combine(
        deepep_out, topk_idx, topk_weights, deep_ep_handle
    )

    # ===================== 路径 B: 手写 All2All =====================
    # B-1 分发
    all2all_recv_x, all2all_recv_count, all2all_token_map, recv_expert_counts = dispatch_only_all2all(x, topk_idx, num_tokens, num_experts, world_size, rank)
    # B-2 专家前向
    all2all_out = expert_forward_all2all(all2all_recv_x, all2all_recv_count, dummy_expert_fn)
    # B-3 聚合 (手动把 token_map / weights / world_size / rank 都传进去)
    all2all_combined = all2all_combine(all2all_out, all2all_recv_count, recv_expert_counts,
        all2all_token_map, topk_weights, num_tokens, num_experts, num_topk, world_size, rank)

    # ===================== 三处对比 =====================
    # 注意: `rank == 0 or 1` 在 Python 里恒为真 (等价于 `(rank==0) or 1`),
    #       这是个常见的「想写每个 rank 都执行」的开发期写法。
    if rank == 0 or 1:
        # 对比 1: dispatch 结果是否一致
        diff = compare_dispatch(deep_ep_recv_x, deep_ep_recv_count,
                                all2all_recv_x, all2all_recv_count)
        print("dispatch max diff:", diff, "rank=", rank)

        # 对比 2: 专家前向输出是否一致
        expert_diff = compare_expert_forward(all2all_out, all2all_recv_count, deepep_out, deep_ep_recv_count)
        print("expert forward max diff:", expert_diff, "rank=", rank)

        # 对比 3: combine 最终输出是否一致 (这是端到端正确性的关键)
        if rank == 1 or 1:
            print("deepep_combined (rank=", rank, "):\n", deepep_combined)
            print("all2all_combined (rank=", rank, "):\n", all2all_combined)
            combine_diff = (all2all_combined - deepep_combined).abs().max().item()   # 绝对误差
            # 用相对误差更稳健 (bf16 本身有量化噪声, 不能要求绝对相等)
            deepep_norm = deepep_combined.abs().max().item() if deepep_combined.abs().max().item() != 0 else 1.0
            combine_rel_diff = combine_diff / deepep_norm
            print("combine max diff (abs):", combine_diff, "rank=", rank)
            print("combine max diff (rel):", combine_rel_diff, "rank=", rank)
            # 相对误差 ≤ 1% 才算通过 —— 两条路径数值等价 ⇒ 我们正确理解了 DeepEP。
            assert combine_rel_diff <= 1e-2, "error, test fail!"
