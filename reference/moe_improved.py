"""
================================================================================
 MoE 专家并行 All-to-All —— 向量化 (Vectorized) 重构版  ·  初学者精读版
================================================================================

【先回忆: 这段代码在解决什么问题?】
  和 moe_origin.py 一样, 我们在做 MoE (混合专家) 的「专家并行」通信:
    - 每个 token 由 router 选中 top-k 个专家 (这里 topk 通常=2)。
    - 专家被切分到不同 GPU (rank) 上, token 选中的专家可能不在本地。
    - 于是需要三步: Dispatch(分发) → Expert Forward(专家计算) → Combine(聚合)。
  Dispatch 把 token 送到「专家所在的 GPU」; Combine 把结果原路送回并按权重求和。

【这份文件和 moe_origin.py 的区别 —— 也是本文件最值得学的地方】
  moe_origin.py 里的 dispatch_only_all2all / all2all_combine 用了大量
  **Python 逐 token、逐专家的 for 循环** + Python list 来分桶、记地图。
  代码好读, 但在 GPU 上极慢: Python 循环跑在 CPU 上, 每次 .item() 都要把
  GPU 数据同步回 CPU, 几百个 token 就有几百次同步, 完全发挥不出 GPU 的并行。

  这份 moe_improved.py 做的事一模一样, 但全部用「张量算子」一次性完成:
      bincount      —— 一次算出每个专家收到多少 token   (替代 for 累加计数)
      argsort       —— 一次得到「按专家排序」的索引       (替代 for 分桶)
      repeat_interleave —— 一次展开「每个槽位属于谁」的标签 (替代 for 复制)
      scatter (out[idx]=val) —— 一次把数据放回原位        (替代 for 逐个写回)
  这就是「AI Infra」最核心的思维方式之一:
      把 CPU 上的串行逻辑, 翻译成 GPU 上的并行张量操作 (vectorization)。

  另外它用了「类」来封装: dispatch 时把几个关键「重排索引」存进 self,
  combine 时再取出来做「逆操作」。因为 combine 必须精确撤销 dispatch 的每一次
  重排, 才能把结果送回正确的 token —— 索引就是这两步之间的「记账凭证」。

【全篇最关键、最容易错的一个概念: gather vs scatter (取 vs 放)】
  这两个是「互逆」的操作, 看着像但方向相反, 本文件 combine 阶段的两处
  「错误修正」注释就是在强调这件事:
      gather  (取):  b = a[idx]        含义: b[i] = a[idx[i]]   「按 idx 去 a 里取」
      scatter (放):  out[idx] = vals   含义: out[idx[i]] = vals[i] 「按 idx 把 vals 放回」
  如果 dispatch 用 `b = a[idx]` 把数据打乱了, 那么 combine 要还原, 就必须用
  `a_restored[idx] = b` (scatter), 而**不能**再写一次 `b[idx]` (那是再打乱一次)。
  初学时极易在这里写错, 记住一句话: 「正向用 gather 打乱, 逆向用 scatter 复位」。

【两层重排 (本文件出现两组 sorted 索引, 别混淆)】
  sorted_idx  : 把 (token × topk) 这条扁平队列, 按「目标专家 id」排序的索引。
                作用域 = 发送端本地的 token 选择。combine 第 3 步用它复位。
  sorted_eidx : 收到数据后, 把「按源 rank 排布(rank-major)」重排成
                「按本地专家排布(expert-major)」的索引。combine 第 1 步用它复位。

【数据在各阶段的「排布顺序」(layout) —— 看不见但必须心里有数】
  发送队列   : 按「全局专家 id 升序」排 (因此也就是按「目标 rank 升序」排)。
  刚收到时   : rank-major = [来自rank0的所有token | 来自rank1的所有token | ...]。
  重排之后   : expert-major = [本地专家0的所有token | 本地专家1的所有token | ...],
               这才是专家计算层想要的「同一个专家的 token 连在一起」。

【运行方式】
      torchrun --nproc_per_node=<GPU数> <你的主程序.py>
  (本文件只定义 AlltoAll 类, 通常被 moe_origin.py 那样的主脚本 import 使用。)
================================================================================
"""


class AlltoAll:
    """
    把 MoE 的 dispatch / combine 封装成一个「有状态」的通信器。
    为什么要有状态 (用类而不是两个独立函数)?
      —— combine 是 dispatch 的逆运算, 它必须复用 dispatch 期间产生的几个
         「重排索引」(sorted_idx / sorted_eidx) 和「通信量切分」(send/recv_split),
         才能把每个专家的输出精确送回它原本所属的 token。
         这些中间量存在 self 上, 就是 dispatch → combine 之间的「记账本」。
    """
    def __init__(self, num_experts, world_size, rank):
        self.num_experts = num_experts                      # 专家总数 (跨所有 GPU)
        self.world_size = world_size                        # GPU / 进程总数
        self.rank = rank                                    # 本进程的全局编号
        self.num_local_experts = num_experts // world_size  # 本 rank 上住着几个专家

        # 保存用于恢复顺序的关键索引 (dispatch 写入, combine 读取)
        self.sorted_idx = None        # [Step 2] 用于恢复 (Batch*TopK) 的原始顺序
        self.sorted_eidx = None       # [Step 4] 用于恢复 Rank-Major 的接收顺序
        self.send_split = None        # 本 rank 发给「每个 rank」各多少 token
        self.recv_split = None        # 本 rank 从「每个 rank」各收多少 token

    def dispatch(self, x, topk_idx):
        """
        把本 rank 的 token 分发到它们各自选中的专家所在的 GPU。
        参数:
          x        : 本 rank 的输入 token, 形状 [num_tokens, hidden_dim]。
          topk_idx : 每个 token 选中的专家 id, 形状 [num_tokens, topk]。
        返回:
          recv_x_permuted   : 收到并已重排成 expert-major 的 token, [total_recv, hidden]。
          local_expert_count: 本地每个专家分到多少 token, [num_local_experts],
                              供专家计算层据此切分这一长条数据。
        """
        # x: [num_tokens, hidden_dim]
        # topk_idx: [num_tokens, topk]
        num_tokens, hidden = x.shape
        _, topk = topk_idx.shape

        # --- Step 1: 计算并交换计数 ---
        # 先把 (num_tokens, topk) 的选择「拍平」成一条长度 num_tokens*topk 的队列。
        # 排布是行优先: [token0的k个专家, token1的k个专家, ...]。
        topk_idx_flat = topk_idx.view(-1)
        # 本地发出的每个 Global Expert 的计数
        # bincount(x, minlength=N): 统计 0..N-1 各值出现几次。
        #   这一行就替代了 origin 里「for 每个 token, for 每个 k, expert_counts[e]+=1」的整段循环。
        #   结果 num_tokens_per_expert[e] = 本 rank 有多少个 (token,k) 选择投向全局专家 e。
        num_tokens_per_expert = torch.bincount(topk_idx_flat, minlength=self.num_experts)
        # 接收到的每个 Global Expert (来自所有 Rank) 的计数
        # all_to_all_single 无 split 参数时 = 「等分交换」: 把长度 num_experts 的数组
        #   平均切成 world_size 段 (每段 num_local_experts 个), 第 i 段发给 rank i。
        #   第 i 段恰好是「rank i 拥有的那些专家」的计数 —— 即「我要发给 rank i 多少」。
        # 交换后, 本 rank 收到的 = 「别人要发给我的本地专家多少」。
        # 注意: 这里接收到的布局是 [Rank0_E0, Rank0_E1... | Rank1_E0...] (假设 num_tokens_per_expert 本身按专家ID排好序)
        #   读法: 来自 rank0 的 token 投向我的 le0/le1.../ 然后来自 rank1 的 .../...
        num_tokens_per_expert_group = torch.empty_like(num_tokens_per_expert)
        dist.all_to_all_single(num_tokens_per_expert_group, num_tokens_per_expert)

        # --- Step 2: 构建 Split 和 Buffer ---
        # 2.1 计算 Rank 级别的通信量 (把「按专家」的细计数聚合成「按 rank」的粗计数)
        # 假设 Expert ID 是连续的, 则 view(world_size, -1) 正确将 Expert 分组归属到 Rank
        #   view(world_size, num_local_experts) 后第 r 行 = rank r 拥有的专家, sum 即发给 rank r 的总量。
        send_count = num_tokens_per_expert.view(self.world_size, -1).sum(dim=1)       # 发给每个 rank 多少
        recv_count = num_tokens_per_expert_group.view(self.world_size, -1).sum(dim=1) # 从每个 rank 收多少
        self.send_split = send_count.tolist()   # 存起来; combine 阶段「反向发送」要用
        self.recv_split = recv_count.tolist()

        # 2.2 排序并构建发送 Buffer
        # 关键: 我们需要保存 sorted_idx (排序索引), 而不是排序后的值
        # argsort: 返回「能让 topk_idx_flat 变有序的下标序列」。
        #   即 topk_idx_flat[sorted_idx] 就是按专家 id 升序排好的。
        #   我们要的是这个「下标」本身, 因为 combine 要靠它把结果送回原始位置。
        #   这一行替代了 origin 里「for 专家 id, 把命中该专家的 token 依次 append 分桶」。
        sorted_idx = torch.argsort(topk_idx_flat)
        self.sorted_idx = sorted_idx  # <--- 保存这个用于 Combine 恢复

        # 扩展 x 到 (Batch * TopK), 然后按目标 Expert 排序
        # 优化: 只取需要的行, 不需要 repeat_interleave 整个 x 再索引
        # src_token_idx[j] = 扁平队列第 j 个槽位「来自哪个原始 token」。
        #   arange(num_tokens).repeat_interleave(topk) = [0,0,...,1,1,...] (每个 token 重复 topk 次)。
        src_token_idx = torch.arange(num_tokens, device=x.device).repeat_interleave(topk)
        # 按「专家升序」重排这份「来源 token」映射 —— 于是知道排序后每个槽位该取哪行 x。
        sorted_src_token_idx = src_token_idx[sorted_idx]
        # gather: 真正把 token 特征按「专家升序」取出来, 拼成发送缓冲区。
        #   send_buff 此刻的排布 = 按全局专家 id 升序 = 按目标 rank 升序, 正好对上 send_split。
        send_buff = x[sorted_src_token_idx]

        # 准备接收缓冲区: 总接收量 = 从各 rank 收到的数量之和。
        total_recv = self.recv_split_sum = sum(self.recv_split)
        recv_buff = torch.empty((total_recv, hidden), dtype=x.dtype, device=x.device)

        # --- Step 3: 交换 Token 数据 ---
        # 真正的「不等长」all_to_all: 按 send_split 发出、按 recv_split 收进。
        #   参数顺序: (接收buf, 发送buf, 每段接收长度, 每段发送长度)。
        dist.all_to_all_single(recv_buff, send_buff, self.recv_split, self.send_split)

        # --- Step 4: 本地重排 (Rank-Major -> Expert-Major) ---
        # 目前 recv_buff 是按发送 Rank 排序的。我们需要按 Local Expert ID 排序。
        #   收到时排布: [来自rank0的token... | 来自rank1的token... | ...] (rank-major)。
        #   但专家计算层需要「同一个本地专家的 token 全部连在一起」(expert-major)。
        # 生成对应的 Local Expert ID 标签
        # 这里的顺序对应 num_tokens_per_expert_group 的扁平化顺序
        # local_expert_ids_flat[s] = group 布局里第 s 个 (源rank, 本地专家) 槽位的「本地专家 id」。
        #   长度 world_size*num_local_experts, 内容 = [0,1,..,L-1, 0,1,..,L-1, ...] (每个源 rank 循环一遍)。
        local_expert_ids_flat = torch.arange(self.world_size * self.num_local_experts, device=x.device) % self.num_local_experts
        # 扩展出每个 Token 对应的 Local Expert ID
        # repeat_interleave: 用每个槽位的「token 数」把上面的标签按量展开,
        #   得到 recv_buff_eid[i] = 收到的第 i 个 token「属于哪个本地专家」(rank-major 顺序)。
        recv_buff_eid = torch.repeat_interleave(local_expert_ids_flat, num_tokens_per_expert_group)

        # argsort 按「本地专家 id」排序: 把分散在各源 rank 的同一专家 token 聚到一起。
        sorted_eidx = torch.argsort(recv_buff_eid)
        self.sorted_eidx = sorted_eidx   # <--- 保存这个用于 Combine 逆操作

        # gather: 一次性把 recv_buff 重排成 expert-major。combine 时要用 scatter 撤销它。
        recv_x_permuted = recv_buff[sorted_eidx]

        # 计算每个本地专家分到的数据量, 返回给计算层切分使用
        # num_tokens_per_expert_group 形状: [World_Size * Num_Local]
        # 我们需要算出 [Local_Expert_0_Total, Local_Expert_1_Total ...]
        #   view(world_size, num_local) 后「沿源 rank 维(dim=0)求和」= 跨所有源 rank 汇总每个本地专家的总量。
        local_expert_count = num_tokens_per_expert_group.view(self.world_size, -1).sum(dim=0)

        # 返回: expert-major 的 token 数据 + 每个本地专家的 token 数。
        #   注意没有显式返回「地图」, 因为复位所需的全部信息已存进 self (sorted_idx/sorted_eidx/splits)。
        return recv_x_permuted, local_expert_count

    def combine(self, expert_out, topk_weights):
        """
        把专家计算结果原路送回, 并按 router 权重做加权求和, 还原每个 token 的最终输出。
        它是 dispatch 的逆过程, 必须严格按相反顺序撤销 dispatch 做过的每一次重排:
            dispatch:  原始顺序 --gather(sorted_idx)--> 发送  --A2A-->  收到(rank-major) --gather(sorted_eidx)--> expert-major
            combine :  expert-major --scatter(sorted_eidx)--> rank-major --A2A(splits互换)--> 发送顺序 --scatter(sorted_idx)--> 原始顺序
        参数:
          expert_out  : 专家前向输出, [Total_Recv, Hidden], 排布与 dispatch 返回的一致 (expert-major)。
          topk_weights: 每个 (token, k) 选择的聚合权重, [num_tokens, topk]。
        返回:
          out: 每个 token 的最终输出, [num_tokens, hidden]。
        """
        # expert_out: [Total_Recv, Hidden] (Expert-Major)
        num_tokens, topk = topk_weights.shape

        # --- Step 1: 本地逆重排 (Expert-Major -> Rank-Major) ---
        # 撤销 dispatch Step 4 的 `recv_x_permuted = recv_buff[sorted_eidx]`。
        # 错误修正: 不能用 expert_out[self.sorted_eidx]
        #   —— 那是再做一次 gather(再打乱一次), 不是还原。
        # 必须用 Scatter 操作:  out[idx] = val
        #   含义: recv_buff_restored[sorted_eidx[i]] = expert_out[i],
        #   恰好把「expert-major 第 i 个」放回它原来的「rank-major 位置」。
        recv_buff_restored = torch.empty_like(expert_out)
        recv_buff_restored[self.sorted_eidx] = expert_out

        # --- Step 2: 数据发回 (All-to-All) ---
        # 这里的 send/recv 是相对于 Dispatch 的逆向
        # 现在要把结果发回「当初发给我们这些 token 的源 rank」, 所以收发的切分正好互换:
        #   发出量 = 当初的 recv_split, 收回量 = 当初的 send_split。
        send_buff = recv_buff_restored
        recv_buff = torch.empty((sum(self.send_split), expert_out.shape[1]), dtype=expert_out.dtype, device=expert_out.device)

        # 注意 splits 互换
        dist.all_to_all_single(recv_buff, send_buff, output_split_sizes=self.send_split, input_split_sizes=self.recv_split)

        # --- Step 3: 恢复原始序列 (Original Batch Order) ---
        # recv_buff 目前是按 (Target_Expert_ID) 排序的 (因为 dispatch 发送时排过序)
        # 我们需要把它放回原来的 (Batch, TopK) 位置
        #   即撤销 dispatch Step 2 的 `send_buff = x[sorted_src_token_idx]` 所隐含的 sorted_idx 重排。

        # 创建一个能容纳所有 results 的 buffer (Flat)
        #   长度 = num_tokens*topk: 每个 token 的 k 份专家结果先各自归位, 第 4 步再加权合并。
        out_flat = torch.empty((num_tokens * topk, expert_out.shape[1]), dtype=expert_out.dtype, device=expert_out.device)

        # 错误修正: 不能用 recv_buff[self.sorted_idx]
        #   —— 同样的坑: 那是 gather(再打乱), 不是复位。
        # 必须用 Scatter:  out[idx] = val
        #   含义: out_flat[sorted_idx[i]] = recv_buff[i], 把「按专家排序的发送顺序」放回「原始扁平顺序」。
        out_flat[self.sorted_idx] = recv_buff

        # --- Step 4: 加权求和 ---
        # 把扁平队列还原成 (num_tokens, topk, hidden), 每行是某 token 的 k 份专家输出。
        out_reshaped = out_flat.view(num_tokens, topk, -1)
        # MoE 核心公式: y_t = Σ_k  w_{t,k} · Expert_{e_{t,k}}(x_t)
        #   topk_weights.unsqueeze(-1): [num_tokens,topk,1] 便于和 hidden 维广播相乘, 再沿 k 维求和。
        out = (out_reshaped * topk_weights.unsqueeze(-1)).sum(dim=1)

        return out
