"""
[Phase 2] ep_reference.py — 专家并行的"参考语义"(单进程模拟，定义正确性基准)

做什么:
    用最直白的单进程逻辑，模拟"专家分散在 world_size 张卡"时一层 MoE 的完整数据流:
      dispatch  : 每个 token 的每个 topk 选择，被送到其专家所在的 rank
      expert    : 各 rank 用本地专家对收到的 token 做 FFN
      combine   : 结果送回原 token 所在 rank，按 topk_weight 加权求和
    这里【不做任何真实通信】，纯在单卡上用索引模拟搬运 —— 目的是定义"正确结果长什么样"，
    作为后续 单卡模拟版 / 真NCCL版 / DeepEP版 的对拍基准。

    与 Phase 1 的衔接:
      Phase1 单卡 = 按专家排序→算→散射回原位。
      Phase2 = 同样的事，但"排序"→"寄到专家所在rank"(dispatch)，"散射"→"寄回原rank"(combine)。
      本文件把这件事拆成 rank 视角，让通信结构显式化。

输入:
    x          : (T, H)            全局所有 token(模拟里所有 rank 的 token 拼在一起看)
    topk_idx   : (T, k)            每 token 选中的全局专家 id
    topk_weight: (T, k)
    weights    : ExpertWeights     全局 E 个专家
    world_size : int               模拟几张卡
输出:
    out        : (T, H)            加权合成结果(应与"不分卡、直接算"完全一致)

注: 专家到 rank 的映射用最简单的连续切分: rank r 持有专家 [r*E_local, (r+1)*E_local)。
"""
import torch
from ..phase1_single_gpu_moe.common_moe import ExpertWeights, single_expert_ffn


def expert_to_rank(expert_id, num_local_experts):
    """专家 id -> 它所在的 rank。连续切分映射。"""
    return expert_id // num_local_experts


def ep_reference_forward(x, topk_idx, topk_weight, weights: ExpertWeights, world_size):
    """单进程参考语义。逐 (token, k) 处理，显式走 dispatch→expert→combine 三段。"""
    T, H = x.shape
    k = topk_idx.shape[1]
    E = weights.num_experts
    assert E % world_size == 0, "专家数需能被卡数整除"
    E_local = E // world_size
    device = x.device

    out = torch.zeros_like(x)

    # ---- 模拟: 每个 rank 维护一个"收件箱"(收到的 token 行 + 它们来自哪个 token + 哪个专家) ----
    # dispatch: 把每个 (token,k) 选择按专家送到对应 rank 的收件箱
    inbox = {r: {"x": [], "src_token": [], "expert": [], "weight": []} for r in range(world_size)}
    for t in range(T):
        for j in range(k):
            e = int(topk_idx[t, j].item())
            if e < 0:
                continue
            r = expert_to_rank(e, E_local)
            inbox[r]["x"].append(x[t])
            inbox[r]["src_token"].append(t)
            inbox[r]["expert"].append(e)
            inbox[r]["weight"].append(float(topk_weight[t, j].item()))

    # ---- expert: 每个 rank 用本地专家算收到的 token ----
    # ---- combine: 算完按 weight 加权，送回原 token 所在位置累加 ----
    for r in range(world_size):
        box = inbox[r]
        if len(box["x"]) == 0:
            continue
        xr = torch.stack(box["x"], dim=0)                  # (n_r, H) 该 rank 收到的所有 token
        for i in range(xr.shape[0]):
            e = box["expert"][i]
            y = single_expert_ffn(xr[i:i+1], weights, e)   # 用专家 e 算(语义基准，逐个算)
            t = box["src_token"][i]
            out[t] += (y[0] * box["weight"][i]).to(out.dtype)  # combine: 加权回原 token
    return out


def dense_reference_forward(x, topk_idx, topk_weight, weights: ExpertWeights):
    """最朴素的"不分卡"基准: 直接逐 token 算其 topk 专家并加权。与 ep_reference 应一致。"""
    T, H = x.shape
    k = topk_idx.shape[1]
    out = torch.zeros_like(x)
    for t in range(T):
        for j in range(k):
            e = int(topk_idx[t, j].item())
            if e < 0:
                continue
            y = single_expert_ffn(x[t:t+1], weights, e)
            out[t] += (y[0] * float(topk_weight[t, j].item())).to(out.dtype)
    return out