"""
[common] comm_utils.py — torch.distributed 初始化与变长 all-to-all 工具

做什么:
    封装进程组初始化/销毁，以及"先报数再发数据"的变长 all-to-all 模板，
    供 phase2 的 dispatch/combine 复用。

输入: 环境变量 RANK / WORLD_SIZE / LOCAL_RANK（torchrun 注入）
输出: 已初始化的 process group；变长 all2all 封装函数
运行: 由 torchrun --nproc_per_node=N 启动；本文件不单独运行。
"""
import os
import torch
import torch.distributed as dist


def init_dist():
    """初始化 NCCL 进程组并绑定本地 GPU。返回 (rank, world_size, local_rank)。"""
    dist.init_process_group(backend="nccl")          # 关键行: 建通信后端
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)                # 关键行: 每进程绑一张卡
    return dist.get_rank(), dist.get_world_size(), local_rank


def cleanup_dist():
    """销毁进程组，释放通信资源。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def all2all_var(send_tensor, send_splits, recv_splits):
    """变长 all-to-all: 按 splits 把 send_tensor 切块发往各 rank。

    输入:
        send_tensor : (sum(send_splits), H) 待发送数据
        send_splits : list[int] 发给每个 rank 的行数
        recv_splits : list[int] 从每个 rank 接收的行数（须先经一次 all2all 换得）
    输出:
        recv_tensor : (sum(recv_splits), H) 接收到的数据
    """
    H = send_tensor.shape[1]
    recv_tensor = torch.empty(
        (sum(recv_splits), H), dtype=send_tensor.dtype, device=send_tensor.device
    )
    # 关键行: 接收 buffer 必须先按 recv_splits 开好——这正是"先报数"的根本原因
    dist.all_to_all_single(recv_tensor, send_tensor, recv_splits, send_splits)
    return recv_tensor
