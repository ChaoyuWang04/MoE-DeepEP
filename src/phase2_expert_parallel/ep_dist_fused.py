"""
[Phase 2] ep_dist_fused.py — 完整专家并行层: 2次all2all + 融合Triton kernel(Phase1/2 合流)

做什么:
    最终形态的单层 MoE 专家并行: dispatch(2 all2all) -> expert_compute_fused(融合 Triton)
    -> combine。多组随机输入对拍"本地全局直算"基准，钉死正确性。这是 naive 计算的替换版，
    也是后续"通信-计算重叠"和"对标 DeepEP"的基线。

运行:
    SINGLE_GPU=1 torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.ep_dist_fused
"""
import os
import torch
import torch.distributed as dist
from ..phase1_single_gpu_moe.common_moe import ExpertWeights, single_expert_ffn
from .ep_dist_opt import dispatch_2a2a, combine_2a2a
from .expert_compute_fused import expert_compute_fused


def _ref_local(x_local, topk_idx_local, topk_weight_local, full, topk):
    """本地用全局专家直算基准。"""
    out = torch.zeros_like(x_local)
    for t in range(x_local.shape[0]):
        for j in range(topk):
            e = int(topk_idx_local[t, j].item())
            y = single_expert_ffn(x_local[t:t+1], full, e)
            out[t] += (y[0] * float(topk_weight_local[t, j].item())).to(out.dtype)
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

    # 用 Phase1 真实规模(H=2048, I=1408, E=64, top-6)，更贴近真模型
    E, H, I, topk = 64, 2048, 1408, 6
    E_local = E // world_size
    full = ExpertWeights.random(E, H, I, device=device, seed=123)
    weights_local = ExpertWeights(
        full.W_gate[rank*E_local:(rank+1)*E_local].contiguous(),
        full.W_up[rank*E_local:(rank+1)*E_local].contiguous(),
        full.W_down[rank*E_local:(rank+1)*E_local].contiguous())

    # 多组随机输入，钉死正确性(不同 token 数 + 不同种子)
    all_pass = True
    for trial, T_local in enumerate([32, 128, 333]):
        g = torch.Generator(device=device).manual_seed(rank * 100 + trial)
        x_local = (torch.randn(T_local, H, generator=g, device=device, dtype=torch.float32) * (H**-0.5)).to(torch.bfloat16)
        topk_idx_local = torch.randint(0, E, (T_local, topk), generator=g, device=device)
        topk_weight_local = torch.rand(T_local, topk, generator=g, device=device)
        topk_weight_local = topk_weight_local / topk_weight_local.sum(dim=1, keepdim=True)  # 非均匀权重,更严格

        recv_x, recv_local_expert, sp = dispatch_2a2a(x_local, topk_idx_local, E, world_size, rank, comm_on_cpu)
        expert_out = expert_compute_fused(recv_x, recv_local_expert, weights_local)
        out_local = combine_2a2a(expert_out, topk_weight_local, sp, world_size, rank, H, comm_on_cpu)

        ref = _ref_local(x_local, topk_idx_local, topk_weight_local, full, topk)
        d = (out_local.float() - ref.float()).abs().max().item()
        ok = d < 8e-2          # bf16 + 融合 kernel + 非均匀权重，容差稍宽
        all_pass = all_pass and ok
        print(f"[rank {rank}] trial{trial} T_local={T_local:>4} 融合EP对拍 max_abs_diff={d:.4e} {'PASS' if ok else 'FAIL'}")

    if rank == 0:
        print(f"[rank 0] === 完整专家并行层(2 all2all + 融合Triton kernel): "
              f"{'ALL PASS' if all_pass else 'SOME FAIL'} ===")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()