# 云端 (RunPod 2×H100) 跑 Phase 2 对标 — 操作清单

## 1. 起 Pod
- 选 2×H100 (SXM 优先，NVLink 带宽高；PCIe 也能跑)，镜像选 PyTorch CUDA 12.x。
- 确认 nvidia-smi 看到 2 张 H100。

## 2. 同步代码
```bash
git clone <你的repo>  # 或 scp 整个 moe-ep-lab
cd moe-ep-lab
pip install -r requirements.txt
# 或用 uv: uv pip install -r requirements.txt
```

## 3. 先验证多卡功能(真 NCCL，不加 SINGLE_GPU)
```bash
# 合流版正确性(真双卡)
torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.ep_dist_fused
# 异步重叠正确性
torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.ep_dist_overlap
```

## 4. 跑总对标(串行 vs 重叠 + 通信分解)
```bash
torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.bench_cloud --tokens 4096 --chunks 4
# 扫 chunk 数看重叠最优点
for c in 1 2 4 8; do
  torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.bench_cloud --tokens 4096 --chunks $c
done
```

## 5. nsys 看重叠时间线(确认通信与计算交叠)
```bash
nsys profile -o ep_overlap --trace=cuda,nvtx,nccl --force-overwrite true \
  torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.bench_cloud --profile --chunks 4
nsys stats --report nvtx_sum ep_overlap.nsys-rep
# 下载 ep_overlap.nsys-rep 到本地 GUI: 看 serial 区间是"通信条→计算条"串行,
# overlap 区间应是"通信条与计算条在时间上交叠"。
```

## 6. 安装并对标真 DeepEP (Hopper)
```bash
# DeepEP 依赖 NVSHMEM，按官方 README 装:
git clone https://github.com/deepseek-ai/DeepEP
cd DeepEP && pip install .   # 需先装 NVSHMEM，详见其文档
cd ..
# 装好后 bench_cloud 会自动跑 DeepEP 对标那段
torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.bench_cloud --tokens 4096
```

## 7. 关机省钱
- 跑完把 .nsys-rep / 控制台数字 下载到本地，立刻 Stop/Terminate Pod。
- 把数字填进 docs/01_why_deepep.md 的结论。

## 注意
- DeepEP 装 NVSHMEM 可能折腾，预留时间；装不上也不影响前 5 步(自研实现的全部数据)。
- ncu kernel 深挖【不在云端做】，本地 5090 即可。