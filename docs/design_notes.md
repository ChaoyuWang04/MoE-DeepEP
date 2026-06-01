# 设计笔记 / 优化轴（持续更新）

## all-to-all 优化的 5 个轴（Phase 2 逐项落地）
1. 通信轮数：dispatch 3 次 all2all -> 2 次。
   recv_splits（每 rank 总数）可由 recv_expert_counts（每专家数）按 rank 分组求和
   本地算出，无需单独再发一次。
       recv_splits = recv_expert_counts.view(world_size, E_local).sum(dim=1)
2. 通信量：dispatch 用 FP8；同一目标 rank 的重复 token 去重，只过一次网络。
3. 重叠：把 batch 切 chunk，多 CUDA stream 让"算上一块"与"传下一块"并行。
4. 执行效率：干掉逐元素 .item()（每次 = 一次 GPU->CPU 同步），路由用
   argsort/bincount/cumsum 向量化，替掉 Python for + append + stack。
5. 显存：packed 紧凑 buffer 取代 (E_local, T*world_size, H) 的最坏情况大 padding。

## Phase 0 观察（真实路由负载分布）
- 数据：DeepSeek-V2-Lite，26 个 MoE 层，64 路由专家，top-6；
  累计 ~11570 token（3 段窄领域文本：MoE 中文/英文/代码），平均每专家每层被选 ~41.7 次。
- 结果：极端且结构性倾斜，而非均匀。
    - 全局聚合 max/mean ≈ 10.6（最热专家是平均的 ~10 倍），CV ≈ 2.65。
    - 冷门专家占比 ~90%：64 个里约 58 个几乎不被选。
    - 0~4 号 5 个专家吃掉绝大多数 token；逐层 max/mean 稳定在 ~10.6（每层都倾斜，不是个别层）。
- 解读：
    - 训练的负载均衡 loss 只保证【训练全分布】上不塌缩，不保证【特定推理输入】下的局部均衡。
    - 窄领域输入会激活一组固定热点专家 -> 真实推理负载是结构性倾斜的。
    - 少数路由专家承担"准共享"角色（高频激活），其余做稀疏领域特化（DeepSeek 细粒度专家形态）。
- 对 Phase 2 的直接影响：
    - all-to-all 通信负载天然不均：热点专家所在 rank 收到海量 token，其余近乎闲置。
    - buffer 必须按最坏情况 (num_tokens*world_size) 预留 -> 冷门专家侧大量 padding 浪费
      -> 这正是"为什么 MoE 的 all-to-all 难做"的实证，也是 DeepEP 要解决的核心之一。
- epistemics（别过度解读 10.67 这个数）：
    混合了 (1) 真实结构性倾斜[主因] (2) 窄领域 3 段文本放大热点 (3) 8-bit 量化极小扰动。
    简历表述用："在特定领域输入下观察到约 10x 的专家负载倾斜"，勿当成模型固有常数。


## Phase 1 观察（单卡专家计算：三版对照 + profile）
- 实测(synthetic, tokens=4096, skew=1.5, E=64, top-6, H=2048, I=1408):
    naive 1.00x | torch_vec 1.32x | torch_bmm 0.12x(慢 8 倍)
- nsys 证据: 一次前向 cudaLaunchKernel ~6332 次；GPU 上真正干活的 GEMM 里出现 492 次
  32x32 极小 tile —— 冷门专家只有几行 token，GEMM 小到喂不饱 tensor core。
- 三版认知链(本 phase 核心):
    1. naive 真瓶颈 = 负载倾斜(Phase0 实测 ~10x)→ 大量 32x32 豆腐块小 GEMM → 算力利用率低。
       注意: 不是单纯"launch 计数"问题，而是"小到喂不饱 GPU"问题(profile 才看得清)。
    2. torch_vec 只向量化了【路由查找】(argsort/bincount/cumsum)，专家计算仍是逐专家
       for 循环小 GEMM —— 小 GEMM 没动，所以只小赢，且 speedup 随 batch 递减。
    3. torch_bmm 用 padding 到最长段 + 3 次大 bmm 消灭了小 GEMM，但 padding 浪费 ~87%
       (有效 24576 行被 padding 到 64*cap)，倾斜越重越灾难 → 慢 8 倍。
- 结论(逼出 Triton 的唯一理由):
    两条死路 —— "小 GEMM 喂不饱" vs "padding 算空气"。Triton grouped GEMM 要同时避开两者:
    把 64 段融成【一次 kernel 启动】(消灭小 GEMM)，又按【真实段长】各算各的(不 padding)。
    torch 算子做不到"变长+融合"，必须下沉到 kernel 自己按 offset 调度 —— 这是 Triton 不可替代之处。
- 方法论教训: 关于性能，先 profile 再下结论。本轮我(助手)两次口头预测均被数据推翻
    (先猜 speedup 递增、再猜 per-expert launch 计数)，真因是"小 GEMM 喂不饱 + padding 浪费"，
    只有 nsys + 对照实验(torch_bmm)才看得清。猜测便宜，profile 便宜，别用猜代替测。


## Phase 1 续：Triton grouped GEMM 的迭代与 profile 驱动优化（完整认知链）

### 六版性能对照（synthetic, E=64, top-6, H=2048, I=1408, skew=1.5）
| 版本 | 445 tok | 4096 tok | 16384 tok | 一句话 |
|------|---------|----------|-----------|--------|
| naive | 1.00x | 1.00x | 1.00x | 64次逐专家小GEMM，32x32豆腐块喂不饱 |
| torch_vec | 1.40x | 1.30x | 1.17x | 只向量化路由查找，专家GEMM仍逐专家launch |
| torch_bmm | 1.70x | 0.11x | OOM | padding到最长段，浪费~87%，大输入显存爆 |
| triton(v1) | 4.74x | 1.45x | 0.51x | grouped GEMM，但kernel内O(E)扫描+小BLOCK拖累 |
| triton(v2) | 4.91x | 1.58x | 0.89x | host预计算tile→expert，去O(E)扫描，最优BLOCK |
| fused | 2.22x | 2.22x | 1.19x | gate+up合并+SILU epilogue，大输入终于全面超naive |
| fused+graph | — | 2.14x | 1.15x | CUDA Graph，但此处无空泡可消，略亏（见下） |
注：speedup 比值受 naive 固定开销干扰，445 tok 时虚高；看绝对效率更可靠。

### 关键认知（按踩坑顺序）
1. naive 真瓶颈 = 负载倾斜(Phase0 ~10x) → 大量 32x32 小 GEMM 喂不饱 tensor core。
   nsys 实证: 一次前向 cudaLaunchKernel ~6332 次，GPU GEMM 里 492 次 32x32 tile。
2. torch_vec 只向量化【路由查找】，专家 GEMM 仍逐专家 launch；speedup 随 batch 递减
   说明固定开销非主因、per-expert GEMM 才是。
3. torch_bmm 用 padding 消灭小 GEMM，却引入"算空气"：浪费~87% + 大输入显存 OOM。
   倾斜越重 padding 越灾难。padding 方案双输(慢且耗显存)。
4. 用户独立提出"热合并/冷不padding"分治思路 → grouped GEMM 是其统一升华：
   按固定 BLOCK_M 行 tile 切(非按专家)，热点段多占 tile、冷门段占 1 个 tile，
   浪费从"对齐最长段(3000行)"降到"对齐 tile 边界(<64行)"，无需手动分热冷。
5. grouped GEMM 灵魂 = "切块单位是行tile，每个 tile 查 offsets 定专家、读对应权重"。
   v1 在 kernel 内用 for i in range(E) 线性扫描定专家 → tile 数∝token，扫描被海量重复，
   大输入下成主开销，且与段对齐耦合(BLOCK_M=256 时暴涨17x)。
   v2 修复: host 端用 searchsorted 预计算 tile_expert 数组，kernel O(1) 查表(vLLM 标准做法)。
   段对齐粒度固定 ALIGN=64 与 BLOCK_M 解耦，避免大 BLOCK 时冷门段 padding 暴增。
6. roofline 快判(实测 16384 单次gate): 算力 82.3% peak、带宽 18% → GEMM 本身高效，
   既非 memory-bound 也无空泡。"triton 慢"不在 GEMM 本身。
7. nsys 通查定位真凶: fused 前 triton 把一次 MoE 拆成几十个 kernel
   (sort/gather/silu独立3.5ms/index_add/copy/fill...)，kernel 间空泡 + 重复读 x_sorted
   累积 > GEMM 省下的时间。
8. 融合优化: gate+up 合一个 kernel(读一遍 x_sorted 同算两投影) + SILU 进 epilogue
   (寄存器里直接算，不再独立 kernel)。3次GEMM+1silu → 2次GEMM。
   金句: elementwise 单独成 kernel = 纯带宽浪费；塞进前一 compute kernel 的 epilogue，
   数据还在寄存器时顺手算完，省一次完整显存往返。
9. CUDA Graph 认知(教科书反例): graph 消除的是"kernel 间 CPU 调度空泡"，收益=空泡占比。
   融合后计算段只剩 2 个饱满大 kernel，本就几乎无空泡 → graph 无收益，还因 replay 的
   固定 buffer copy 略亏。graph 主场是"kernel 多而碎"(如未融合版/decode 阶段)。
   工程折中: sort(输出长度依赖数据)做不进 graph，只把固定形状计算段录入，sort 留图外。

### Profiler 分工（重要方法论，曾用错）
- nsys = 听诊器: 通查整个系统，找出哪个 kernel/阶段占大头、看空泡。先用它。
- ncu  = 显微镜: 只对 nsys 定位出的【单个】可疑 kernel 深挖(occupancy/tensor利用率/带宽)。
- 错误用法(曾犯): 用 ncu --set 对全部 100+ kernel 无差别 profile，既慢又无重点。
- roofline 粗算(实测时间反推 算力%/带宽%) 可在上 nsys 前先定性 compute/memory-bound。

### 如何读 nsys 时间线找空泡（学习笔记）
- 看 kernel 行是否"连续密实" vs "稀疏带缝"；每条缝=GPU 算完在等 CPU 发下一个 kernel=空泡。
- 看方块大小: cuLibraryLoadData 等是启动/加载开销(占时间但不算数)；一堆小蓝块=小算子喂不饱。
- 目标(老师原话): 让 kernel 行连成一片无缝 = GPU 永远有下一个 kernel 排队，CPU 喂得上。
- 本项目实例: 融合前 triton 时间线是小方块带缝(sort/gather/silu/index_add 各自成段)；
  融合后应是少数饱满大 GEMM 连成密实片 —— 这也解释了为何 graph 再加无收益。


### Phase 1 收官补充（对照实验 + 承上启下）
- CUDA Graph 对照实验结果: 碎kernel版 graph x1.04、大kernel版 x1.02 —— 两边都几乎无提升。
  修正认知(比"碎才有用"更准): graph 收益 = 空泡占总时间比例。本项目所有版本单 kernel
  都已几十微秒~毫秒级，CPU 调度的几微秒空泡占比极小 → graph 无感。
  graph 真正主场 = decode 阶段 batch=1、kernel 仅几微秒的极端碎场景(空泡能占一半)。
  prefill-like 大 batch MoE 不是它的菜。真正胜负手是【融合】(A逐专家2.25ms vs B融合0.92ms，2.4x)。
- fused 版 nsys: GPU 时间 67.6% 在两个大 kernel(fused_gate_up_silu 42% + down 25.6%)，
  独立 silu kernel 已消失(进 epilogue)。剩 ~32% 是 gather/index_add/copy = sort_and_align
  与散射回原 token 的数据搬运。
- 【承上启下关键】单卡上这些 gather/scatter 是纯本地开销；到 Phase 2 多卡，"按专家重排 +
  散射回原 token"将变成【跨卡 all-to-all 通信本身】。Phase 1 优化掉的数据搬运，Phase 2 会
  以"通信"形式重现，而那正是 DeepEP 要解决的核心。完美衔接。

## 踩坑记录（随做随记）
- token_map 存的是 append 后长度(1-indexed)，取值要 index-1（off-by-one 高发区）。
- 抓 MoE 路由的拦截点：本版 HF DeepseekV2Moe 的 routing 不在 gate.forward（gate 只是
  存权重的 Linear，forward 用 F.linear(hidden, gate.weight) 内联算 logits），真正产出
  topk 的是 mlp.route_tokens_to_experts(router_logits) -> (topk_idx, topk_weight)。
  正确做法是 monkeypatch 该方法。
- register_forward_hook / 实例 monkeypatch gate.forward 在 accelerate CPU offload 下
  不触发（调用被重包装绕过 __call__）。但本案最终根因不是 offload，而是拦错了对象——
  教训：工具失效时先 inspect.getsource 看实现，定位真正产出点，再决定拦哪里，
  不要对黑盒连续打补丁（我们这次连打 3 次补丁才回头看源码，应更早 inspect）。
- DeepSeek-V2 新版 HF 建模已内置，加载【不要】传 trust_remote_code=True，否则触发兼容问题。
- Phase 0 加载策略：避免 device_map="auto" 的 CPU/meta offload（会让中间量抓取变复杂）；
  改用 8-bit 量化 device_map={"":0} 全上 GPU(~16GB)，gate 不量化照常调用。
  要 bit-exact 路由时用纯 CPU bf16（慢，需 ~36GB 内存）。