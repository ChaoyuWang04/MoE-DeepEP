"""
[Phase 2] deepep_baseline.py — 真 DeepEP 对标封装（仅 Hopper 云端可用）

做什么:
    云端 H100 上调用真正的 deep_ep.Buffer.low_latency_dispatch / combine，
    作为我们手写实现的"标准答案 + 性能上界"。用同一份真实路由 trace 喂三方:
    naive / ours(optimized+overlap) / DeepEP，对比延迟、通信量、带宽。

前置: 仅在装好 deep_ep 的 Hopper 节点运行；5090 上 import 失败属正常。

输入: x, topk_idx, topk_weight, num_tokens, num_experts
输出: combine 后的 (T, H) + 计时
"""
try:
    import deep_ep                       # 仅 Hopper 云端可用
    _HAS_DEEPEP = True
except Exception:
    _HAS_DEEPEP = False


def run_deepep(*args, **kwargs):
    if not _HAS_DEEPEP:
        raise RuntimeError("当前环境无 deep_ep（5090 跑不了，需 Hopper 云端）")
    # TODO(Phase 2): 迁移老师代码的 buffer 构造 + low_latency_dispatch/combine + 计时
    raise NotImplementedError
