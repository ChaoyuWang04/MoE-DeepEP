"""
[Phase 0] capture_routing.py — 跑通 MoE 模型并抓取真实路由（monkeypatch 版）

做什么:
    加载 DeepSeek-V2-Lite，抓取每个 token 在每个 MoE 层选中的专家(topk_idx)与权重
    (topk_weight)，作为 Phase 2 all-to-all 的真实数据源。

    为什么不用 register_forward_hook（关键认知）:
        在 accelerate 的 CPU offload 下，模块调用被重新包装成直接走 forward，
        不经过 nn.Module.__call__ 的 hook 派发 —— 于是 forward(_pre)_hook 一次都不触发。
        鲁棒做法: 直接 monkeypatch gate 实例的 forward 方法本身（每条调用路径必经此处）。

    其它保留: 首次触发打印 gate 返回结构作证据；按 dtype 自动认 idx(整型)/weight(浮点)，
    不怕返回顺序与版本差异；抓到空则 raise，绝不静默存空。

输入:
    - 模型权重 (config.MODEL_DIR 优先；否则 config.MODEL_NAME)
    - PROMPTS: 几大段真实长文本
输出:
    - datas/routing_traces/deepseek_v2_lite.pt
      结构: {layer_idx: (topk_idx[T,k] long, topk_weight[T,k] float)}

运行:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      uv run python -m src.phase0_moe_literacy.capture_routing
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import config
from src.common.trace_io import save_trace

_traces = {}                         # layer_idx -> list[(idx_cpu, wgt_cpu)]
_fired = {"n": 0}                    # gate.forward 被调用的总次数
_structure_printed = {"done": False}


def _tensors_in(output):
    """从 gate 返回值里挑出所有 tensor，兼容 tuple/list/namedtuple/单 tensor。"""
    seq = output if isinstance(output, (tuple, list)) else [output]
    return [o for o in seq if torch.is_tensor(o)]


def _extract_idx_weight(output):
    """按 dtype 认: 整型=topk_idx, 浮点=topk_weight。不依赖返回顺序。"""
    tensors = _tensors_in(output)
    idx = next((t for t in tensors if not t.is_floating_point()), None)
    wgt = next((t for t in tensors if t.is_floating_point()), None)
    return idx, wgt


def _patch_gate_forward(gate, layer_idx):
    """monkeypatch 单个 gate 实例的 forward: 调原 forward -> 记录 -> 原样返回。

    关键: orig_forward 可能已是 accelerate 的包装(含设备对齐)，我们包在它外面，
    既不破坏 offload 的搬运逻辑，又能拿到真实输出。
    """
    orig_forward = gate.forward                      # 关键行: 可能是 accelerate 包装后的 forward

    def patched(*args, **kwargs):
        out = orig_forward(*args, **kwargs)          # 关键行: 先跑原逻辑(含 offload 搬运)
        _fired["n"] += 1
        if not _structure_printed["done"]:           # 首次打印真实结构作证据
            _structure_printed["done"] = True
            print(f"[capture][gate output] type={type(out)}")
            for j, o in enumerate(_tensors_in(out)):
                print(f"    tensor[{j}] shape={tuple(o.shape)} dtype={o.dtype} device={o.device}")
        idx, wgt = _extract_idx_weight(out)
        if idx is not None and wgt is not None:
            _traces.setdefault(layer_idx, []).append(
                (idx.detach().to("cpu"), wgt.detach().to("cpu")))
        return out

    gate.forward = patched                           # 关键行: 替换实例 forward，覆盖所有调用路径


PROMPTS = [
    "混合专家模型（Mixture of Experts, MoE）通过门控网络为每个 token 选择少数专家进行计算，"
    "从而在保持激活参数量较小的同时显著扩大总参数规模。它的核心挑战不在于专家本身的前向计算，"
    "而在于专家分散到多张 GPU 后，token 必须被路由到其选中专家所在的设备，这引入了昂贵的全对全通信。"
    "训练时通常会引入负载均衡的辅助损失，以缓解专家之间的负载倾斜，但这种均衡只在整个数据分布上成立。"
    "在实际推理时，针对某一类输入，少数专家可能被频繁激活，造成明显的热点。请详细解释这一现象的成因与影响。",

    "The mixture-of-experts architecture scales model capacity by activating only a small subset of "
    "expert feed-forward networks per token, chosen by a learned gating function. While this keeps the "
    "active parameter count low, it turns inference into a communication-bound problem once experts are "
    "sharded across many devices: every layer must dispatch tokens to the devices that hold their "
    "selected experts, run the expert computation, and combine the weighted results back. Explain in "
    "depth why general-purpose collective libraries are inefficient for this small-message all-to-all "
    "pattern, and what specialized kernels can do to reduce latency and overlap communication with compute.",

    "def fused_moe_forward(hidden_states, gate, experts, top_k):\n"
    "    logits = gate(hidden_states)\n"
    "    weights, idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)\n"
    "    out = torch.zeros_like(hidden_states)\n"
    "    for e in range(len(experts)):\n"
    "        mask = (idx == e)\n"
    "        tok, k = mask.nonzero(as_tuple=True)\n"
    "        if tok.numel() == 0:\n"
    "            continue\n"
    "        y = experts[e](hidden_states[tok])\n"
    "        out.index_add_(0, tok, y * weights[tok, k].unsqueeze(-1))\n"
    "    return out\n",
]


def main():
    src = str(config.MODEL_DIR) if config.MODEL_DIR.exists() else config.MODEL_NAME
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True,
    ).eval()

    n_patched = 0
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            _patch_gate_forward(layer.mlp.gate, i)   # 关键行: monkeypatch 而非 register_hook
            n_patched += 1
    print(f"[capture] 已 patch 的 MoE 层数: {n_patched}")
    print(f"[capture] n_routed_experts={model.config.n_routed_experts}, "
          f"top_k={model.config.num_experts_per_tok}")

    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").to(model.device)
            model(**ids)

    print(f"[capture] gate.forward 调用次数: {_fired['n']}")
    if _fired["n"] == 0:
        raise RuntimeError(
            "monkeypatch 后 gate.forward 仍未被调用 -> layer.mlp.gate 不是运行时对象，"
            "请把 print(model.model.layers[1].mlp) 的结构发我。")
    if len(_traces) == 0:
        raise RuntimeError(
            "gate.forward 调用了但没抓到 idx/weight -> 返回结构异常，把上面 [gate output] 结构发我。")

    merged = {}
    for layer_idx, items in _traces.items():
        idx = torch.cat([a for a, _ in items], dim=0)
        wgt = torch.cat([b for _, b in items], dim=0)
        merged[layer_idx] = (idx, wgt)

    path = save_trace(merged)
    s = next(iter(merged))
    print(f"[capture] 抓到 {len(merged)} 个 MoE 层；示例层 {s} topk_idx shape={tuple(merged[s][0].shape)}")
    print(f"[capture] 路由 trace 已保存: {path}")


if __name__ == "__main__":
    main()