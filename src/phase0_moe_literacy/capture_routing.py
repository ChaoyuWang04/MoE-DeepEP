"""
[Phase 0] capture_routing.py — 跑通 MoE 模型并抓取真实路由（最终版）

做什么:
    抓取每个 token 在每个 MoE 层选中的专家(topk_idx)与权重(topk_weight)。

    定位经过（踩坑总结，已写进 design_notes）:
        本版 HF modeling (DeepseekV2Moe) 的 routing 不在 gate.forward 里产出——
        gate 只是普通 Linear(存权重)，forward 用 F.linear(hidden, gate.weight) 内联算 logits，
        再交给 self.route_tokens_to_experts(router_logits) 返回 (topk_indices, topk_weights)。
        所以正确拦截点是 route_tokens_to_experts，而非 gate.forward / forward hook。
        教训: 工具失效时先 inspect.getsource 看实现，再决定拦哪里，别对黑盒连打补丁。

输入:  模型权重 + PROMPTS
输出:  datas/routing_traces/deepseek_v2_lite.pt  结构 {layer_idx: (idx[T,k] long, wgt[T,k] float)}
运行:  uv run python -m src.phase0_moe_literacy.capture_routing
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import config
from src.common.trace_io import save_trace

# "gpu8bit": 8-bit 全上 GPU(~16GB)，快，路由有极小量化扰动但不改变负载分布结论。【默认】
# "gpu4bit": 4-bit(~9GB)，更省显存。
# "cpu":     纯 bf16 CPU 前向，路由 bit-exact，慢，需 ~36GB 内存。【要绝对精确路由时用】
LOAD_MODE = "gpu8bit"

_traces = {}
_fired = {"n": 0}
_structure_printed = {"done": False}


def _tensors_in(output):
    seq = output if isinstance(output, (tuple, list)) else [output]
    return [o for o in seq if torch.is_tensor(o)]


def _extract_idx_weight(output):
    """按 dtype 认: 整型=topk_idx, 浮点=topk_weight。route_tokens_to_experts 返回 (idx, wgt)。"""
    tensors = _tensors_in(output)
    idx = next((t for t in tensors if not t.is_floating_point()), None)
    wgt = next((t for t in tensors if t.is_floating_point()), None)
    return idx, wgt


def _patch_router(mlp, layer_idx):
    """monkeypatch mlp.route_tokens_to_experts: 调原方法 -> 记录 (idx, wgt) -> 原样返回。"""
    orig = mlp.route_tokens_to_experts                # 关键行: 真正产出路由的方法

    def patched(router_logits, *args, **kwargs):
        out = orig(router_logits, *args, **kwargs)    # out = (topk_indices, topk_weights)
        _fired["n"] += 1
        if not _structure_printed["done"]:
            _structure_printed["done"] = True
            print(f"[capture][route output] type={type(out)}")
            for j, o in enumerate(_tensors_in(out)):
                print(f"    tensor[{j}] shape={tuple(o.shape)} dtype={o.dtype} device={o.device}")
        idx, wgt = _extract_idx_weight(out)
        if idx is not None and wgt is not None:
            _traces.setdefault(layer_idx, []).append(
                (idx.detach().to("cpu"), wgt.detach().to("cpu")))
        return out

    mlp.route_tokens_to_experts = patched


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


def _load_model(src):
    if LOAD_MODE == "cpu":
        return AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).eval()
    from transformers import BitsAndBytesConfig
    if LOAD_MODE == "gpu8bit":
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
    elif LOAD_MODE == "gpu4bit":
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        raise ValueError(f"未知 LOAD_MODE: {LOAD_MODE}")
    return AutoModelForCausalLM.from_pretrained(
        src, quantization_config=qcfg, device_map={"": 0},
    ).eval()


def main():
    src = str(config.MODEL_DIR) if config.MODEL_DIR.exists() else config.MODEL_NAME
    tok = AutoTokenizer.from_pretrained(src)
    print(f"[capture] LOAD_MODE={LOAD_MODE}")
    model = _load_model(src)
    dev = "cpu" if LOAD_MODE == "cpu" else "cuda"

    n_patched = 0
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "route_tokens_to_experts"):   # 只有 MoE 层有此方法
            _patch_router(layer.mlp, i)
            n_patched += 1
    print(f"[capture] 已 patch 的 MoE 层数: {n_patched}")
    print(f"[capture] n_routed_experts={model.config.n_routed_experts}, "
          f"top_k={model.config.num_experts_per_tok}")

    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").to(dev)
            model(**ids)

    print(f"[capture] route_tokens_to_experts 调用次数: {_fired['n']}")
    if _fired["n"] == 0:
        raise RuntimeError("route_tokens_to_experts 未被调用，把 print(model.model.layers[1].mlp) 发我。")
    if len(_traces) == 0:
        raise RuntimeError("调用了但没抓到 idx/weight，把上面 [route output] 结构发我。")

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