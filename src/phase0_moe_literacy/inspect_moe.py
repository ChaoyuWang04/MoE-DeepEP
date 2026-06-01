"""
[Phase 0] inspect_moe.py — 一次性诊断: 看清本 modeling 版本在哪里产出路由

为什么:
    多次 monkeypatch gate.forward 都是 0 次调用（且已排除 offload），说明本版本
    不通过 gate.forward 暴露 topk 路由——很可能在 mlp.forward 里内联计算。
    与其继续盲打补丁，不如直接读源码，看 topk_idx 到底在哪产出，再一次写对 capture。

输出: 打印 MoE 层结构、gate 类型、mlp.forward 与 gate.forward 的源码
运行: uv run python -m src.phase0_moe_literacy.inspect_moe
"""
import inspect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.common import config


def main():
    src = str(config.MODEL_DIR) if config.MODEL_DIR.exists() else config.MODEL_NAME
    model = AutoModelForCausalLM.from_pretrained(
        src,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0},
    ).eval()

    mlp = model.model.layers[1].mlp          # 第 1 层通常是首个 MoE 层
    line = "=" * 64

    print(line); print("mlp 类型:", type(mlp).__name__)
    print("mlp 结构:\n", mlp)

    print(line); print("gate 类型:", type(mlp.gate).__name__)
    print("gate 公开属性:", [a for a in dir(mlp.gate) if not a.startswith("_")])

    print(line); print(">>> MoE 层 forward 源码:")
    try:
        print(inspect.getsource(type(mlp).forward))
    except Exception as e:
        print("取源码失败:", e)

    print(line); print(">>> gate forward 源码:")
    try:
        print(inspect.getsource(type(mlp.gate).forward))
    except Exception as e:
        print("取源码失败:", e)
    print(line)


if __name__ == "__main__":
    main()