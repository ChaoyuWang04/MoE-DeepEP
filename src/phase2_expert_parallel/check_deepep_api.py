"""
[Phase 2] check_deepep_api.py — 核对本机 DeepEP 的真实接口(跑对标前先确认，省机时)

做什么:
    打印已装 deep_ep 的版本、Buffer 的方法签名、low_latency_dispatch/combine 的参数，
    确认我们 deepep_baseline.py 的调用方式与实际 API 一致。各版本 DeepEP 接口有差异，
    盲跑容易报参数错，先核对。

运行:
    python -m src.phase2_expert_parallel.check_deepep_api
"""
import inspect


def main():
    import deep_ep
    print("deep_ep 模块路径:", deep_ep.__file__)
    print("deep_ep 顶层属性:", [a for a in dir(deep_ep) if not a.startswith("_")])

    Buffer = deep_ep.Buffer
    print("\n=== Buffer 方法 ===")
    methods = [m for m in dir(Buffer) if not m.startswith("_")]
    print(methods)

    for name in ["get_low_latency_rdma_size_hint", "low_latency_dispatch",
                 "low_latency_combine", "dispatch", "combine", "__init__"]:
        fn = getattr(Buffer, name, None)
        if fn is None:
            print(f"\n[{name}] 不存在")
            continue
        try:
            sig = inspect.signature(fn)
            print(f"\n[{name}] 签名: {sig}")
        except (ValueError, TypeError) as e:
            print(f"\n[{name}] 无法取签名({e})，打印 doc:")
            print((getattr(fn, "__doc__", "") or "")[:500])


if __name__ == "__main__":
    main()