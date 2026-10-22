# -*- coding: utf-8 -*-
"""meeting53 探针：只读 introspect 既有算子 API（不改任何既有件）。"""
import sys, inspect, json

BASE = r"<workspace>/03_AHI\AHI4"
sys.path.insert(0, BASE + r"\meeting49\实施")
sys.path.insert(0, BASE + r"\meeting50\实施")
sys.path.insert(0, BASE + r"\meeting38\实施")

import importlib

for mod in ["算子_两区室核", "算子_规则表征"]:
    try:
        m = importlib.import_module(mod)
        print("==", mod, "==")
        for n, o in vars(m).items():
            if inspect.isclass(o) and getattr(o, "__module__", "") == m.__name__:
                print("class", n, "bases", [b.__name__ for b in o.__bases__])
                for fn in ("__init__", "前向", "前向_双区室", "前向_规则", "学自我", "一拍", "步进"):
                    if hasattr(o, fn):
                        try:
                            print("  ", fn, str(inspect.signature(getattr(o, fn))))
                        except Exception as e:
                            print("  ", fn, "sig?", e)
    except Exception as e:
        print("!!", mod, repr(e)[:300])

# 能力核本体
try:
    import 算子_能力核 as 核m
    print("== 算子_能力核 ==")
    for n, o in vars(核m).items():
        if inspect.isclass(o) and getattr(o, "__module__", "") == 核m.__name__:
            print("class", n)
            for fn in ("__init__", "前向", "一拍"):
                if hasattr(o, fn):
                    try:
                        print("  ", fn, str(inspect.signature(getattr(o, fn))))
                    except Exception as e:
                        print("  ", fn, "sig?", e)
    for fn in ("绑", "解绑", "门", "叠", "归", "读", "迭代", "符"):
        if hasattr(核m, fn):
            print("op", fn, str(inspect.signature(getattr(核m, fn))))
except Exception as e:
    print("!! 算子_能力核", repr(e)[:300])
