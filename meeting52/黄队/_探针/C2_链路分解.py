# -*- coding: utf-8 -*-
"""复核探针 C2：V4 贯通链的组件内存分解（D 值、桥/核数组、RSS）。
只读：不写文件；python -B 运行。不跑 120 拍，只构造后量。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\C2_链路分解.py
"""
import sys, importlib.util, json
import psutil, numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

路径 = r"<workspace>/03_AHI\AHI4\meeting49\实施\贯通_多核心.py"
spec = importlib.util.spec_from_file_location("gt", 路径)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = psutil.Process()
MI = 1048576.0


def rss(tag):
    x = p.memory_info()
    print("%-16s rss_MB %6.1f peak_wset_MB %6.1f" % (tag, x.rss / MI, x.peak_wset / MI))


def walk(obj, depth=0, seen=None, out=None, maxdepth=6):
    if seen is None:
        seen = set()
    if out is None:
        out = {}
    if depth > maxdepth or id(obj) in seen:
        return out
    if isinstance(obj, np.ndarray):
        seen.add(id(obj))
        out["array"] = out.get("array", 0) + int(obj.nbytes)
        return out
    if isinstance(obj, (str, bytes, int, float, bool, complex, type(None))):
        return out
    if isinstance(obj, dict):
        seen.add(id(obj))
        for k, v in obj.items():
            if isinstance(k, str):
                walk(v, depth + 1, seen, out, maxdepth)
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        seen.add(id(obj))
        for v in obj:
            walk(v, depth + 1, seen, out, maxdepth)
        return out
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        seen.add(id(obj))
        for k, v in vars(obj).items():
            walk(v, depth + 1, seen, out, maxdepth)
    return out


rss("base")
双 = m.造双区室()
rss("造双区室后")
核, 桥 = 双["核"], 双["桥"]
print("双[D] =", 双["D"], " 核.D =", 核.D, " 桥类型 =", type(桥).__name__)
try:
    print("桥.探针.shape =", getattr(桥, "探针", None).shape)
except Exception as e:
    print("桥.探针 读取失败", repr(e))
print("核类型 =", type(核).__name__, " 核.MRO =", [c.__name__ for c in type(核).__mro__][:5])

核账 = walk(核, maxdepth=3)
print("核数组合计 MiB =", round(核账.get("array", 0) / MI, 3))
桥账 = walk(桥, maxdepth=6)
print("桥数组合计 MiB =", round(桥账.get("array", 0) / MI, 3))
print("核+桥 数组合计 MiB =", round((核账.get("array", 0) + 桥账.get("array", 0)) / MI, 3))
# 两区室核单独构造（不含桥）
核2 = m.两区室核(D=双["D"], 种子=310, 模式="两区室", 慢率=0.20, 学模式="credit")
账2 = walk(核2, maxdepth=3)
print("两区室核(D=%d)数组合计 MiB = %.3f  顶层键数=%d" %
      (双["D"], 账2.get("array", 0) / MI, len(vars(核2))))
for name, v in vars(核2).items():
    if isinstance(v, np.ndarray) and v.nbytes > 65536:
        print("   ", name, v.shape, v.dtype, round(v.nbytes / MI, 3), "MiB")
rss("构造核2后")
