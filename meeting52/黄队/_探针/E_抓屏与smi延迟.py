# -*- coding: utf-8 -*-
"""复核探针 E：抓屏延迟（mss 全屏 1920×1080，项目实现路径）+ nvidia-smi 单次查询延迟。
只读：不写 PNG、不写文件；python -B 运行。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\E_抓屏与smi延迟.py
"""
import sys, time, statistics, subprocess
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def 统计(tag, xs):
    xs = sorted(xs)
    print("%-28s n=%d  median %.2f ms  min %.2f  p95 %.2f  max %.2f" %
          (tag, len(xs), statistics.median(xs), xs[0], xs[min(len(xs) - 1, int(len(xs) * .95))], xs[-1]))
    print("    全样本", [round(x, 2) for x in xs])


import mss  # noqa: E402

with mss.mss() as sct:
    mon = sct.monitors[1]
    print("显示器", mon, " 预期 RGB字节", mon["width"] * mon["height"] * 3,
          "= %.3f MiB / %.3f MB十进制" % (mon["width"] * mon["height"] * 3 / 1048576,
                                        mon["width"] * mon["height"] * 3 / 1e6))
    for _ in range(3):
        img = sct.grab(mon)
    ts, tc = [], []
    for _ in range(10):
        t = time.perf_counter()
        img = sct.grab(mon)
        ts.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        arr = np.asarray(img)[:, :, :3]
        tc.append((time.perf_counter() - t) * 1000)
    统计("mss.grab 全屏 1920x1080", ts)
    统计("np.asarray→RGB 转换", tc)
    print("帧形状", arr.shape, arr.dtype, "nbytes", arr.nbytes,
          "= %.3f MiB" % (arr.nbytes / 1048576), " raw img.rgb nbytes", len(img.rgb),
          "= %.3f MiB" % (len(img.rgb) / 1048576))

# nvidia-smi 单次查询
xs = []
for _ in range(10):
    t = time.perf_counter()
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                        "--format=csv,noheader"], capture_output=True, text=True)
    xs.append((time.perf_counter() - t) * 1000)
统计("nvidia-smi 单次查询(含进程启动)", xs)
print("末次输出", o.stdout.strip())
xs2 = []
for _ in range(10):
    t = time.perf_counter()
    o = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory",
                        "--format=csv,noheader"], capture_output=True, text=True)
    xs2.append((time.perf_counter() - t) * 1000)
统计("nvidia-smi compute-apps", xs2)
print("末次输出", repr(o.stdout.strip()))
