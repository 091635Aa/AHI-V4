# -*- coding: utf-8 -*-
"""复核探针 E2：连续采集（桌面复制 DDA）路径延迟 —— 眼现行后端；与 mss 降级路径对照。
只读：不写文件；python -B 运行。注意：连续采集须在 import torch 之前建相机（本探针不导 torch）。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\E2_连续采集.py
"""
import sys, time, statistics
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting38\实施")
try:
    import dxcam
    print("dxcam 可用:", getattr(dxcam, "__version__", "?"))
except Exception as e:
    print("dxcam 不可用:", repr(e))
    raise SystemExit

import 连续采集 as C  # noqa: E402

t0 = time.perf_counter()
ok = C.预热()
print("预热() =", ok, "耗时 %.1f ms" % ((time.perf_counter() - t0) * 1000))
流 = C.连续采集()
t0 = time.perf_counter()
print("起() =", 流.起(), "耗时 %.1f ms" % ((time.perf_counter() - t0) * 1000))
print("状态", 流.状态())
# 等新帧
t0 = time.perf_counter()
while 流.状态().get("新帧数", 0) < 5 and time.perf_counter() - t0 < 3:
    time.sleep(0.005)
print("状态(5帧后)", 流.状态())
f, tf, 活 = 流.取()
print("取()首帧 shape", None if f is None else f.shape, "活", 活, "帧龄s", round(time.time() - tf, 3))
# 取帧成本（最新帧槽读取）
xs = []
for _ in range(30):
    t = time.perf_counter()
    f, tf, 活 = 流.取()
    xs.append((time.perf_counter() - t) * 1e6)
xs.sort()
print("取() 30 次  median %.1f us  p95 %.1f us  max %.1f us" %
      (statistics.median(xs), xs[int(len(xs) * .95) - 1], xs[-1]))
# 新帧处理速率：静屏下 30fps 上限；统计 1s 内新帧数
n0 = 流.状态().get("新帧数", 0)
t0 = time.perf_counter()
time.sleep(1.0)
n1 = 流.状态().get("新帧数", 0)
print("1s 内新帧数 %d（静屏受 30fps 上限）" % (n1 - n0))
print("状态(结束)", 流.状态())
流.停()
