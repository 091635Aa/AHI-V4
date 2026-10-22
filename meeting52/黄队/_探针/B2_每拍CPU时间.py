# -*- coding: utf-8 -*-
"""复核探针 B2：每拍「墙钟 vs 本进程 CPU 时间」——区分真实计算成本与并发抢占。
只读：不写文件；python -B 运行。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\B2_每拍CPU时间.py
"""
import sys, time, gc, statistics
import psutil

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting51\实施")
import 算子_千层骨 as q  # noqa: E402

print("起始 CPU%%(0.5s)", psutil.cpu_percent(interval=0.5), "空闲GB",
      round(psutil.virtual_memory().available / 1073741824, 2))
for D in (64, 256):
    k = q.千层核(D=D, 种子=7)
    for _ in range(5):
        k.前向一拍(None)
    结果 = {}
    for H in (1, 3):
        k.开关["环路"] = H
        for _ in range(2):
            k.前向一拍(None)
        gc.collect()
        gc.disable()
        w, c = [], []
        for _ in range(10):
            t0w, t0c = time.perf_counter(), time.process_time()
            for _ in range(10):
                k.前向一拍(None)
            w.append((time.perf_counter() - t0w) / 10 * 1000)
            c.append((time.process_time() - t0c) / 10 * 1000)
        gc.enable()
        wm, cm = statistics.median(w), statistics.median(c)
        结果[H] = (wm, cm, min(w), min(c))
        print("D=%d H=%d  墙钟 median %.2f ms (min %.2f)  CPU时间 median %.2f ms (min %.2f)  墙/CPU %.2f" %
              (D, H, wm, min(w), cm, min(c), wm / cm))
    wm1, cm1, _, _ = 结果[1]
    wm3, cm3, _, _ = 结果[3]
    print("  D=%d  H3/H1: 墙钟 %.2f  CPU %.2f   | 两点解(CPU时间) F=%.2f R=%.2f" %
          (D, wm3 / wm1, cm3 / cm1, cm1 - (cm3 - cm1) / 2, (cm3 - cm1) / 2))
    del k
print("结束 CPU%%", psutil.cpu_percent(interval=None))
