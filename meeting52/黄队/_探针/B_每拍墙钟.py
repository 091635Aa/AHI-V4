# -*- coding: utf-8 -*-
"""复核探针 B：每拍墙钟（千层核 D=64/256，H=1/3），按黄队 R1 §12.2 协议复跑 + 重复。
只读：不写任何文件；python -B 运行。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\B_每拍墙钟.py [重复轮次]
"""
import sys, time, gc, statistics, json
import psutil, numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting51\实施")
import 算子_千层骨 as q  # noqa: E402

轮 = int(sys.argv[1]) if len(sys.argv) > 1 else 1
proc = psutil.Process()
cpu0 = psutil.cpu_percent(interval=0.5)
print("轮次", 轮, "CPU%%(0.5s)", cpu0, "本进程RSS_MB", round(proc.memory_info().rss / 1048576, 1))
print("硬件", psutil.cpu_count(logical=True), "逻辑核; 内存GB",
      round(psutil.virtual_memory().total / 1073741824, 2),
      "空闲GB", round(psutil.virtual_memory().available / 1073741824, 2))

结果 = {}
for D in (64, 256):
    k = q.千层核(D=D, 种子=7)
    for _ in range(5):
        k.前向一拍(None)
    for H in (1, 3):
        k.开关["环路"] = H
        for _ in range(3):
            k.前向一拍(None)
        gc.collect()
        gc.disable()
        r = []
        for _ in range(8):
            t = time.perf_counter()
            for _ in range(10):
                k.前向一拍(None)
            r.append((time.perf_counter() - t) / 10 * 1000)
        gc.enable()
        r.sort()
        结果["D%d_H%d" % (D, H)] = dict(
            median=round(statistics.median(r), 2), mean=round(statistics.mean(r), 2),
            p95=round(r[int(len(r) * 0.95) - 1], 2), min=round(r[0], 2), max=round(r[-1], 2),
            all=[round(x, 1) for x in r])
        print("D=%d H=%d  median %.2f ms  mean %.2f  p95 %.2f  min %.2f  max %.2f" %
              (D, H, statistics.median(r), statistics.mean(r), r[int(len(r) * 0.95) - 1], r[0], r[-1]))
        print("    全样本", [round(x, 1) for x in r])
    del k

print("H3/H1 比: D64 %.2f  D256 %.2f" %
      (结果["D64_H3"]["median"] / 结果["D64_H1"]["median"],
       结果["D256_H3"]["median"] / 结果["D256_H1"]["median"]))
# 两点解 F + R
for D in (64, 256):
    a, b = 结果["D%d_H1" % D]["median"], 结果["D%d_H3" % D]["median"]
    R = (b - a) / 2.0
    print("D=%d  拟合 F=%.2f ms  R=%.2f ms/轮" % (D, a - R, R))
print("CPU%%(结束)", psutil.cpu_percent(interval=None))
