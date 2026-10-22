# -*- coding: utf-8 -*-
"""复核探针 C4：同进程「import torch + V4 全链路」峰值 —— 检验 torch 常驻账是否漏了全链路自身。
只读：不写文件；python -B 运行。不建 CUDA 上下文（只 import torch）。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\C4_链路加torch.py
"""
import sys, time, threading, runpy
import psutil

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

p = psutil.Process()
MI = 1048576.0


def rss(tag):
    m = p.memory_info()
    print("%-14s rss_MB %6.1f  peak_wset_MB %6.1f" % (tag, m.rss / MI, m.peak_wset / MI))


rss("base")
t0 = time.perf_counter()
import torch  # noqa: E402
print("import torch %.2f s  cuda初始化? %s" % (time.perf_counter() - t0, torch.cuda.is_initialized()))
rss("+torch")
samples, stop = [], threading.Event()


def 采样():
    while not stop.is_set():
        samples.append(p.memory_info().rss / MI)
        stop.wait(0.010)


th = threading.Thread(target=采样, daemon=True)
th.start()
t0 = time.perf_counter()
runpy.run_path(r"<workspace>/03_AHI\AHI4\meeting49\实施\贯通_多核心.py", run_name="__main__")
dt = time.perf_counter() - t0
stop.set()
th.join(timeout=2)
rss("after_run")
s = sorted(samples)
print("墙钟 %.2f s 采样 n=%d mean=%.1f median=%.1f p95=%.1f max=%.1f" %
      (dt, len(s), sum(s) / len(s), s[len(s) // 2], s[int(len(s) * .95) - 1], s[-1]))
