# -*- coding: utf-8 -*-
"""复核探针 C1：V4 贯通全链路（六内核 × 两区室，120 拍）进程级峰值/均值/采样 + 模块与 CUDA 检查。
只读：不写文件；python -B 运行。等价于黄队 R1 §12.4 的 runpy 口径，外加 10ms RSS 采样。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\C1_全链路峰值.py
"""
import sys, os, time, threading, subprocess, json
import psutil

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

脚本 = r"<workspace>/03_AHI\AHI4\meeting49\实施\贯通_多核心.py"
p = psutil.Process()


def smi(tag):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        v = out.stdout.strip()
    except Exception as e:
        v = "ERR " + repr(e)
    print("%-14s nvidia-smi used: %s" % (tag, v))
    return v


def mi(tag):
    m = p.memory_info()
    print("%-14s rss_MB %6.1f  peak_wset_MB %6.1f" % (tag, m.rss / 1048576, m.peak_wset / 1048576))


mi("base")
smi("base")
samples = []
stop = threading.Event()


def 采样():
    while not stop.is_set():
        try:
            samples.append(p.memory_info().rss / 1048576)
        except Exception:
            pass
        stop.wait(0.010)


t = threading.Thread(target=采样, daemon=True)
t.start()
mi("before_run")
import runpy  # noqa: E402

t0 = time.perf_counter()
g = runpy.run_path(脚本, run_name="__main__")
dt = time.perf_counter() - t0
stop.set()
t.join(timeout=2)
mi("after_run")
smi("after_run")

n = len(samples)
if n:
    s = sorted(samples)
    print("采样: n=%d 间隔=10ms  mean=%.1f  median=%.1f  p95=%.1f  max=%.1f  min=%.1f" %
          (n, sum(s) / n, s[n // 2], s[int(n * 0.95) - 1], s[-1], s[0]))
    print("采样前20:", [round(x, 1) for x in samples[:20]])
print("墙钟: %.2f s  每拍 %.2f ms" % (dt, dt / 120 * 1000))
mods = sorted(k for k in sys.modules if any(w in k.lower() for w in ("torch", "cupy", "cuda", "tensorflow")))
print("重模块 in sys.modules:", mods)
print("自测全局键:", sorted(k for k in g if not k.startswith("__"))[:20])
print("（自测结果已由 __main__ 打印；此处不重跑，避免二次计峰）")
try:
    torch = sys.modules.get("torch")
    print("torch 导入?", torch is not None)
    if torch is not None:
        print("cuda 已初始化?", torch.cuda.is_initialized())
except Exception as e:
    print("torch 检查异常", repr(e))
