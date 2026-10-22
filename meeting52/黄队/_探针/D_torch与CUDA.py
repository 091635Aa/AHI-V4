# -*- coding: utf-8 -*-
"""复核探针 D：import torch 成本 + CUDA 上下文（max_memory_allocated vs nvidia-smi 整卡）。
只读：不写文件；python -B 运行。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\D_torch与CUDA.py
"""
import sys, time, subprocess
import psutil

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

p = psutil.Process()
MI = 1048576.0


def rss(tag):
    m = p.memory_info()
    print("%-22s rss_MB %7.1f  peak_wset_MB %7.1f" % (tag, m.rss / MI, m.peak_wset / MI))


def smi(tag):
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        print("%-22s nvidia-smi: %s" % (tag, o.stdout.strip()))
    except Exception as e:
        print("%-22s nvidia-smi ERR %r" % (tag, e))


rss("阶段0 空进程")
smi("阶段0 空进程")
import numpy  # noqa: E402
rss("阶段1 +numpy")
smi("阶段1 +numpy")
t0 = time.perf_counter()
import torch  # noqa: E402
dt = time.perf_counter() - t0
rss("阶段2 +torch")
smi("阶段2 +torch")
print("import torch 耗时 %.3f s  torch %s  cuda可用 %s  cuda已初始化 %s" %
      (dt, torch.__version__, torch.cuda.is_available(), torch.cuda.is_initialized()))
if torch.cuda.is_available():
    rss("阶段3a init前")
    smi("阶段3a init前")
    torch.cuda.init()
    torch.cuda.synchronize()
    rss("阶段3b init后")
    smi("阶段3b init后")
    print("阶段3 max_memory_allocated_MB %.3f" % (torch.cuda.max_memory_allocated() / MI))
    print("阶段3 memory_reserved_MB %.3f" % (torch.cuda.memory_reserved() / MI))
    print("阶段3 mem_get_info %s" % str(torch.cuda.mem_get_info()))
    torch.cuda.reset_peak_memory_stats()
    x = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    rss("阶段4 分配256MiB后")
    smi("阶段4 分配256MiB后")
    print("阶段4 max_memory_allocated_MB %.3f" % (torch.cuda.max_memory_allocated() / MI))
    print("阶段4 memory_reserved_MB %.3f" % (torch.cuda.memory_reserved() / MI))
    del x
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    rss("阶段5 释放后")
    smi("阶段5 释放后")
    print("阶段5 max_memory_allocated_MB %.3f" % (torch.cuda.max_memory_allocated() / MI))
    print("阶段5 memory_reserved_MB %.3f" % (torch.cuda.memory_reserved() / MI))
    time.sleep(0.3)
    smi("阶段5 释放后+0.3s")
rss("结束")
