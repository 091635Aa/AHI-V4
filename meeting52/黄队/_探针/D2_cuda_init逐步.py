# -*- coding: utf-8 -*-
"""复核探针 D2：逐步量 import torch / is_available / cuda.init 的 host RSS 与整卡 used。
只读：不写文件；python -B 运行。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\D2_cuda_init逐步.py
"""
import sys, subprocess
import psutil

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

p = psutil.Process()
MI = 1048576.0


def 报(tag):
    m = p.memory_info()
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        o = repr(e)
    print("%-28s rss_MB %7.1f peak_wset %7.1f smi_used %s" %
          (tag, m.rss / MI, m.peak_wset / MI, o))


报("0 空进程")
import numpy  # noqa: E402
报("1 +numpy")
import torch  # noqa: E402
报("2 +torch(import)")
print("cuda.is_available() =", torch.cuda.is_available())
报("3 is_available后")
print("cuda.is_initialized() =", torch.cuda.is_initialized())
报("4 is_initialized后")
torch.cuda.init()
报("5 cuda.init后")
torch.cuda.synchronize()
报("6 synchronize后")
print("max_memory_allocated_MB", torch.cuda.max_memory_allocated() / MI)
报("7 读数后")
