# -*- coding: utf-8 -*-
"""复核探针 C3：显式 D=256 的 V4 贯通全链路（120 拍）峰值/均值/墙钟。
说明：默认全链路是 D=1024（桥母语维）；本探针显式 造双区室(D=256) 做对照。
只读：不写文件；python -B 运行。
复跑：$env:PYTHONIOENCODING='utf-8'; python -B 黄队\\_探针\\C3_全链路D256.py
"""
import sys, importlib.util, time, threading
import psutil

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
print("base rss_MB %.1f" % (p.memory_info().rss / MI))
双 = m.造双区室(D=256)
print("双[D] =", 双["D"], " 桥.探针 =", 双["桥"].探针.shape)
print("造后 rss_MB %.1f" % (p.memory_info().rss / MI))
samples, stop = [], threading.Event()


def 采样():
    while not stop.is_set():
        samples.append(p.memory_info().rss / MI)
        stop.wait(0.010)


th = threading.Thread(target=采样, daemon=True)
th.start()
t0 = time.perf_counter()
果 = m.跑(双, 120)
dt = time.perf_counter() - t0
stop.set()
th.join(timeout=2)
mm = p.memory_info()
print("已跑", 果["已跑"], "异常", 果["异常"], "自我态有限", 果["自我态有限"])
print("墙钟 %.2f s  每拍 %.2f ms" % (dt, dt / 120 * 1000))
print("跑完 rss_MB %.1f  peak_wset_MB %.1f" % (mm.rss / MI, mm.peak_wset / MI))
s = sorted(samples)
print("采样 n=%d  mean=%.1f  median=%.1f  p95=%.1f  max=%.1f  min=%.1f" %
      (len(s), sum(s) / len(s), s[len(s) // 2], s[int(len(s) * .95) - 1], s[-1], s[0]))
