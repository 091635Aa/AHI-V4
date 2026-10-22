# meeting55 黄队：声带真机成本（合成延迟/实时率/内存），不播放、不占声卡
import sys, os, time, json
import numpy as np

p = r"<workspace>/03_AHI\AHI4\meeting32\声带.py"
sys.path.insert(0, os.path.dirname(p))
tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_声带")
os.makedirs(tmp, exist_ok=True)
import 声带 as V

out = {"FS": V.FS, "tmp": tmp}
import tracemalloc
tracemalloc.start()
t0 = time.perf_counter()
try:
    器 = V.声带器官(seed=300, D=1024, 目录=tmp)
    out["构造_ms"] = round((time.perf_counter() - t0) * 1000, 1)
except TypeError:
    器 = V.声带器官(seed=300, D=1024)
    out["构造_ms"] = round((time.perf_counter() - t0) * 1000, 1)

for 文本 in ("今天天气不错", "今天天气不错我们一起出去走走吧好不好"):
    t0 = time.perf_counter()
    r = 器.发声(np.zeros(1024), 文本)
    dt = time.perf_counter() - t0
    w = r.get("波形") if isinstance(r, dict) else None
    n = int(np.size(w)) if w is not None else None
    out["文本%d字" % len(文本)] = dict(合成_ms=round(dt * 1000, 1), 采样点=n,
        时长秒=(round(n / V.FS, 3) if n else None),
        实时率=(round(dt / (n / V.FS), 4) if n else None),
        键=sorted(r.keys())[:12] if isinstance(r, dict) else str(type(r)))
cur, peak = tracemalloc.get_traced_memory()
out["tracemalloc峰值MB"] = round(peak / 1e6, 2)
tracemalloc.stop()
print(json.dumps(out, ensure_ascii=False, indent=1))
