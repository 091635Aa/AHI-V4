# meeting55 黄队 R1 探针（≤25s）：声卡设备/延迟 + 麦克风 RMS + 声带实时率
import json, time, os, sys, math

out = {}
import numpy as np

# ── 1) 设备清单 ──
try:
    import sounddevice as sd
    devs = sd.query_devices()
    out["默认设备"] = dict(In=int(sd.default.device[0]), Out=int(sd.default.device[1]))
    rows = []
    for i, d in enumerate(devs):
        rows.append(dict(i=i, name=d["name"][:28], in_ch=d["max_input_channels"],
                         out_ch=d["max_output_channels"], sr=int(d["default_samplerate"]),
                         lat_in=round(d.get("default_low_input_latency", 0), 4),
                         lat_out=round(d.get("default_low_output_latency", 0), 4)))
    out["设备"] = rows
    out["hostapi"] = [a["name"] for a in sd.query_hostapis()]
except Exception as e:
    out["sd_err"] = repr(e)[:200]

# ── 2) 麦克风 2.0s：RMS / 峰值 / 实测延迟 ──
try:
    import sounddevice as sd
    sr, sec, bs = 48000, 2.0, 512
    n = int(sr * sec)
    buf = np.zeros((n, 1), dtype=np.float32)
    idx = 0
    t0 = time.perf_counter()
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=bs) as st:
        while idx < n:
            blk, ov = st.read(min(bs, n - idx))
            buf[idx:idx + len(blk)] = blk[:1]
            idx += len(blk)
    wall = time.perf_counter() - t0
    x = buf[:, 0]
    out["麦克风"] = dict(sr=sr, 块=bs, 音频秒=round(wall, 3), 采集墙钟秒=round(wall, 3),
                        RMS=round(float(np.sqrt(np.mean(x * x))), 6),
                        RMS_dBFS=round(float(20 * math.log10(max(1e-9, np.sqrt(np.mean(x * x))))), 1),
                        峰值=round(float(np.max(np.abs(x))), 5),
                        stream_latency=round(float(st.latency), 4),
                        有削波=bool(np.max(np.abs(x)) > 0.999))
except Exception as e:
    out["mic_err"] = repr(e)[:200]

# ── 3) 声带.py 实时率 ──
try:
    p = r"<workspace>/03_AHI\AHI4\meeting32\声带.py"
    sys.path.insert(0, os.path.dirname(p))
    import 声带 as V
    out["声带_符号"] = [k for k in dir(V) if not k.startswith("_")][:40]
    fn = None
    for name in ("合成", "发声", "说", "合成_wav", "生成", "main", "合成一句"):
        if hasattr(V, name):
            fn = name; break
    out["声带_入口"] = fn
    if fn:
        f = getattr(V, fn)
        import inspect
        out["声带_签名"] = str(inspect.signature(f))[:200]
        t0 = time.perf_counter()
        try:
            r = f(1.0)                       # 试 1 秒
        except TypeError:
            r = f()
        dt = time.perf_counter() - t0
        arr = r if isinstance(r, np.ndarray) else (r[0] if isinstance(r, tuple) else None)
        out["声带"] = dict(调用=fn, 墙钟秒=round(dt, 3),
                          输出类型=type(r).__name__,
                          采样点数=(int(arr.size) if isinstance(arr, np.ndarray) else None),
                          实时率=round(dt / 1.0, 3))
except Exception as e:
    out["vo_err"] = repr(e)[:300]

# ── 4) pyttsx3 初始化成本（不发声） ──
try:
    import pyttsx3
    t0 = time.perf_counter()
    eng = pyttsx3.init()
    dt = time.perf_counter() - t0
    vs = [v.name + "/" + v.id for v in eng.getProperty("voices")][:6]
    rate = eng.getProperty("rate")
    out["pyttsx3"] = dict(init_ms=round(dt * 1000, 1), rate=rate, voices=vs)
    del eng
except Exception as e:
    out["tts_err"] = repr(e)[:200]

print(json.dumps(out, ensure_ascii=False, indent=1))
