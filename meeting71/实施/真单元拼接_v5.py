# -*- coding: utf-8 -*-
"""meeting71/实施/真单元拼接_v5.py —— v5a：真音频单元（声学切分 + DP 标签）检索 + 40 ms 交叉淡化拼接。

不训练、不做 PSOLA。库落 <path>\AHI_m71\库_真单元.pkl。
用法：
  & $py 真单元拼接_v5.py 建库
  & $py 真单元拼接_v5.py 合成
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M71 = HERE.parent
M67 = M71.parent / "meeting67"
M66 = M71.parent / "meeting66"
E = Path(r"<path>\AHI_m71")
SR = 22050
FR = 110
L根 = Path(r"<path>\SpeakerA7代模型训练数据")
库路 = E / "库_真单元.pkl"
句表 = {
    "V5a主句": "今天天气很好我听见了自己的声音我想再说一遍",
    "V5a新句1": "今天的风很轻我慢慢说给你听",
    "V5a新句2": "窗外的雨停了我该休息一会",
    "V5a新句3": "你先别走我还想再说两句",
}


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


耳 = 载("耳蜗前端_S0_2", M71.parent / "meeting69" / "实施" / "耳蜗前端_S0_2.py")
V3 = 载("v3", HERE / "声学单元移植_v3.py")


def 读妙(p):
    return 耳.读wav(p)


def 找单元(x, 最短=0.06, 最长=0.45):
    hop = 220
    e = np.sqrt(np.convolve(np.asarray(x, np.float64) ** 2, np.ones(hop) / hop, mode="same")[::hop])
    e = np.convolve(e, np.ones(3) / 3, mode="same")
    if e.size < 5:
        return []
    thr = max(0.06 * float(np.percentile(e, 95)), 1e-5)
    cand = [i for i in range(1, len(e) - 1) if e[i] >= e[i - 1] and e[i] >= e[i + 1] and e[i] > thr]
    keep = []
    for i in cand:
        if not keep or (i - keep[-1]) * hop / SR >= 最短:
            keep.append(i)
        elif e[i] > e[keep[-1]]:
            keep[-1] = i
    if len(keep) < 2:
        return []
    边界 = [0]
    for a, b in zip(keep[:-1], keep[1:]):
        j = int(a + np.argmin(e[a:b + 1]))
        边界.append(j)
    边界.append(len(e) - 1)
    单元 = []
    最短短 = 最短 * SR / hop
    for a, b in zip(边界[:-1], 边界[1:]):
        if (b - a) < 最短短 or (b - a) * hop / SR > 最长:
            continue
        单元.append((a * hop, b * hop))
    return 单元


def 源列表():
    """(wav, npz, 文本, 声/韵/调/帧数, 偏移帧, sid)"""
    出 = []
    对齐 = json.loads((M67 / "产物" / "P2" / "对齐证据.json").read_text(encoding="utf-8"))
    偏移 = {r["id"]: int(r["偏移帧"]) for r in 对齐["per_sample"]}
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    for f in sorted((M66 / "产物" / "数据_对齐asr" / "样本").glob("*.npz")):
        sid = int(f.stem)
        出.append(dict(wav=L根 / 样本[sid]["wav"], npz=f, 文本=样本[sid]["文"],
                      偏移=偏移.get(sid, 0), sid=sid))
    # 新数据
    m2 = E / "数据npz" / "meta_扩容.json"
    if m2.exists():
        d = json.loads(m2.read_text(encoding="utf-8"))
        rec = {r["id"]: r for r in d["记录"]}
        for f in sorted((E / "数据npz" / "样本").glob("*.npz")):
            sid = int(f.stem)
            if sid not in rec:
                continue
            w = Path(rec[sid]["wav"])
            if not w.exists():
                for root in (E / "切_6h" / "片段", E / "切_7h" / "片段"):
                    cand = root / Path(rec[sid]["wav"]).name
                    if cand.exists():
                        w = cand
                        break
            if w.exists():
                出.append(dict(wav=w, npz=f, 文本=rec[sid].get("文本", ""), 偏移=0, sid=10000 + sid))
    return 出


def 对齐标签(单元中心秒, 字中心秒, 罚=0.20):
    """DP 单调对齐 → (标签, 是否匹配)。未匹配的字单元不参与建库。"""
    n = len(单元中心秒); m = len(字中心秒)
    if n == 0 or m == 0:
        return [-1] * n, [False] * n
    INF = 1e18
    dp = np.full((n + 1, m + 1), INF)
    bk = np.zeros((n + 1, m + 1), np.int8)
    dp[0, 0] = 0.0
    cu = np.asarray(单元中心秒, np.float64); cc = np.asarray(字中心秒, np.float64)
    for i in range(n + 1):
        for j in range(m + 1):
            v = dp[i, j]
            if v >= INF:
                continue
            if i < n and j < m:
                c = abs(cu[i] - cc[j])
                if v + c < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = v + c; bk[i + 1, j + 1] = 1
            if i < n and v + 罚 < dp[i + 1, j]:
                dp[i + 1, j] = v + 罚; bk[i + 1, j] = 2
            if j < m and v + 罚 < dp[i, j + 1]:
                dp[i, j + 1] = v + 罚; bk[i, j + 1] = 3
    标签 = [-1] * n; 匹配 = [False] * n
    i, j = n, m
    while i > 0 or j > 0:
        op = bk[i, j]
        if op == 1:
            标签[i - 1] = j - 1; 匹配[i - 1] = True; i -= 1; j -= 1
        elif op == 2:
            i -= 1
        else:
            j -= 1
    return 标签, 匹配


def 建库():
    E.mkdir(parents=True, exist_ok=True)
    库 = []
    t0 = time.time()
    for k, src in enumerate(源列表()):
        x = 读妙(src["wav"])
        z = np.load(src["npz"], allow_pickle=True)
        声, 韵, 调id, 帧数 = z["声母"], z["韵母"], z["调"], z["音节帧数"]
        chs = [c for c in src["文本"] if "\u4e00" <= c <= "\u9fff"]
        n = min(len(chs), len(声), len(帧数))
        if n < 2:
            continue
        cen = (src["偏移"] + np.cumsum(帧数[:n]) - 帧数[:n] / 2) * 0.01
        单元秒 = 找单元(x)
        if len(单元秒) < 2:
            continue
        中心 = np.array([(a + b) / 2 / SR for a, b in 单元秒])
        标签, 匹配 = 对齐标签(中心, cen)
        for (a, b), j, ok in zip(单元秒, 标签, 匹配):
            if not ok or j < 0 or j >= n:
                continue
            段 = np.asarray(x[a:b], np.float32)
            if len(段) < int(0.05 * SR):
                continue
            rms = float(np.sqrt(np.mean(段 ** 2))) + 1e-9
            库.append(dict(ch=chs[j], sm=int(声[j]), ym=int(韵[j]), tone=int(调id[j]),
                           波形=(段 / rms * 0.05).astype(np.float32), 秒=round(len(段) / SR, 4)))
        if (k + 1) % 40 == 0:
            print("|", k + 1, "样本 单元", len(库), round(time.time() - t0, 1), "s", flush=True)
    with open(库路, "wb") as f:
        pickle.dump(库, f, protocol=4)
    print("库单元", len(库), "耗时", round(time.time() - t0, 1), flush=True)
    return 0


def 建字表(库):
    from collections import Counter
    表 = {}
    for u in 库:
        表.setdefault(u["ch"], []).append((u["sm"], u["ym"], u["tone"]))
    return {c: Counter(v).most_common(1)[0][0] for c, v in 表.items()}


def 选单元(库, 键, ch, 目标秒):
    sm, ym, tone = 键
    for 条件 in ((lambda u: u["sm"] == sm and u["ym"] == ym and u["tone"] == tone and u["ch"] == ch),
                 (lambda u: u["sm"] == sm and u["ym"] == ym and u["tone"] == tone),
                 (lambda u: u["sm"] == sm and u["ym"] == ym)):
        候 = [u for u in 库 if 条件(u)]
        if 候:
            候.sort(key=lambda u: abs(u["秒"] - 目标秒))
            return 候[0]
    return None


def 合成(文本, 库, 字表, 时长表, 全中位秒=0.22):
    keys = V3.键_of(文本, 字表)
    选 = []
    缺 = 0
    for ch, 键, src in keys:
        目标 = 时长表.get((int(键[0]), int(键[1]), int(键[2])), None)
        目标秒 = 目标 * 0.01 if 目标 else 全中位秒
        u = 选单元(库, 键, ch, 目标秒)
        if u is None:
            缺 += 1
            continue
        选.append(u)
    if not 选:
        return None, 缺
    淡 = int(0.040 * SR)
    out = 选[0]["波形"].copy()
    for u in 选[1:]:
        y = u["波形"]
        n = min(淡, len(out), len(y))
        if n > 1:
            w = np.linspace(0, 1, n).astype(np.float32)
            out = np.concatenate([out[:-n], out[-n:] * (1 - w) + y[:n] * w, y[n:]])
        else:
            out = np.concatenate([out, y])
    p = float(np.abs(out).max())
    if p > 1e-9:
        out = out / p * 0.9
    return out.astype(np.float32), 缺


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    E.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "建库":
        return 建库()
    with open(库路, "rb") as f:
        库 = pickle.load(f)
    字表 = 建字表(库)
    # 时长表：来自 P6v4 数据（含 346 条中位时长）
    with open(E / "P6v4_数据.pkl", "rb") as f:
        d = pickle.load(f)
    时长表 = d["时长表"]; 全中位 = d["全中位"]
    from faster_whisper import WhisperModel
    import glob
    snap = glob.glob(str(Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-medium" / "snapshots" / "*"))[0]
    wm = WhisperModel(snap, device="cpu", compute_type="int8")
    量 = V3.量
    引 = 量.载引擎()
    报 = dict(工具="真单元拼接_v5.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              库单元=len(库), 项=[])
    for 名, 文 in 句表.items():
        import hashlib
        y, 缺 = 合成(文, 库, 字表, 时长表, 全中位 * 0.01)
        if y is None:
            报["项"].append(dict(样本=名, 失败="无单元")); continue
        p = E / "P6v5" / f"{名}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        写(p, y)
        sha = hashlib.sha256((y * 32767).astype(np.int16).tobytes()).hexdigest()[:16]
        q = 引["转写"](str(p))
        segs, _ = wm.transcribe(str(p), language="zh", beam_size=5)
        wh = "".join(s.text for s in segs).strip()
        cer = round(量.CER(文, q), 4)
        报["项"].append(dict(样本=名, 文本=文, 路径=str(p), 秒=round(len(y) / SR, 3), 缺=缺,
                            sha=sha, 转写=q, Whisper=wh, CER=cer))
        print("|", 名, "CER=%.3f" % cer, "缺", 缺, "|", q[:30], flush=True)
    r0 = [x for x in 报["项"] if "CER" in x]
    报["均值CER"] = round(float(np.mean([x["CER"] for x in r0])), 4) if r0 else None
    (E / "P6v5" / "读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("均值", 报["均值CER"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
