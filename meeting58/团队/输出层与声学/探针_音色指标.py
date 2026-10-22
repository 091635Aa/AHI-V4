# -*- coding: utf-8 -*-
"""meeting58 · E2 探针 v2：4 音色档案的可分量测 + 8 参数活性 + 增益不变性 + 确定性。

v2 相对 v1 的修正（v1 自检发现的问题，必须一起读）：
  · v1 用 0.09s 的「啊」做频谱找峰，被谐波串吞掉 → 峰位是 F0 谐波不是共振峰；
    v2 改为**帧参数真值**：直接调 声带器官.音节帧()，比 f1/f2/f3 轨迹 (零估计误差)；
    频谱侧只作旁证，用 5 音节「啊啊啊啊啊」+ 15 bin (~470Hz) 平滑。
  · v1 的 F0 中位来自自相关估计器；v2 同时给 帧参数中位（真值）与估计器读数，差值即估计器误差。

纪律：只读 import；所有 wav/台账写本目录 产物/；sys.dont_write_bytecode；CPU only；单次 <10s。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import wave

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _找(*段):
    p = HERE
    for _ in range(5):
        cand = os.path.join(p, *段)
        if os.path.exists(cand):
            return cand
        p = os.path.dirname(p)
    raise SystemExit("找不到: %s" % (段,))


M55 = _找("meeting55", "实施")
sys.path.insert(0, M55)
import 说_音色 as S  # noqa: E402  只读 import

OUT = os.path.join(HERE, "产物")
WAV = os.path.join(OUT, "发声")
os.makedirs(WAV, exist_ok=True)

FS = 16000
带 = ((0, 250), (250, 500), (500, 1000), (1000, 2000),
      (2000, 3000), (3000, 4000), (4000, 8000))

器 = S.声带发声(seed=300, D=1024)
器.目录 = WAV                                    # 不写 meeting55 产物
器.台账路径 = os.path.join(OUT, "发声台账.jsonl")
S._器 = 器

文本s = ("我在听。", "今天很好。", "你好吗？")
档案s = tuple(S.音色档案.keys())
固定向量 = np.sign(np.random.default_rng(55).standard_normal(器.D)).astype(np.float32)
QS = 器.分位读数(固定向量)


def 合成(参: dict, 文本: str, 情绪="平静", 语速倍=1.0, 音量倍=1.0) -> dict:
    """精确复刻 说_音色.py:105-114（去文件名/溯源包装），用于参数扫描。"""
    器.当前律 = S.情绪律(参, 情绪)
    器.当前律["语速"] = float(min(max(器.当前律["语速"] * 语速倍, 1.5), 7.0))
    器.当前律["幅度"] = float(min(max(器.当前律["幅度"] * 音量倍, 0.1), 1.0))
    遥测 = dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5)
    器.当前律["_ñ"] = 0.5
    return 器.发声(固定向量[:器.D], 文本, 遥测)


def 律of(名, 情绪="平静"):
    律 = S.情绪律(dict(S.音色档案[名]), 情绪)
    律["_ñ"] = 0.5
    return 律


def 帧真值(名, 文本, 情绪="平静"):
    """直接读 音节帧() 产出的三轨帧参数（合成前真值，零估计误差）。"""
    return 帧真值律(律of(名, 情绪), 文本)


def 帧真值律(律, 文本):
    帧 = 器.音节帧(文本, 律, QS)
    f0 = np.asarray(帧["f0"], float)
    f1 = np.asarray(帧["f1"], float)
    f2 = np.asarray(帧["f2"], float)
    f3 = np.asarray(帧["f3"], float)
    v = f0[f0 > 0]
    h = hashlib.sha256()
    for a in (np.round(f1, 3), np.round(f2, 3), np.round(f3, 3)):
        h.update(a.tobytes())
    return dict(帧数=len(f0),
                半音中位=round(float(np.median(12 * np.log2(v / 100.0))), 3) if len(v) else None,
                半音std=round(float(np.std(12 * np.log2(v / 100.0))), 3) if len(v) else None,
                f1中位=round(float(np.median(f1)), 1), f2中位=round(float(np.median(f2)), 1),
                f3中位=round(float(np.median(f3)), 1),
                共振峰轨迹指纹=h.hexdigest()[:16],
                时长秒=round(len(f0) / 100.0, 3))


def 路径(rec):
    return os.path.join(器.目录, rec["wav"])


def _lpc(ac: np.ndarray, p: int) -> np.ndarray:
    """Levinson-Durbin：自相关 → LPC 系数 a（a[0]=1）。"""
    a = np.zeros(p + 1)
    a[0] = 1.0
    e = float(ac[0]) + 1e-12
    for i in range(1, p + 1):
        累 = ac[i] + float(np.dot(a[1:i], ac[i - 1:0:-1]))
        k = -累 / e
        新 = a.copy()
        for j in range(1, i):
            新[j] = a[j] + k * a[i - j]
        新[i] = k
        a = 新
        e *= max(1.0 - k * k, 1e-9)
    return a


def LPC共振峰(x, sr=FS, 阶=12):
    """LPC 极点→F1/F2/F3（浊帧中位）。纯 numpy、确定性。"""
    win, hop = int(0.025 * sr), int(0.010 * sr)
    if len(x) < win + 1:
        return None
    E = 帧能量(x, win, hop)
    if E.max() <= 0:
        return None
    门 = E > 0.05 * E.max()
    F = [[], [], []]
    for i in np.where(门)[0]:
        seg = x[i * hop:i * hop + win].astype(float)
        seg = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])     # 预加重
        seg *= np.hamming(len(seg))
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:len(seg) + 阶]
        if ac[0] <= 0:
            continue
        a = _lpc(ac, 阶)
        try:
            根 = np.roots(a)
        except Exception:
            continue
        根 = 根[np.imag(根) > 0]
        频 = np.angle(根) * sr / (2 * np.pi)
        带 = -np.log(np.maximum(np.abs(根), 1e-9)) * sr / np.pi
        好 = sorted(float(f) for f, b in zip(频, 带)
                   if 90 < f < sr / 2 - 200 and b < 500)
        if len(好) >= 3:
            for j in range(3):
                F[j].append(好[j])
    if not F[0]:
        return None
    return [round(float(np.median(z)), 1) for z in F]


def 轨迹(律, 文本, 点数=101):
    """f1/f2/f3 轨迹插值到固定点数（消除语速→帧数差异），用于逐点对齐比较。"""
    帧 = 器.音节帧(文本, 律, QS)
    out = []
    for 键 in ("f1", "f2", "f3"):
        a = np.asarray(帧[键], float)
        out.append(np.interp(np.linspace(0, 1, 点数), np.linspace(0, 1, len(a)), a))
    return np.stack(out)


def 倒谱包络峰(x, sr=FS, 帧长=0.032, 帧移=0.010, 提升=30, 上限=3600.0):
    """倒谱 lifter 平滑谱包络→峰（对周期脉冲源比 LPC 稳）。返回 [(Hz, 相对电平)]。"""
    win, hop = int(帧长 * sr), int(帧移 * sr)
    if len(x) < win + 1:
        return None
    E = 帧能量(x, win, hop)
    if E.max() <= 0:
        return None
    门 = E > 0.05 * E.max()
    S = np.zeros(win // 2 + 1)
    for i in np.where(门)[0]:
        seg = x[i * hop:i * hop + win] * np.hamming(win)
        S += np.abs(np.fft.rfft(seg)) ** 2
    S /= max(int(门.sum()), 1)
    logS = np.log(S + 1e-12)
    c = np.fft.irfft(logS, n=win)
    c2 = np.zeros_like(c)
    c2[:提升] = c[:提升]
    c2[-(提升 - 1):] = c[-(提升 - 1):]
    env = np.fft.rfft(c2).real[:len(logS)]
    fr = np.fft.rfftfreq(win, 1.0 / sr)
    cand = [i for i in range(3, len(env) - 3)
            if 250 < fr[i] < 上限 and env[i] > env[i - 1] and env[i] >= env[i + 1]]
    cand.sort(key=lambda i: -env[i])
    选 = []
    for i in cand:
        if all(abs(fr[i] - fr[j]) > 250 for j in 选):
            选.append(i)
        if len(选) >= 3:
            break
    return [(round(float(fr[i])), round(float(env[i] - env[cand[0]]), 2)) for i in sorted(选)]


def 读wav(p):
    with wave.open(p, "rb") as w:
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64) / 32768.0
        return w.getframerate(), x


def 字节(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def 帧能量(x, win, hop):
    nf = 1 + max(0, (len(x) - win) // hop)
    return np.array([float(np.sum(x[i * hop:i * hop + win] ** 2)) for i in range(nf)])


def 包络(x, sr=FS, 帧长=0.032, 帧移=0.010):
    """7 带相对 log 包络（dB，逐帧减总能量 ≡ 增益不变）。"""
    win, hop = int(帧长 * sr), int(帧移 * sr)
    if len(x) < win + 1:
        return None
    E = 帧能量(x, win, hop)
    if E.max() <= 0:
        return None
    门 = E > 0.01 * E.max()
    fr = np.fft.rfftfreq(win, 1.0 / sr)
    掩 = [(fr >= lo) & (fr < hi) for lo, hi in 带]
    L = []
    for i in np.where(门)[0]:
        seg = x[i * hop:i * hop + win] * np.hanning(win)
        mag = np.abs(np.fft.rfft(seg))
        e = np.array([np.sum(mag[m] ** 2) for m in 掩]) + 1e-14
        L.append(10.0 * np.log10(e / e.sum()))
    L = np.asarray(L)
    中心 = np.array([np.sqrt(max(lo, 1.0) * hi) for lo, hi in 带])
    return dict(均值=L.mean(0), 帧std=L.std(0),
                谱倾斜=float(np.polyfit(np.log2(中心), L.mean(0), 1)[0]), 帧数=int(len(L)))


def f0统计(x, sr=FS):
    f0 = S._F0轨迹(x, sr)
    v = f0[np.isfinite(f0) & (f0 > 60)]
    if len(v) == 0:
        return dict(中位=None, 半音中位=None, 半音std=None, 浊帧=0)
    st = 12.0 * np.log2(v / 100.0)
    return dict(中位=float(np.median(v)), 半音中位=float(np.median(st)),
                半音std=float(np.std(st)), 浊帧=int(len(v)))


def 频谱峰(p, 上限=3600.0):
    """旁证：全帧平均 log 谱 + 15bin(~470Hz) 平滑后找峰（去谐波串）。"""
    sr, x = 读wav(p)
    win, hop = int(0.032 * sr), int(0.010 * sr)
    E = 帧能量(x, win, hop)
    门 = E > 0.05 * E.max()
    累 = np.zeros(win // 2 + 1)
    for i in np.where(门)[0]:
        seg = x[i * hop:i * hop + win] * np.hanning(win)
        m = np.abs(np.fft.rfft(seg))
        累 += np.log(m + 1e-9)
    累 /= max(int(门.sum()), 1)
    平 = np.convolve(累, np.ones(15) / 15, mode="same")
    fr = np.fft.rfftfreq(win, 1.0 / sr)
    cand = [i for i in range(8, len(平) - 8)
            if 300 < fr[i] < 上限 and 平[i] > 平[i - 1] and 平[i] >= 平[i + 1]]
    cand.sort(key=lambda i: -平[i])
    选 = []
    for i in cand:
        if all(abs(fr[i] - fr[j]) > 250 for j in 选):
            选.append(i)
        if len(选) >= 3:
            break
    return sorted(round(float(fr[i])) for i in 选)


def 音色特征(p):
    sr, x = 读wav(p)
    env = 包络(x, sr)
    f = f0统计(x, sr)
    if env is None or f["中位"] is None:
        return dict(错误="特征不足")
    v17 = np.concatenate([env["均值"], env["帧std"],
                          [f["半音中位"], f["半音std"], env["谱倾斜"]]])
    return dict(F0中位=round(f["中位"], 1), 半音中位=round(f["半音中位"], 3),
                半音std=round(f["半音std"], 3), 谱倾斜=round(env["谱倾斜"], 3),
                包络dB=[round(float(z), 2) for z in env["均值"]],
                帧stddB=[round(float(z), 2) for z in env["帧std"]],
                v17=v17, 能量=round(float(np.sqrt(np.mean(x ** 2))), 5),
                时长=round(len(x) / sr, 3))


# ───────── A. 4 档案 × 3 文本（现成 说()）+ 帧参数真值 ─────────
t0 = time.perf_counter()
格, 真 = {}, {}
for 名 in 档案s:
    for 文 in 文本s:
        rec = S.说(文, 音色=名, 情绪="平静")
        ft = 音色特征(路径(rec))
        ft["wav"] = rec["wav"]
        ft["实时率"] = rec["实时率"]
        ft["通过"] = rec["通过"]
        格[(名, 文)] = ft
        真[(名, 文)] = 帧真值(名, 文)
耗时A = time.perf_counter() - t0

v17 = {k: 格[k]["v17"] for k in 格}
尺度 = np.zeros(17)
for 名 in 档案s:
    M = np.stack([v17[(名, 文)] for 文 in 文本s])
    尺度 += M.std(0, ddof=1)
尺度 /= len(档案s)
尺度 = np.maximum(尺度, np.array([0.5] * 7 + [0.5] * 7 + [0.15, 0.15, 0.5]))


def 距离(a, b):
    d = (a - b) / 尺度
    return float(np.sqrt(np.mean(d * d)))


def 余弦(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


对 = []
for i in range(len(档案s)):
    for j in range(i + 1, len(档案s)):
        A, B = 档案s[i], 档案s[j]
        ds, cs, f0d = [], [], []
        for 文 in 文本s:
            ds.append(距离(v17[(A, 文)], v17[(B, 文)]))
            cs.append(余弦(格[(A, 文)]["包络dB"], 格[(B, 文)]["包络dB"]))
            f0d.append(abs(真[(A, 文)]["半音中位"] - 真[(B, 文)]["半音中位"]))
        # 同档案跨文本波动（判别式的天然噪声）
        噪 = []
        for 名 in (A, B):
            for a in range(len(文本s)):
                for b in range(a + 1, len(文本s)):
                    噪.append(距离(v17[(名, 文本s[a])], v17[(名, 文本s[b])]))
        对.append(dict(档对="%s|%s" % (A, B),
                       声纹代理距离=round(float(np.mean(ds)), 2),
                       同档跨文本=round(float(np.mean(噪)), 2),
                       判别比=round(float(np.mean(ds) / (np.mean(噪) + 1e-9)), 3),
                       包络余弦=round(float(np.mean(cs)), 4),
                       帧参数F0半音距=round(float(np.mean(f0d)), 2)))

内 = []
for 名 in 档案s:
    for a in range(len(文本s)):
        for b in range(a + 1, len(文本s)):
            x1, x2 = 文本s[a], 文本s[b]
            内.append(dict(档案=名, 文对="%s|%s" % (x1, x2),
                           声纹代理距离=round(距离(v17[(名, x1)], v17[(名, x2)]), 2),
                           包络余弦=round(余弦(格[(名, x1)]["包络dB"],
                                             格[(名, x2)]["包络dB"]), 4),
                           帧参数F0半音距=round(abs(真[(名, x1)]["半音中位"] -
                                             真[(名, x2)]["半音中位"]), 2)))

# ───────── B. 共振峰：固定语速下的敏感性（严格）+ 倒谱包络实测 ─────────
基律 = 律of("女声-明亮")
B1 = {}
for 键, 值 in (("基频", 301.0), ("气声度", 0.60), ("声区", 0.65),
               ("紧喉度", 0.90), ("粗糙度", 0.90), ("送气量", 0.80), ("幅度", 0.20)):
    律 = dict(基律)
    律[键] = 值
    B1[键] = round(float(np.max([np.max(np.abs(轨迹(律, 文) - 轨迹(基律, 文)))
                                for 文 in 文本s])), 6)
B1["语速改(帧数变)"] = None

倒谱 = {}
for 名 in 档案s:
    倒谱[名] = {}
    for 文 in 文本s:
        倒谱[名][文] = 倒谱包络峰(读wav(路径(合成(dict(S.音色档案[名]), 文)))[1])

甲 = {}
for 名 in 档案s:
    rec = 合成(dict(S.音色档案[名]), "啊啊啊啊啊")
    甲[名] = dict(倒谱峰=倒谱包络峰(读wav(路径(rec))[1]),
                  帧参数=(真[(名, "我在听。")]["f1中位"], 真[(名, "我在听。")]["f2中位"],
                        真[(名, "我在听。")]["f3中位"]))

# ───────── C. 8 参数活性 ─────────
基准 = dict(S.音色档案["女声-明亮"])
基准f = 音色特征(路径(合成(基准, "我在听。")))
基准rec = 合成(基准, "我在听。")
基准字节 = 字节(路径(基准rec))
扫 = {
    "基频": ("基频", 基准["基频"] * 1.40),
    "语速": ("语速", 基准["语速"] * 0.55),
    "幅度": ("幅度", 基准["幅度"] * 0.50),
    "气声度": ("气声度", min(基准["气声度"] + 0.45, 0.8)),
    "紧喉度": ("紧喉度", min(基准["紧喉度"] + 0.55, 1.0)),
    "粗糙度": ("粗糙度", 0.95),
    "送气量": ("送气量", min(基准["送气量"] + 0.60, 1.0)),
    "声区": ("声区", min(基准["声区"] + 0.60, 1.0)),
}
活性 = {}
基准真 = 真[("女声-明亮", "我在听。")]
for 名, (键, 新值) in 扫.items():
    参 = dict(基准)
    参[键] = 新值
    r = 合成(参, "我在听。")
    ft = 音色特征(路径(r))
    律 = 律of("女声-明亮")
    律[键] = 新值
    d2 = 帧真值律(律, "我在听。")
    活性[名] = dict(参数=名, 值=round(float(新值), 3), 逐位同=字节(路径(r)) == 基准字节,
                    Δ帧F0半音=round(d2["半音中位"] - 基准真["半音中位"], 2),
                    Δ帧数=d2["帧数"] - 基准真["帧数"],
                    Δ时长=round(ft["时长"] - 基准f["时长"], 3),
                    Δ能量dB=round(20 * np.log10((ft["能量"] + 1e-9) / (基准f["能量"] + 1e-9)), 2),
                    Δ谱倾斜=round(ft["谱倾斜"] - 基准f["谱倾斜"], 3),
                    包络余弦=round(余弦(ft["包络dB"], 基准f["包络dB"]), 4))

# ───────── D. 增益不变性 / 峰值归一化 ─────────
增益 = {}
for 倍 in (0.4, 0.7, 1.0, 1.4, 2.0):
    r = 合成(基准, "我在听。", 音量倍=倍)
    ft = 音色特征(路径(r))
    增益["音量倍%.1f" % 倍] = dict(能量=ft["能量"], 峰值=round(r["自检"]["峰值"], 4),
                               包络余弦=round(余弦(ft["包络dB"], 基准f["包络dB"]), 4),
                               字节=字节(路径(r)))

# ───────── E. 确定性 ─────────
r1 = S.说("你好吗？", 音色="女声-明亮", 情绪="平静")
r2 = S.说("你好吗？", 音色="女声-明亮", 情绪="平静")
确定 = dict(两趟同字节=字节(路径(r1)) == 字节(路径(r2)), 指纹1=字节(路径(r1)),
            指纹2=字节(路径(r2)), 文件名不同=r1["wav"] != r2["wav"], wav1=r1["wav"], wav2=r2["wav"])

# ───────── F. 情绪位移 vs 档案位移（谁吞掉谁） ─────────
文0 = "我在听。"
基f0 = 真[("女声-明亮", 文0)]["半音中位"]
基包 = 格[("女声-明亮", 文0)]["包络dB"]
情绪位移 = {}
for 情 in ("平静", "高兴", "低落", "急促"):
    r = S.说(文0, 音色="女声-明亮", 情绪=情)
    ft = 音色特征(路径(r))
    d = 帧真值("女声-明亮", 文0, 情)
    情绪位移[情] = dict(帧F0st=d["半音中位"], Δ半音=round(d["半音中位"] - 基f0, 2),
                     包络余弦=round(余弦(ft["包络dB"], 基包), 4), 时长=ft["时长"])
档案F0距 = {d["档对"]: d["帧参数F0半音距"] for d in 对}

总 = time.perf_counter() - t0
结果 = dict(
    元=dict(耗时秒=round(总, 2), 采样率=FS, 档案=list(档案s), 文本=list(文本s),
            尺度=[round(float(z), 3) for z in 尺度]),
    A_格={"%s||%s" % k: {kk: vv for kk, vv in v.items() if kk != "v17"} for k, v in 格.items()},
    A_帧真值={"%s||%s" % k: v for k, v in 真.items()},
    A_档对=对, A_同档换文本=内,
    B_参数敏感性=B1, B_倒谱峰=倒谱, B_旁证=甲,
    C_参数活性=活性, D_增益=增益, E_确定性=确定, F_情绪位移=情绪位移,
)
with open(os.path.join(HERE, "探针结果.json"), "w", encoding="utf-8") as f:
    json.dump(结果, f, ensure_ascii=False, indent=1)

# ───────── 打印 ─────────
print("=" * 92)
print("A. 4 档案 × 3 文本（现成 说()）  12 次生成 %.2fs  ·  帧参数=音节帧真值 / 包络=wav 估计"
      % 耗时A)
print("%-9s %-9s %7s %8s %8s %7s %7s %7s %7s %7s" %
      ("档案", "文本", "帧F0st", "估F0Hz", "估F0st", "谱倾斜", "能量", "时长", "帧数", "实时率"))
for 名 in 档案s:
    for 文 in 文本s:
        g, d = 格[(名, 文)], 真[(名, 文)]
        print("%-9s %-9s %7.2f %8.1f %8.2f %7.2f %7.4f %7.3f %7d %7.3f" %
              (名, 文, d["半音中位"], g["F0中位"], g["半音中位"], g["谱倾斜"],
               g["能量"], g["时长"], d["帧数"], g["实时率"]))
print("\n档案两两（同文本配对均值）：")
for d in 对:
    print("  %-20s 声纹代理距离 %5.2f | 同档跨文本 %5.2f | 判别比 %5.3f | 包络余弦 %7.4f | 帧F0半音距 %5.2f"
          % (d["档对"], d["声纹代理距离"], d["同档跨文本"], d["判别比"],
             d["包络余弦"], d["帧参数F0半音距"]))
print("\n同档案换文本（跨文本基线波动）：")
for d in 内:
    print("  %-9s %-14s 距离 %5.2f | 余弦 %7.4f | 帧F0半音距 %4.2f" %
          (d["档案"], d["文对"], d["声纹代理距离"], d["包络余弦"], d["帧参数F0半音距"]))
print("\nB1. 固定语速下改音色参数 → f1/f2/f3 轨迹 max|Δ|（0 = 共振峰对音色参数完全不敏感）：")
for k, v in B1.items():
    print("  改 %-8s → %s" % (k, "帧数随语速变（轨迹长度变，取值仍文本决定）" if v is None else v))
print("B2. 倒谱包络实测峰（Hz, 相对首个峰 dB）—— 4 档案 / 3 文本：")
for 名 in 档案s:
    print("  %-9s %s" % (名, {k: v for k, v in 倒谱[名].items()}))
print("B3. 5 音节 啊 的倒谱峰 vs 帧参数真值：")
for 名 in 档案s:
    print("  %-9s 帧参数(f1,f2,f3) %s | 倒谱峰 %s" % (名, 甲[名]["帧参数"], 甲[名]["倒谱峰"]))
print("\nC. 8 参数活性（单参数大改；基准=女声-明亮；逐位同=True 即该参数对波形零影响）：")
print("  %-6s %-8s %-8s %8s %9s %8s %8s %8s" %
      ("参数", "新值", "逐位同", "Δ时长", "Δ能量dB", "Δ谱倾斜", "包络余弦", "理论ΔF0st"))
for 名, v in 活性.items():
    print("  %-6s %-8.3f %-8s %8.3f %9.2f %8.3f %8.4f %8.2f" %
          (名, v["值"], v["逐位同"], v["Δ时长"], v["Δ能量dB"], v["Δ谱倾斜"],
           v["包络余弦"], v["Δ帧F0半音"]))
print("\nD. 音量倍 → 峰值归一化是否吃掉音量：")
for k, v in 增益.items():
    print("  %-10s 能量 %.5f 峰值 %.4f 包络余弦 %.4f 指纹 %s" %
          (k, v["能量"], v["峰值"], v["包络余弦"], v["字节"]))
print("\nE. 确定性：%s（wav1=%s wav2=%s；文件名随 发声号 变，指纹不变 ⇒ 身份用指纹）"
      % (确定["两趟同字节"], 确定["wav1"], 确定["wav2"]))
print("\nF. 情绪位移（女声-明亮，同文本）vs 档案位移：")
for 情, v in 情绪位移.items():
    print("  %-4s 帧F0 %.2f st  Δ%+.2f st  包络余弦 %.4f  时长 %.2f" %
          (情, v["帧F0st"], v["Δ半音"], v["包络余弦"], v["时长"]))
print("  对比：女声-明亮|女声-温柔 档案F0距 = %.2f st"
      % 档案F0距["女声-明亮|女声-温柔"])
print("总耗时 %.2fs（预算 ≤10s）；产物 %s" % (总, OUT))
print("=" * 92)
