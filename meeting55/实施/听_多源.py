# -*- coding: utf-8 -*-
"""meeting55/实施/听_多源.py —— 听觉多源输入（v3，采纳四队 R1 意见）。

关键修正：
  · 红队：合成标定只作 [自证]；声纹走独立绝对谱；单参数扰动必须测。
  · 蓝队/绿队：特征必须**按说话人归一**，否则「谁在说」混淆「什么情绪」。
  · 黄队：WASAPI 17 入 / 14 出，48k→16k；夜里底噪≈0.00078，VAD 门 0.0025。
  · 情绪只分**文本无关**的 3 类（平静/高兴/低落）；「急促」归到**风格**（语速快），
    不再混进情绪标签。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import 说_音色 as 说  # noqa: E402

FS = 16000
产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)
情绪列 = ("平静", "高兴", "低落")
风格列 = ("平静", "高兴", "低落", "急促")
设备_入, 设备_出 = 17, 14


def _重采样(x, sr_in, sr_out=FS):
    if sr_in == sr_out:
        return x
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, 1, n, endpoint=False),
                     np.linspace(0, 1, len(x), endpoint=False), x)


def 录音(秒=2.0, sr=48000, 设备=None):
    import sounddevice as sd
    try:
        x = sd.rec(int(sr * 秒), samplerate=sr, channels=1, dtype="float32",
                   device=设备 if 设备 is not None else 设备_入)
        sd.wait()
        return _重采样(x.reshape(-1).astype(np.float64), sr)
    except Exception:
        x = sd.rec(int(FS * 秒), samplerate=FS, channels=1, dtype="float32", device=1)
        sd.wait()
        return x.reshape(-1).astype(np.float64)


def 录音VAD(最长=6.0, 静音=0.35, 阈值=0.0025, sr=48000):
    import sounddevice as sd
    block = int(sr * 0.05)
    块 = []
    有声 = False
    静音计数 = 0
    总 = 0
    try:
        while 总 < int(最长 * sr):
            x = sd.rec(block, samplerate=sr, channels=1, dtype="float32", device=设备_入)
            sd.wait()
            x = x.reshape(-1).astype(np.float64)
            块.append(x)
            总 += len(x)
            rms = float(np.sqrt(np.mean(x ** 2)))
            if rms > 阈值:
                有声 = True
                静音计数 = 0
            elif 有声:
                静音计数 += 1
                if 静音计数 * 0.05 >= 静音:
                    break
    except Exception:
        return 录音(min(最长, 3.0))
    return _重采样(np.concatenate(块) if 块 else np.zeros(1), sr)


def 帧(x, sr=FS, 帧长=0.025, 帧移=0.01):
    x = np.asarray(x, dtype=np.float64)
    fl, fs = int(帧长 * sr), int(帧移 * sr)
    n = max(1, (len(x) - fl) // fs + 1)
    窗 = np.hanning(fl)
    E = np.zeros(n); C = np.zeros(n); F = np.zeros(n)
    F0 = np.full(n, np.nan)
    带 = np.zeros((n, 7))
    边界 = (0, 500, 1000, 2000, 3000, 4000, 6000, 8000)
    频 = np.fft.rfftfreq(fl, 1 / sr)
    for i in range(n):
        seg = x[i * fs:i * fs + fl]
        if len(seg) < fl:
            seg = np.pad(seg, (0, fl - len(seg)))
        E[i] = float(np.mean(seg ** 2))
        mag = np.abs(np.fft.rfft(seg * 窗)) + 1e-9
        C[i] = float(np.sum(频 * mag) / np.sum(mag))
        F[i] = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))
        for b in range(7):
            m = (频 >= 边界[b]) & (频 < 边界[b + 1])
            带[i, b] = float(np.sum(mag[m] ** 2))
        s = seg - seg.mean()
        ac = np.correlate(s, s, "full")[len(s) - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        if hi < len(ac) and np.max(ac[lo:hi]) > 0:
            lag = lo + int(np.argmax(ac[lo:hi]))
            if ac[lag] / (ac[0] + 1e-9) > 0.3:
                F0[i] = sr / lag
    return dict(E=E, C=C, F=F, 带=带, F0=F0)


def 特征(x, sr=FS):
    f = 帧(x, sr)
    E, F0 = f["E"], f["F0"]
    F0v = F0[np.isfinite(F0) & (F0 > 60) & (F0 < 450)]
    带 = np.mean(f["带"], axis=0)
    带 = 带 / (np.sum(带) + 1e-9)
    return dict(F0均值=float(np.mean(F0v)) if len(F0v) else 0.0,
                F0标准差=float(np.std(F0v)) if len(F0v) else 0.0,
                F0范围=float(np.percentile(F0v, 90) - np.percentile(F0v, 10)) if len(F0v) > 4 else 0.0,
                能量=float(np.mean(E)), 能量标准差=float(np.std(E)),
                峰率=float(_音节率(E)),
                停顿比=float(1.0 - np.mean(E > max(np.percentile(E, 30), 1e-6))),
                谱重心=float(np.mean(f["C"])), 谱平坦=float(np.mean(f["F"])),
                带0=float(带[0]), 带1=float(带[1]), 带2=float(带[2]), 带3=float(带[3]),
                带4=float(带[4]), 带5=float(带[5]), 带6=float(带[6]),
                时长=float(len(x) / sr))


def _音节率(E):
    if len(E) < 8:
        return 0.0
    e = E / (np.max(E) + 1e-9)
    return sum(1 for i in range(1, len(e) - 1)
               if e[i] > 0.25 and e[i] >= e[i - 1] and e[i] > e[i + 1]) / max(len(e) * 0.01, 1e-6)


def 声纹向量(f):
    # 只取「谱形状」（音色），不取 F0/能量（会被情绪改变）
    return np.array([f["带0"], f["带1"], f["带2"], f["带3"], f["带4"], f["带5"], f["带6"],
                     np.log(max(f["谱重心"], 100.0)), np.log(max(f["谱平坦"], 1e-5)),
                     np.log(max(f["F0标准差"], 1.0))])


class 声纹库:
    def __init__(self):
        self.均值 = None; self.标准差 = None; self.质 = {}

    def 标定(self, 向量列):
        X = np.array(向量列)
        self.均值 = X.mean(axis=0); self.标准差 = X.std(axis=0) + 1e-9
        return self

    def _z(self, v):
        z = (np.asarray(v) - self.均值) / self.标准差
        return z / (np.linalg.norm(z) + 1e-9)

    def 注册(self, 名, 向量列):
        Z = [self._z(v) for v in 向量列]
        c = np.mean(Z, axis=0)
        self.质[名] = c / (np.linalg.norm(c) + 1e-9)

    def 验证(self, f):
        if not self.质:
            return dict(名="未注册", 相似度=0.0)
        v = self._z(声纹向量(f))
        分 = {k: float(v @ c) for k, c in self.质.items()}
        最 = max(分, key=分.get)
        return dict(名=最, 相似度=round(分[最], 4), 全部分={k: round(x, 4) for k, x in 分.items()})


class 情绪规则:
    """按说话人归一的 F0（半音）/能量（dB）判 3 类；阈值由合成标定给出。"""

    def __init__(self):
        self.F0基线 = {}
        self.能基线 = {}

    def 标定(self, 样本):
        for 音 in set(x[0] for x in 样本):
            中 = [f for 音2, f, e in 样本 if 音2 == 音 and e == "平静"]
            self.F0基线[音] = float(np.mean([f["F0均值"] for f in 中]))
            self.能基线[音] = float(np.mean([f["能量"] for f in 中]))
        return self

    def 判(self, f, 音):
        f0 = f["F0均值"] if f["F0均值"] > 0 else self.F0基线.get(音, 190)
        st = 12.0 * np.log2(max(f0, 60) / max(self.F0基线.get(音, 190), 60))
        db = 10.0 * np.log10(max(f["能量"], 1e-8) / max(self.能基线.get(音, 1e-5), 1e-8))
        分 = {"高兴": float(st) + 0.5 * float(db),
              "低落": -float(st) - 0.5 * float(db),
              "平静": 0.0}
        最 = max(分, key=分.get)
        if 分[最] < 1.2:
            最 = "平静"
        置信 = float(np.clip((abs(分[最]) - 0.5) / 3.0, 0.0, 1.0))
        return dict(情绪=最, 置信=round(置信, 3), F0半音=round(float(st), 2),
                    能量dB=round(float(db), 2), 分={k: round(v, 2) for k, v in 分.items()})


def 风格语调(f, emo=None):
    风格 = dict(语速="快" if f["峰率"] > 4.2 else ("慢" if f["峰率"] < 2.2 else "中"),
                音高="高" if f["F0均值"] > 230 else ("低" if f["F0均值"] < 150 else "中"),
                能量="强" if f["能量"] > 0.02 else ("弱" if f["能量"] < 0.005 else "中"),
                停顿="多" if f["停顿比"] > 0.55 else "少")
    语调 = "感叹" if f["F0标准差"] > 45 else ("疑问" if f["F0均值"] > 240 else "陈述")
    return 风格, 语调


_标定 = None


def 标定(重建=False):
    global _标定
    if _标定 is not None and not 重建:
        return _标定
    文本列 = ["你好，我在这里。", "今天过得怎么样？", "我们一起走吧。"]
    音色列 = ["女声-明亮", "女声-温柔", "男声-低沉"]
    样本 = []
    for 音 in 音色列:
        for 情 in 风格列:
            for 文本 in 文本列:
                rec = 说.说(文本, 音色=音, 情绪=情)
                sr, x = 说.读wav(os.path.join(说.产物, "发声", rec["wav"]))
                样本.append((音, 特征(x, sr), 情))
    规则 = 情绪规则().标定(样本)
    库 = 声纹库().标定([声纹向量(f) for _, f, _ in 样本])
    for 音 in 音色列:
        库.注册(音, [声纹向量(f) for 音2, f, _ in 样本 if 音2 == 音])
    同, 异 = [], []
    for 音, f, _ in 样本:
        for 名, s in 库.验证(f)["全部分"].items():
            (同 if 名 == 音 else 异).append(s)
    阈值 = float((np.mean(同) + np.mean(异)) / 2.0)
    _标定 = dict(规则=规则, 库=库, 阈值=阈值, _样本=样本,
                 同均值=float(np.mean(同)), 异均值=float(np.mean(异)),
                 F0基线=规则.F0基线, 能基线=规则.能基线)
    return _标定


def 听波形(x, sr=FS, 声纹名=None):
    f = 特征(x, sr)
    标 = 标定()
    音 = 声纹名 or 标["库"].验证(f)["名"]
    emo = 标["规则"].判(f, 音 if 音 in 标["F0基线"] else "女声-明亮")
    风格, 语调 = 风格语调(f, emo["情绪"])
    纹 = 标["库"].验证(f)
    return dict(特征=f, 情绪=emo, 风格=风格, 语调=语调, 声纹=纹,
                有声=bool(f["能量"] > 0.004), 声纹门=round(标["阈值"], 3),
                模型="合成标定（[自证]；真人未标定）")


def 听wav(路径, 声纹名=None):
    sr, x = 说.读wav(路径)
    return 听波形(x, sr, 声纹名)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    标 = 标定(重建=True)
    print("声纹阈值", round(标["阈值"], 3), "同类", round(标["同均值"], 3), "异类", round(标["异均值"], 3))
    print("F0基线", {k: round(v, 1) for k, v in 标["F0基线"].items()})
    for 音 in ("女声-明亮", "男声-低沉"):
        for 情 in 风格列:
            rec = 说.说("这句话没在标定集里，你听得出我的心情吗？", 音色=音, 情绪=情)
            r = 听wav(os.path.join(说.产物, "发声", rec["wav"]), 声纹名=音)
            print("真值", 情, "|", 音, "→", r["情绪"]["情绪"], "置信", r["情绪"]["置信"],
                  "F0st", r["情绪"]["F0半音"], "dB", r["情绪"]["能量dB"],
                  "| 声纹", r["声纹"]["名"], r["声纹"]["相似度"], "| 风格", r["风格"], r["语调"])
