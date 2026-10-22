# -*- coding: utf-8 -*-
"""meeting32/耳朵.py —— #320 耳朵器官：声音→向量→声觉学习（判据先定、纯 numpy）。

回应遗留项「声带只有口没有耳」：让 AHI 能听见（含听见自己说的话），
并把声音学进声觉舱——听到 → 特征帧 → HDC 声向量 → 与文本/情绪关联存档 →
即学即检索（无迭代训练，计数关联同律）。

双通道架构（声音学习的核心设计）：
  内容通道（声向量）：频带包络的槽均值→log→分位归一→HDC 槽位绑定；
    F0 用相对半音域（情绪基频偏移=韵律，不进内容通道；音高轮廓=声调=内容）；
    log-after-average（先槽内平均功率再取对数）抗噪。
  情绪通道（韵律指纹）：原始 F0 统计 + 能量 + 谱平坦度（气声代理）+ 时长。
确定性：固定种子 sha256 导出，同音频两次编码逐位一致（L5）。
"""
from __future__ import annotations

import hashlib
import json
import os
import wave

import numpy as np

from 声带 import 数轴, FS

特征表 = ("F0轮廓", "能量", "带0", "带1", "带2", "带3", "带4", "带5", "带6")
带边界 = (0, 500, 1000, 2000, 3000, 4000, 6000, 8000)
# 注：8kHz 以上频带不取（语音内容 <8kHz；白噪声会主导该带，破坏抗噪判据）


def _基置换(键: str, D: int) -> np.ndarray:
    h = hashlib.sha256(("耳|%s" % 键).encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.permutation(D)


class 耳朵器官:
    """耳朵：听(波形/wav) → 声向量+韵律指纹；学/辨 → 声觉舱。"""

    def __init__(self, seed: int = 400, D: int = 1024,
                 帧长秒: float = 0.025, 帧移秒: float = 0.010):
        self.seed = int(seed)
        self.D = int(D)
        self.帧长 = int(帧长秒 * FS)
        self.帧移 = int(帧移秒 * FS)
        self.槽数 = 8
        # 内容通道数轴：F0轮廓/能量=粗档16级（幅高不是内容，轮廓才是）；
        # 频带=细档256级（共振峰包络是内容主体）
        self.轴s = {
            "F0轮廓": 数轴("耳|F0轮廓", D, S1=4, S2=4, 下界=0.0, 上界=1.0),
            "能量": 数轴("耳|能量", D, S1=4, S2=4, 下界=0.0, 上界=1.0),
        }
        for b in range(7):
            self.轴s["带%d" % b] = 数轴("耳|带%d" % b, D, 下界=0.0, 上界=1.0)
        self._位置表: dict = {}              # 槽号→置换（惰性）
        self.舱: list = []                   # 条目= dict(声向量,文本,情绪码,指纹)
        self.听次 = 0

    # ---------- 特征提取（原始量，不做逐句归一） ----------
    def 特征帧(self, 波形: np.ndarray):
        """波形→dict(功率(n,7)线性频带功率, 能量(n), F0(n)Hz原始, 平坦度(n))。"""
        x = np.asarray(波形, dtype=np.float64)
        n = max((len(x) - self.帧长) // self.帧移 + 1, 1)
        功率 = np.zeros((n, 7))
        能量 = np.zeros(n)
        F0 = np.zeros(n)
        平坦度 = np.zeros(n)
        滞后低, 滞后高 = int(FS / 400), int(FS / 70)
        freq = np.fft.rfftfreq(self.帧长, 1.0 / FS)
        for t in range(n):
            seg = x[t * self.帧移: t * self.帧移 + self.帧长]
            if len(seg) < self.帧长:
                seg = np.pad(seg, (0, self.帧长 - len(seg)))
            能量[t] = float(np.mean(seg ** 2))
            mag = np.abs(np.fft.rfft(seg * np.hanning(self.帧长)))
            mag2 = mag ** 2
            for b in range(7):
                lo, hi = 带边界[b], 带边界[b + 1]
                m = (freq >= lo) & (freq < hi)
                功率[t, b] = float(np.sum(mag2[m]))
            p = mag2[:int(len(mag) * 0.72)]          # 0-8kHz
            p = p / (np.sum(p) + 1e-9)
            平坦度[t] = float(np.exp(np.mean(np.log(p + 1e-9))) /
                              (np.mean(p) + 1e-9))
            s = seg - seg.mean()
            ac = np.correlate(s, s, "full")[self.帧长 - 1:]
            if ac[0] > 1e-12:
                ac /= ac[0]
                窗 = ac[滞后低:滞后高]
                峰 = int(np.argmax(窗)) + 滞后低
                F0[t] = FS / 峰 if ac[峰] > 0.3 else 0.0
        # F0 中值平滑（窗5）：稳住孤帧清浊翻转（标准基音追踪后处理）
        平滑 = F0.copy()
        for t in range(n):
            邻 = F0[max(t - 2, 0):t + 3]
            邻浊 = 邻[邻 > 0]
            if F0[t] == 0 and len(邻浊) >= 2:
                平滑[t] = float(np.median(邻浊))
            elif F0[t] > 0 and len(邻浊) >= 1:
                邻s = np.sort(邻浊)
                if abs(F0[t] - np.median(邻s)) > 0.5 * max(np.median(邻s), 1):
                    平滑[t] = float(np.median(邻浊))     # 孤立倍频/半频纠偏
        return dict(功率=功率, 能量=能量, F0=平滑, 平坦度=平坦度)

    def _位置置换(self, k: int) -> np.ndarray:
        p = self._位置表.get(k)
        if p is None:
            p = _基置换("pos%d" % k, self.D)
            self._位置表[k] = p
        return p

    # ---------- 内容通道编码 ----------
    def 声向量(self, 特征: dict) -> np.ndarray:
        """槽均值→归一→HDC 槽位绑定累加。log-after-average 抗噪。"""
        功率, 能量, F0 = 特征["功率"], 特征["能量"], 特征["F0"]
        n = len(能量)
        累积 = np.zeros(self.D, dtype=np.float64)
        浊F0 = F0[F0 > 0]
        基 = float(np.median(浊F0)) if len(浊F0) else 1.0
        for k in range(self.槽数):
            lo = int(n * k / self.槽数)
            hi = max(int(n * (k + 1) / self.槽数), lo + 1)
            带均值 = 功率[lo:hi].mean(axis=0)
            带对数 = np.log(带均值 + 1e-12)
            lo9, hi9 = (float(np.quantile(带对数, 0.02)),
                        float(np.quantile(带对数, 0.98)))
            带归一 = np.clip((带对数 - lo9) / max(hi9 - lo9, 1e-9), 0.0, 1.0)
            能 = float(np.mean(能量[lo:hi]))
            能n = 能 / (float(np.max(能量)) + 1e-12)      # 包络形状（逐句归一）
            槽F0 = F0[lo:hi]
            浊槽 = 槽F0[槽F0 > 0]
            if len(浊槽):
                st = 12.0 * np.log2(np.mean(浊槽) / max(基, 1e-3))
                f0v = float(np.clip((st + 6.0) / 12.0, 0.0, 1.0))
            else:
                f0v = 0.0                                  # 全清音槽
            v = np.zeros(self.D, dtype=np.float64)
            v += self.轴s["F0轮廓"].编码向量(f0v)
            v += self.轴s["能量"].编码向量(能n)
            for b in range(7):
                v += self.轴s["带%d" % b].编码向量(float(带归一[b]))
            累积 += v[self._位置置换(k)]
        return 累积.astype(np.float32)

    # ---------- 情绪通道 ----------
    def 韵律指纹(self, 特征: dict, 时长秒: float) -> tuple:
        """8 维韵律指纹：原始F0均值/std/十分位距 + 能量均值/std + 平坦度均值/std + 时长。
        使用方需 z-score 标准化后再比近邻。"""
        F0, 能量, 平 = 特征["F0"], 特征["能量"], 特征["平坦度"]
        浊 = F0[F0 > 0]
        if len(浊) < 3:
            浊 = np.array([0.0])
        return (float(np.mean(浊)),
                float(np.std(浊)),
                float(np.quantile(浊, 0.9) - np.quantile(浊, 0.1)),
                float(np.mean(能量)),
                float(np.std(能量)),
                float(np.mean(平)),
                float(np.std(平)),
                float(时长秒))

    # ---------- 听 ----------
    def 听(self, 源, 路径: str = None) -> dict:
        """源 = 波形数组 或 wav 路径。"""
        self.听次 += 1
        if isinstance(源, (str, os.PathLike)) or (源 is None and 路径):
            with wave.open(str(源 or 路径), "rb") as w:
                assert w.getframerate() == FS and w.getnchannels() == 1
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            波形 = pcm.astype(np.float64) / 32767.0
        else:
            波形 = np.asarray(源, dtype=np.float64)
        特征 = self.特征帧(波形)
        return dict(特征=特征, 声向量=self.声向量(特征),
                    指纹=self.韵律指纹(特征, len(波形) / FS), 波形=波形)

    # ---------- 声觉学习 ----------
    def 学(self, 听到: dict, 文本: str, 情绪码: str = "") -> int:
        """听到即学：声向量+文本+情绪+指纹入舱（即学即用，无迭代训练）。"""
        self.舱.append(dict(声向量=听到["声向量"], 文本=文本,
                            情绪码=情绪码, 指纹=听到["指纹"]))
        return len(self.舱) - 1

    def 辨(self, 听到: dict, k: int = 1) -> list:
        """最近邻检索：听到→最像的学过的声音（返回 文本/情绪/相似度）。"""
        q = 听到["声向量"]
        qn = float(np.linalg.norm(q)) + 1e-9
        打分s = []
        for i, 条 in enumerate(self.舱):
            v = 条["声向量"]
            cos = float(q @ v) / (qn * float(np.linalg.norm(v)) + 1e-9)
            打分s.append((cos, i))
        打分s.sort(reverse=True)
        return [dict(文本=self.舱[i]["文本"], 情绪码=self.舱[i]["情绪码"],
                     相似度=round(c, 4), 条目=i) for c, i in 打分s[:k]]

    def 辨情绪(self, 听到: dict) -> str:
        """韵律指纹最近邻→情绪码（内容无关的嗓音状态辨识）。"""
        q = np.asarray(听到["指纹"], dtype=np.float64)
        最佳, 最佳距 = "", 1e18
        for 条 in self.舱:
            d = float(np.sum((np.asarray(条["指纹"]) - q) ** 2))
            if d < 最佳距:
                最佳距, 最佳 = d, 条["情绪码"]
        return 最佳

    # ---------- 持久化 ----------
    def 存档(self, 路径: str) -> None:
        with open(路径, "w", encoding="utf-8") as f:
            json.dump(dict(seed=self.seed, D=self.D,
                           舱=[dict(声向量=条["声向量"].tolist(),
                                    文本=条["文本"], 情绪码=条["情绪码"],
                                    指纹=list(条["指纹"])) for 条 in self.舱]),
                      f, ensure_ascii=False)

    def 载档(self, 路径: str) -> None:
        d = json.load(open(路径, encoding="utf-8"))
        self.seed, self.D = d["seed"], d["D"]
        self.舱 = [dict(声向量=np.asarray(t["声向量"], np.float32),
                        文本=t["文本"], 情绪码=t["情绪码"],
                        指纹=tuple(t["指纹"])) for t in d["舱"]]
