# -*- coding: utf-8 -*-
"""meeting62/实施/谱_亚版本_声库_v7.py —— 谱 → 亚版本 → **声带后端_v7** → wav（纯新增，C14）。

复用 Task 3 的谱管线（`契约_谱与亚版本.py` 数据契约 + `建亚版本`），但把声学底座换成
**v7**（`韵律_v7` 的 F0 颤音 + `引擎_v7` 的单调高支）。相对 Task 3 的 `谱_亚版本_声库.py`：

  · 只读 import：`meeting58/实施/声线库`、`meeting62/实施/{韵律_v7,引擎_v7,声带后端_v7,契约_谱与亚版本}`。
  · 「唱」的两个硬点保持不变：**音高由谱音符决定**、**时长由时值决定**；
    叠加 v7 修① 的 F0 颤音（`颤音Hz=5.8 / 颤音半音=1.0 / 颤音起延迟=0.5s`）⇒ 长音有真颤音。
  · 产物目录默认 `meeting62/产物/谱管线_v7/`（不写 meeting61，C14）。

只有 numpy/scipy + 既有件；无 TTS/RVC/神经声码器（C1/C9/C10）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET62 = os.path.dirname(HERE)
AHI4 = os.path.dirname(MEET62)
MEET61 = os.path.join(AHI4, "meeting61")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))
sys.path.insert(0, os.path.join(MEET61, "实施"))

import 契约_谱与亚版本 as 契  # noqa: E402
import 声线库 as 库  # noqa: E402  只读
import 韵律_v7 as H7  # noqa: E402
import 引擎_v7 as E7  # noqa: E402
import 声带后端_v7 as 后7  # noqa: E402

声带V7 = 后7.声带V7
FS = E7.FS
帧率 = E7.帧率
平滑步 = H7._平滑步


class 谱韵律_v7(H7.人类韵律_v5):
    """韵律_v7 的只读子类：f0 由**谱音符**决定（唱）；长音叠 v7 口径 F0 颤音。"""

    def __init__(self, 档案, 种子: int = 300):
        super().__init__(档案, 种子)
        self.音符f0 = float(self.基准F0)
        self.滑起f0 = None
        self.滑移比例 = 0.16

    def _f0(self, 调, t, j, n, 标点, 重音, 首加成, 微颤相=0.0, 时长=0.12) -> float:
        f0 = float(self.音符f0)
        if self.滑起f0 is not None and t < self.滑移比例:
            w = 平滑步(t / self.滑移比例)
            f0 = float(self.滑起f0) + (f0 - float(self.滑起f0)) * w
        半音 = self._颤音半音(t * 时长, 时长, 微颤相)   # v7 修①：F0 颤音（长音）
        return f0 * 2.0 ** (半音 / 12.0)

    def 建帧_亚版本(self, 亚版本, 换气: bool = True) -> dict:
        self._清()
        if 换气:
            self._换气(0.035, 0.10)
        音节s = 亚版本.音节s
        n = len(音节s)
        前一 = None
        self.滑起f0 = None
        for j, syl in enumerate(音节s):
            if syl.休止:
                self.音符f0 = float(self.基准F0)
                self.滑起f0 = None
                self._静音(float(syl.时长秒))
                continue
            s = dict(声=syl.声, 韵=syl.韵, 调=1, 目标s=syl.目标s,
                     尾=(syl.尾 or None), 拼音=syl.拼音, 命中=True)
            self.音符f0 = float(syl.音高Hz)
            self.滑起f0 = float(前一) if (syl.连音 and 前一 is not None) else None
            目标首 = np.array([s["目标s"][0][0] * self.s[0],
                              s["目标s"][0][1] * self.s[1],
                              s["目标s"][0][2] * self.s[2]])
            f0起 = self._f0(1, 0.0, j, n, "", False, 0.0)
            起始F = self._声母(s, 目标首, f0起)
            微颤相 = float(self.rng.uniform(0.0, 2.0 * np.pi))
            时长倍 = float(syl.时长秒) / max(0.200 * self.缩, 1e-9)
            self._元音(s, j, n, "", 时长倍, False, 微颤相, 0.0, True, 起始F)
            前一 = self.音符f0
        return self.帧


class 谱声带V7:
    协议版本 = 契.契约版本

    def __init__(self, 种子: int = 300, 目录: str | None = None):
        self.种子 = int(种子)
        self.目录 = 目录 or os.path.join(MEET62, "产物", "谱管线_v7")
        os.makedirs(self.目录, exist_ok=True)
        self.后端 = 声带V7(seed=self.种子, 目录=os.path.join(self.目录, "后端输出"),
                          台账路径=os.path.join(self.目录, "后端台账.jsonl"))
        self.号 = 0

    def 唱(self, 亚版本: 契.亚版本, *, 种子: int | None = None, 名: str | None = None) -> dict:
        亚版本.校验()
        种子 = int(self.种子 if 种子 is None else 种子)
        档 = 契.取音色(亚版本.声线引用)
        律 = 谱韵律_v7(档, 种子)
        t0 = time.perf_counter()
        帧 = 律.建帧_亚版本(亚版本)
        波形, 元 = E7.合成(帧, 音色=律.音色(), 种子=种子)
        耗时 = time.perf_counter() - t0
        self.号 += 1
        名 = 名 or ("唱_%s_%03d" % (亚版本.谱指纹, self.号))
        路径 = os.path.join(self.目录, 名 + ".wav")
        声带V7._写wav(路径, 波形)
        with open(路径, "rb") as f:
            wavsha = hashlib.sha256(f.read()).hexdigest()
        样本sha = hashlib.sha256(np.asarray(波形, "<f8").tobytes()).hexdigest()
        时长 = 波形.size / FS if 波形.size else 0.0
        峰 = float(np.max(np.abs(波形))) if 波形.size else 0.0
        rec = dict(名=名, wav=路径, 声线=亚版本.声线引用, 声线指纹=档.指纹(),
                   谱指纹=亚版本.谱指纹, 亚版本指纹=亚版本.指纹(),
                   采样率=FS, 样本数=int(波形.size), 时长秒=round(时长, 4),
                   帧数=len(帧.get("f0", [])), 峰值=round(峰, 4),
                   实时率=round(耗时 / max(时长, 1e-9), 4),
                   wav_sha256=wavsha, 样本sha256=样本sha, 种子=种子,
                   无NaN=bool(np.all(np.isfinite(波形))) if 波形.size else True,
                   音色=dict(音高源="音符Hz", 颤音Hz=H7.颤音Hz, 颤音半音=H7.颤音半音,
                            颤音起延迟=H7.颤音起延迟),
                   目标=[round(float(s.音高Hz), 3) for s in 亚版本.音节s if not s.休止])
        rec["波形"] = 波形 if 波形.size < 8_000_000 else None
        rec["帧"] = 帧
        return rec


def _f0轨(x, sr=FS, 窗=0.040, 跳=0.010):
    窗n = max(int(窗 * sr), 64)
    跳n = max(int(跳 * sr), 1)
    ts, f0s = [], []
    if len(x) < 窗n:
        return np.zeros(0), np.zeros(0)
    lo, hi = int(sr / 1200.0), int(sr / 70.0)
    for a in range(0, len(x) - 窗n, 跳n):
        seg = x[a:a + 窗n]
        seg = seg - seg.mean()
        e = float(np.sqrt(np.mean(seg * seg)))
        if e < 1e-4:
            ts.append(a / sr); f0s.append(0.0); continue
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        if hi >= len(ac):
            continue
        段 = ac[lo:hi]
        if 段.size == 0 or ac[0] <= 0:
            continue
        lag = lo + int(np.argmax(段))
        r = ac[lag] / (ac[0] + 1e-12)
        ts.append(a / sr)
        f0s.append(sr / lag if r >= 0.35 else 0.0)
    return np.asarray(ts), np.asarray(f0s)


def 测音符(波形, 亚版本, 偏移=0.035, 辅音估=0.05) -> list:
    ts, f0s = _f0轨(波形)
    读物 = []
    t = 偏移
    for s in 亚版本.音节s:
        if s.休止:
            t += s.时长秒
            continue
        起 = t + (辅音估 if s.声 else 0.0)
        中 = 起 + s.时长秒 * 0.5
        半宽 = max(s.时长秒 * 0.20, 0.04)
        m = (ts >= 中 - 半宽) & (ts <= 中 + 半宽) & (f0s > 0)
        v = f0s[m]
        实 = float(np.median(v)) if v.size else 0.0
        读物.append(dict(歌词=s.歌词, 期望Hz=round(s.音高Hz, 3), 实测Hz=round(实, 2),
                         音分=round(1200.0 * np.log2(max(实, 1e-6) / s.音高Hz), 1) if 实 > 0 else None,
                         f0标准差Hz=round(float(np.std(v)), 2) if v.size else 0.0,
                         时长秒=round(s.时长秒, 3), 帧数=int(v.size)))
        t += s.时长秒
    return 读物


def 主(输出目录: str | None = None) -> dict:
    目录 = 输出目录 or os.path.join(MEET62, "产物", "谱管线_v7")
    os.makedirs(目录, exist_ok=True)
    谱路 = os.path.join(MEET62, "素材", "小谱_样本.json")
    谱对象 = 契.谱.从JSON文件(谱路)
    亚版本 = 契.建亚版本(谱对象)
    r1 = 谱声带V7(种子=300, 目录=目录).唱(亚版本, 名="谱管线_v7_小谱_样本_趟1")
    r2 = 谱声带V7(种子=300, 目录=目录).唱(亚版本, 名="谱管线_v7_小谱_样本_趟2")
    读数 = 测音符(r1["波形"], 亚版本)
    # 参考 A/B：同一旋律用 v6 引擎渲染（对照）
    亚路 = os.path.join(输出目录 or 目录, "亚版本_v7_小谱_样本.json")
    with open(亚路, "w", encoding="utf-8") as f:
        json.dump(亚版本.转dict(), f, ensure_ascii=False, indent=2)
    rec = dict(契约版本=契.契约版本, 谱=谱对象.转dict(),
               wav=dict(趟1=r1["wav"], 趟2=r2["wav"]),
               确定性=dict(两趟逐位同=bool(r1["wav_sha256"] == r2["wav_sha256"]),
                         趟1_wavsha256=r1["wav_sha256"], 趟2_wavsha256=r2["wav_sha256"],
                         样本逐位同=bool(r1["样本sha256"] == r2["样本sha256"])),
               唱=dict(时长秒=r1["时长秒"], 峰值=r1["峰值"], 帧数=r1["帧数"],
                      实时率=r1["实时率"], 无NaN=r1["无NaN"], 声线=r1["声线"]), f0实测=读数)
    with open(os.path.join(目录, "读数_谱管线_v7.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    t = time.perf_counter()
    r = 主()
    print("=" * 88)
    print("meeting62 Task4 · 谱 → 亚版本 → 声带后端_v7 → wav（唱）")
    print("趟1 wav:", r["wav"]["趟1"])
    print("确定性 两趟逐位同=%s | 样本逐位同=%s" % (r["确定性"]["两趟逐位同"], r["确定性"]["样本逐位同"]))
    print("唱: 时长 %.3fs 峰值 %.3f 实时率 %.4f 无NaN=%s"
          % (r["唱"]["时长秒"], r["唱"]["峰值"], r["唱"]["实时率"], r["唱"]["无NaN"]))
    for x in r["f0实测"]:
        print("  %-2s 期望%8.2fHz 实测%8.2fHz 音分%7.1f f0σ%6.2f" % (x["歌词"], x["期望Hz"], x["实测Hz"], x["音分"] or 0, x["f0标准差Hz"]))
    print("总用时 %.2fs" % (time.perf_counter() - t))
