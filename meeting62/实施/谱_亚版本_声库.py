# -*- coding: utf-8 -*-
"""meeting62/实施/谱_亚版本_声库.py —— 最小可跑链路：小谱 → 亚版本 → v6 声带后端 → wav。

链路（Task 3）：
    md谱(JSON/简谱)  --契约_谱与亚版本.建亚版本-->  亚版本(音节序列)  --谱韵律-->  帧  --引擎_v6-->  44.1k 波形  -->  wav

「唱」而非「念」的两个硬点（对齐任务要求）：
  1) **音高由谱的音符决定**：`谱韵律` 覆写 韵律_v6.__f0，忽略歌词声调，f0 直接取自音符 Hz。
  2) **时长由时值决定**：按 `音符.时值秒()` 反算 韵律_v6 的时长倍，使元音段时长 = 音符时值。
  另：连音⇒自上一音符 f0 滑入(portamento)；长音(时长≥0.80s)⇒叠加 5.2Hz 颤音(vibrato)。

与既有件的关系（C14：meeting32/58/59/60/61 一字不改）：
  · 只读 import：`meeting58/实施/声线库`、`meeting61/实施/声带后端_v6/韵律_v6/引擎_v6/音韵_v5`。
  · **调用现有声带后端**：用 `声带后端_v6.声带V6` 负责 (a) 声线档案规整口径、(b) 44.1k wav 写盘；
    并**真实调用** `声带V6.发声(向量,文本,遥测,音色)` 作为「文本态对照」。
    注意：`发声()` 的主入口只吃**文本**、音高随**声调**走 ⇒ 不支持「音符定音高/时值/颤音」；
    故**谱驱动**的帧由 `韵律_v6.人类韵律_v5` 的子类 `谱韵律` 生成（同族构件，隶属同一 v6 后端），
    再交 `引擎_v6.合成`(即 `声带后端_v6` 内部所用引擎) 渲染。此即任务所述「能支持的最小方式」。
  · 后端产物目录/台账**重定向到 meeting62**，绝不写进 meeting61（避免越界改动）。
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
AHI4 = os.path.dirname(os.path.dirname(HERE))
MEET61 = os.path.join(AHI4, "meeting61")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))
sys.path.insert(0, os.path.join(MEET61, "实施"))

import 契约_谱与亚版本 as 契  # noqa: E402
import 声线库 as 库  # noqa: E402  只读
import 声带后端_v6 as 后  # noqa: E402  只读（真实后端）
import 韵律_v6 as H6  # noqa: E402  只读（后端内部韵律）
import 引擎_v6 as E6  # noqa: E402  只读（后端内部引擎）

声带V6 = 后.声带V6
FS = E6.FS
帧率 = E6.帧率
平滑步 = H6._平滑步


class 谱韵律(H6.人类韵律_v5):
    """韵律_v6 的**只读子类**：把 f0 从「声调驱动」改成「音符驱动」（唱）。"""

    def __init__(self, 档案, 种子: int = 300, 颤音深: float = 0.35, 颤音Hz: float = 5.2,
                 颤音起秒: float = 0.80):
        super().__init__(档案, 种子)
        self.音符f0 = float(self.基准F0)
        self.滑起f0 = None
        self.颤音深 = float(颤音深)      # 半音
        self.颤音Hz = float(颤音Hz)
        self.颤音起秒 = float(颤音起秒)  # 仅 ≥此秒的长音叠颤音
        self.滑移比例 = 0.16             # 连音滑移占该音前段的比例(≈60–100ms)

    def _f0(self, 调, t, j, n, 标点, 重音, 首加成, 微颤相=0.0, 时长=0.12) -> float:
        """忽略 `调`（声调）；f0 由 `self.音符f0` 决定；连音滑入 + 长音颤音。"""
        f0 = float(self.音符f0)
        if self.滑起f0 is not None and t < self.滑移比例:
            w = 平滑步(t / self.滑移比例)
            f0 = float(self.滑起f0) + (f0 - float(self.滑起f0)) * w
        半音 = 0.0
        if 时长 >= self.颤音起秒 and 0.22 <= t <= 0.96:
            半音 += self.颤音深 * float(np.sin(2.0 * np.pi * self.颤音Hz * t * 时长 + 微颤相))
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


class 谱声带:
    """谱 → 亚版本 → v6 后端 → wav（最小可跑链路）。"""

    协议版本 = 契.契约版本

    def __init__(self, 种子: int = 300, 目录: str | None = None):
        self.种子 = int(种子)
        self.目录 = 目录 or os.path.join(MEET62, "产物", "谱管线")
        os.makedirs(self.目录, exist_ok=True)
        # 真实后端实例；产物目录/台账重定向到 meeting62（C14：不写 meeting61）
        self.后端 = 声带V6(seed=self.种子, 目录=os.path.join(self.目录, "后端输出"),
                          台账路径=os.path.join(self.目录, "后端台账.jsonl"))
        self.号 = 0

    def 唱(self, 亚版本: 契.亚版本, *, 种子: int | None = None, 名: str | None = None) -> dict:
        亚版本.校验()
        种子 = int(self.种子 if 种子 is None else 种子)
        档 = 契.取音色(亚版本.声线引用)          # 声库只读
        律 = 谱韵律(档, 种子)
        t0 = time.perf_counter()
        帧 = 律.建帧_亚版本(亚版本)
        波形, 元 = E6.合成(帧, 音色=律.音色(), 种子=种子)
        耗时 = time.perf_counter() - t0
        self.号 += 1
        名 = 名 or ("唱_%s_%03d" % (亚版本.谱指纹, self.号))
        路径 = os.path.join(self.目录, 名 + ".wav")
        声带V6._写wav(路径, 波形)               # 后端同款写盘
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
                   音色=dict(音高源="音符Hz", 颤音深半音=律.颤音深, 颤音Hz=律.颤音Hz,
                            颤音起秒=律.颤音起秒),
                   目标=[round(float(s.音高Hz), 3) for s in 亚版本.音节s if not s.休止])
        rec["波形"] = 波形 if 波形.size < 8_000_000 else None
        return rec


# ───────────────────────────── f0 实测（自波形自相关；不依赖元信息） ─────────────────────────────
def _f0轨(x: np.ndarray, sr: int = FS, 窗=0.040, 跳=0.010):
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


def _测音符f0(波形, 亚版本, 偏移=0.035, 辅音估=0.05) -> list:
    """按音符标称时窗取中位 f0（辅音前段不计）。返回每音符 {期望Hz, 实测Hz, 比}。"""
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
                         比=round(实 / s.音高Hz, 4) if 实 > 0 else 0.0,
                         f0标准差Hz=round(float(np.std(v)), 2) if v.size else 0.0,
                         帧数=int(v.size),
                         窗秒=[round(中 - 半宽, 3), round(中 + 半宽, 3)]))
        t += s.时长秒
    return 读物


# ───────────────────────────── 主流程 ─────────────────────────────
def 主(输出目录: str | None = None) -> dict:
    目录 = 输出目录 or os.path.join(MEET62, "产物", "谱管线")
    os.makedirs(目录, exist_ok=True)
    谱路 = os.path.join(MEET62, "素材", "小谱_样本.json")
    谱对象 = 契.谱.从JSON文件(谱路)
    亚版本 = 契.建亚版本(谱对象)

    # 亚版本落盘（样例）
    亚路 = os.path.join(目录, "亚版本_%s.json" % 谱对象.名)
    with open(亚路, "w", encoding="utf-8") as f:
        json.dump(亚版本.转dict(), f, ensure_ascii=False, indent=2)

    # —— 两趟独立重跑（不同实例、同输入、同种子）→ 逐位同 ——
    r1 = 谱声带(种子=300, 目录=目录).唱(亚版本, 名="谱管线_小谱_样本_趟1")
    r2 = 谱声带(种子=300, 目录=目录).唱(亚版本, 名="谱管线_小谱_样本_趟2")

    读数 = _测音符f0(r1["波形"], 亚版本)

    # —— 文本态对照：真实调用现有后端 声带V6.发声(向量,文本,遥测,音色) ——
    档 = 契.取音色(亚版本.声线引用)
    后端 = 声带V6(seed=300, 目录=os.path.join(目录, "后端输出"),
                  台账路径=os.path.join(目录, "后端台账.jsonl"))
    对 = 后端.发声(np.zeros(1024), "啦啦啦啊", 遥测=None, 音色=档.转dict())
    对路 = 对.get("wav")
    对波 = 对.get("波形")
    对f0 = None
    if 对波 is not None and len(对波):
        ts, f0s = _f0轨(对波)
        v = f0s[f0s > 0]
        对f0 = round(float(np.median(v)), 2) if v.size else None

    # —— 简谱解析旁证（同一旋律的简谱形态） ——
    简谱文本 = "# 拍速=100\n5 la 1\n6 la 1\n7 la 1 +\n6 ao 2\n0 r 1\n"
    简谱对象 = 契.谱.从简谱(简谱文本, 声线引用="EmoCompanion", 名="小谱_样本_简谱旁证")
    简亚 = 契.建亚版本(简谱对象)

    rec = dict(
        契约版本=契.契约版本, 谱=谱对象.转dict(), 亚版本=亚版本.转dict(),
        wav=dict(趟1=r1["wav"], 趟2=r2["wav"]),
        确定性=dict(趟1_wavsha256=r1["wav_sha256"], 趟2_wavsha256=r2["wav_sha256"],
                  两趟逐位同=bool(r1["wav_sha256"] == r2["wav_sha256"]),
                  趟1_样本sha256=r1["样本sha256"], 趟2_样本sha256=r2["样本sha256"],
                  样本逐位同=bool(r1["样本sha256"] == r2["样本sha256"]),
                  亚版本指纹=亚版本.指纹(), 亚版本重算指纹=契.建亚版本(谱对象).指纹()),
        唱=dict(趟1=dict(采样率=r1["采样率"], 样本数=r1["样本数"], 时长秒=r1["时长秒"],
                        帧数=r1["帧数"], 峰值=r1["峰值"], 实时率=r1["实时率"],
                        声线=r1["声线"], 声线指纹=r1["声线指纹"], 无NaN=r1["无NaN"],
                        音色=r1["音色"]),
               趟2=dict(时长秒=r2["时长秒"], 峰值=r2["峰值"], 实时率=r2["实时率"])),
        f0实测=读数,
        文本态对照=dict(wav=对路, 文本="啦啦啦啊", 中位f0Hz=对f0,
                      通过=对.get("通过"), 自检=对.get("自检"),
                      时长秒=对.get("时长秒"), 实时率=对.get("实时率"),
                      说明="发声()主入口：音高随声调，不随谱；与谱驱动音高对照"),
        简谱旁证=dict(音符数=len(简谱对象.音符s), 亚版本指纹=简亚.指纹(),
                    音高Hz=[round(float(s.音高Hz), 2) for s in 简亚.音节s if not s.休止]),
    )
    with open(os.path.join(目录, "读数_谱管线.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    rec["亚版本路径"] = 亚路
    rec["谱路径"] = 谱路
    return rec


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    t = time.perf_counter()
    r = 主()
    print("=" * 88)
    print("meeting62 Task3 · 谱 → 亚版本 → 声带后端(v6) → wav")
    print("=" * 88)
    print("谱:", r["谱"]["名"], "拍速", r["谱"]["拍速"], "声线", r["谱"]["声线引用"],
          "总时长", r["谱"]["总时长秒"], "s | 谱指纹", r["谱"]["指纹"])
    print("亚版本: 音节", r["亚版本"]["音节数"], "时长", r["亚版本"]["总时长秒"],
          "s | 指纹", r["亚版本"]["指纹"])
    print("趟1 wav:", r["wav"]["趟1"])
    print("趟2 wav:", r["wav"]["趟2"])
    d = r["确定性"]
    print("确定性 两趟逐位同=%s | 样本逐位同=%s | 亚版本重算同=%s"
          % (d["两趟逐位同"], d["样本逐位同"],
             d["亚版本指纹"] == d["亚版本重算指纹"]))
    print("唱: 时长 %.3fs 峰值 %.3f 帧数 %d 实时率 %.4f 无NaN=%s"
          % (r["唱"]["趟1"]["时长秒"], r["唱"]["趟1"]["峰值"], r["唱"]["趟1"]["帧数"],
             r["唱"]["趟1"]["实时率"], r["唱"]["趟1"]["无NaN"]))
    print("-" * 88)
    print("f0 实测（音符决定音高）:")
    for x in r["f0实测"]:
        print("  %-2s 期望 %7.2f Hz | 实测 %7.2f Hz | 比 %.4f | f0σ %.2f Hz"
              % (x["歌词"], x["期望Hz"], x["实测Hz"], x["比"], x["f0标准差Hz"]))
    print("文本态对照(发声):", r["文本态对照"]["wav"], "中位f0", r["文本态对照"]["中位f0Hz"],
          "Hz 通过", r["文本态对照"]["通过"], "自检", r["文本态对照"]["自检"])
    print("简谱旁证:", r["简谱旁证"]["音高Hz"], "指纹", r["简谱旁证"]["亚版本指纹"])
    print("读数:", os.path.join(os.path.dirname(r["wav"]["趟1"]), "读数_谱管线.json"))
    print("总用时 %.2fs" % (time.perf_counter() - t))
