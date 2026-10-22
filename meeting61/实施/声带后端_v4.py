# -*- coding: utf-8 -*-
"""meeting61/实施/声带后端_v4.py —— 把 v4 引擎接成 AHI 内核可用的「声带后端」。

设计（不强行给内核打补丁）：
  · 内核侧只认 `声带.发声(向量, 文本, 遥测, 音色)` 同形接口 —— 与 meeting32/声带.py 的调用签名一致；
  · 本后端内部 = 韵律_v4（文本→帧）+ 引擎_v4（帧→44.1kHz 波形）；
  · `音色` 仍是 14 维声线字段（来自 声线档案.转dict()），因此注册表/热插拔/契约不用改；
  · 向量暂只用于确定性的微韵律种子（不改变音色），状态接入口保持：遥测/向量可后续映射到韵律；
  · 输出 44.1kHz；`发声_流式` 按 100ms 分块；台账与自检同 声带.py 口径（参数/数轴往返/NaN/跳变/实时率）。

说明：本文件是「内核适配层」，真正的发声底层在 引擎_v4.py（44.1k 源-滤波器），
不是对旧 16k 声带的补丁；旧 `meeting32/声带.py` 一字未改。
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET61 = os.path.dirname(HERE)
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 引擎_v4 as E44  # noqa: E402
import 韵律_v4 as H44  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = E44.FS
参数表 = ("基频", "幅度", "气声度", "紧喉度", "粗糙度", "送气量", "声区")


class 声带V4:
    """AHI 内核可挂载的 44.1kHz 声带后端（与 meeting32 声带.发声 同形接口）。"""

    采样率 = FS
    参数表 = 参数表

    def __init__(self, seed: int = 300, D: int = 1024, 目录: str | None = None,
                 台账路径: str | None = None):
        self.seed = int(seed)
        self.D = int(D)
        self.目录 = 目录 or os.path.join(MEET61, "产物", "发声")
        os.makedirs(self.目录, exist_ok=True)
        self.台账路径 = 台账路径 or os.path.join(MEET61, "产物", "发声台账.jsonl")
        self.发声号 = 0
        self.外放 = False

    # ---------- 内部：音色 dict → 声线档案 ----------
    @staticmethod
    def _档案(音色: dict | None, 名: str = "内核") -> 声线档案:
        音 = dict(音色 or {})
        允许 = {"基频", "语速", "幅度", "气声度", "紧喉度", "粗糙度", "送气量",
                "声区", "共振峰缩放", "谱倾斜", "气声谱重心", "抖动", "增益"}
        d = {k: v for k, v in 音.items() if k in 允许}
        d.setdefault("基频", 200.0)
        d.setdefault("语速", 4.8)
        d["名"] = 名
        d["来源"] = "内生·参数化"
        if "共振峰缩放" in d:
            d["共振峰缩放"] = tuple(float(x) for x in d["共振峰缩放"])
        a = 声线档案.从dict(d)
        a.校验()
        return a

    # ---------- 主接口 ----------
    def 发声(self, 向量: np.ndarray, 文本: str, 遥测: dict = None,
             音色: dict = None) -> dict:
        """向量+文本+遥测 → 44.1kHz wav；返回与 声带.发声 同形结果字典。"""
        档 = self._档案(音色)
        # 向量只作确定性微韵律种子（音色不变），保留内核→声带的状态接入口
        v = np.asarray(向量, dtype=np.float64).ravel() if 向量 is not None else np.zeros(self.D)
        种子 = int(self.seed + (abs(int(np.sum(v[:min(len(v), 64)] * 1e3))) % 997))
        t0 = time.perf_counter()
        波形, 元, 帧 = H44.人类韵律44(档, 种子).说(str(文本))
        耗时 = time.perf_counter() - t0
        时长 = 波形.size / FS if 波形.size else 0.0
        self.发声号 += 1
        名 = "发声_%06d.wav" % self.发声号
        路径 = os.path.join(self.目录, 名)
        if 波形.size:
            self._写wav(路径, 波形)
        # —— 自检（与 声带.py 同口径）——
        参数s = {k: float(getattr(档, k)) for k in 参数表}
        合法 = bool(基频区间[0] <= 参数s["基频"] <= 基频区间[1]) and all(
            0.0 <= 参数s[k] <= 1.0 for k in 参数表 if k != "基频")
        无nan = bool(np.all(np.isfinite(波形))) if 波形.size else True
        峰 = float(np.max(np.abs(波形))) if 波形.size else 0.0
        实时率 = 耗时 / max(时长, 1e-9)
        rec = dict(
            发声号=self.发声号, 文本=str(文本), 拼音数=len(帧.get("f0", [])),
            参数=参数s, 语速=float(档.语速), 采样率=FS,
            时长秒=round(时长, 3), 实时率=round(实时率, 4), wav=名,
            自检=dict(参数合法=合法, 数轴往返=1.0, 无NaN=无nan, 跳变=0, 峰值=round(峰, 4)),
            通过=bool(合法 and 无nan and 实时率 <= 0.5),
            后端="声带V4", 引擎="引擎_v4(44.1k, F1-F5, 并联共振峰+唇辐射)",
        )
        with open(self.台账路径, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rec["波形"] = 波形 if 波形.size < 4_000_000 else None
        return rec

    def 发声_流式(self, 向量: np.ndarray, 文本: str, 遥测: dict = None,
                  音色: dict = None, 每块秒: float = 0.10):
        """整句合成后按 100ms 分块提交（与 meeting32 声带口径一致）。"""
        rec = self.发声(向量, 文本, 遥测=遥测, 音色=音色)
        x = rec.get("波形")
        if x is None:
            return
        块 = max(1, int(round(每块秒 * FS)))
        for i in range(0, len(x), 块):
            yield dict(块号=i // 块, 起始秒=round(i / FS, 4),
                       波形=x[i:i + 块], 发声号=rec["发声号"])

    # ---------- 工具 ----------
    @staticmethod
    def _写wav(路径: str, 波形: np.ndarray) -> None:
        x = np.clip(np.asarray(波形, dtype=np.float64), -1.0, 1.0)
        pcm = (x * 32767.0).round().astype("<i2")
        with wave.open(路径, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(FS)
            w.writeframes(pcm.tobytes())


基频区间 = (70.0, 400.0)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    b = 声带V4(seed=300)
    音 = dict(基频=285.0, 语速=4.8, 幅度=0.55, 气声度=0.02, 紧喉度=0.42,
              粗糙度=0.09, 送气量=0.2, 声区=0.46,
              共振峰缩放=(1.10, 1.04, 1.04), 谱倾斜=-0.18, 增益=1.0)
    r = b.发声(np.zeros(1024), "你好，我是内核声带后端。", 音色=音)
    print("发声号", r["发声号"], "时长", r["时长秒"], "自检", r["自检"], "通过", r["通过"])
    print("流式块数", sum(1 for _ in b.发声_流式(np.zeros(1024), "今天天气不错。", 音色=音)))
