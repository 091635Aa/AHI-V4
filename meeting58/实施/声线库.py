# -*- coding: utf-8 -*-
"""meeting58/实施/声线库.py —— 角色声线档案库（声层数据；纯数据 + 定制估计，不含渲染）。

口径（不越宪）：
  · 洛天依/初音未来/MOSS 三档来自**公开资料的特征归纳**（[文献类比]），是**参数化模仿**，
    不是本人声纹、不是声克隆；对外只能说「参数化角色声线」。
  · EmoCompanion=项目自有角色（内生参数化）。
  · 定制=从一段参考录音估参数（[外采·参考录音]，只估参数，不复制波形；回放必被判 FAIL）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from 接口契约_角色声线 import 声线档案, 契约错误, E1006  # noqa: E402

内置 = {
    "洛天依": 声线档案(
        名="洛天依", 来源="外采·公开资料", 基频=265.0, 语速=3.7, 幅度=0.55,
        气声度=0.20, 紧喉度=0.26, 粗糙度=0.04, 送气量=0.20, 声区=0.10,
        共振峰缩放=(1.14, 1.18, 1.22), 谱倾斜=0.30, 气声谱重心=0.55, 抖动=0.04,
        备注="公开资料特征归纳（明亮少女声线），参数化模仿，非本人"),
    "初音未来": 声线档案(
        名="初音未来", 来源="外采·公开资料", 基频=360.0, 语速=3.9, 幅度=0.52,
        气声度=0.12, 紧喉度=0.20, 粗糙度=0.03, 送气量=0.16, 声区=0.32,
        共振峰缩放=(1.26, 1.30, 1.30), 谱倾斜=0.45, 气声谱重心=0.50, 抖动=0.02,
        备注="公开资料特征归纳（清澈高音声线），参数化模仿，非本人"),
    "MOSS": 声线档案(
        名="MOSS", 来源="外采·公开资料", 基频=95.0, 语速=2.6, 幅度=0.62,
        气声度=0.06, 紧喉度=0.55, 粗糙度=0.08, 送气量=0.15, 声区=0.02,
        共振峰缩放=(0.82, 0.80, 0.78), 谱倾斜=-0.35, 气声谱重心=0.40, 抖动=0.02,
        备注="《流浪地球》550W 行为参考的播报腔声线（公开剧情摘要，非台词/非本人）"),
    "EmoCompanion": 声线档案(
        名="EmoCompanion", 来源="内生·参数化", 基频=200.0, 语速=3.2, 幅度=0.50,
        气声度=0.30, 紧喉度=0.18, 粗糙度=0.05, 送气量=0.30, 声区=0.05,
        共振峰缩放=(0.92, 0.94, 0.96), 谱倾斜=0.00, 气声谱重心=0.60, 抖动=0.05,
        备注="项目自有角色（打标日记人设）；内生参数化声线"),
    "中性": 声线档案(
        名="中性", 来源="内生·参数化", 基频=150.0, 语速=3.2, 幅度=0.50,
        气声度=0.10, 紧喉度=0.30, 粗糙度=0.05, 送气量=0.20, 声区=0.02,
        共振峰缩放=(0.90, 0.90, 0.90), 谱倾斜=-0.10, 气声谱重心=0.5, 抖动=0.0,
        备注="基线声线"),
}


外部目录 = os.path.join(HERE, "产物", "声线档案")


def 取(名: str) -> 声线档案:
    if 名 in 内置:
        a = 内置[名]
        a.校验()
        return a
    p = os.path.join(外部目录, 名 + ".json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            a = 声线档案.从dict(json.load(f))
        a.校验()
        return a
    raise 契约错误(E1006, "未知声线 %r（已知：%s）"
                   % (名, "/".join(list(内置) + _外部名单())))


def 存(档案: 声线档案, 目录: str | None = None) -> str:
    档案.校验()
    目录 = 目录 or 外部目录
    os.makedirs(目录, exist_ok=True)
    p = os.path.join(目录, 档案.名 + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dict(档案.转dict(), 指纹=档案.指纹()), f, ensure_ascii=False, indent=1)
    return p


def _外部名单() -> list:
    if not os.path.isdir(外部目录):
        return []
    return sorted(x[:-5] for x in os.listdir(外部目录) if x.endswith(".json"))


def 全部() -> dict:
    d = dict(内置)
    for 名 in _外部名单():
        try:
            d[名] = 取(名)
        except Exception:
            continue
    return d


def 导出(目录: str | None = None) -> list:
    目录 = 目录 or os.path.join(HERE, "产物", "声线档案")
    os.makedirs(目录, exist_ok=True)
    路 = []
    for 名, a in 内置.items():
        p = os.path.join(目录, "%s.json" % 名)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dict(a.转dict(), 指纹=a.指纹()), f, ensure_ascii=False, indent=1)
        路.append(p)
    return 路


# ───────────────────────────── 定制：从参考录音估参数 ─────────────────────────────
def _读wav(路径: str):
    from scipy.io import wavfile
    sr, x = wavfile.read(路径)
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if np.max(np.abs(x)) > 1.0:
        x = x / 32768.0
    if sr != 16000:
        n = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)
        sr = 16000
    return sr, x


def _F0轨迹(x: np.ndarray, sr: int = 16000, 帧长=0.04, 帧移=0.01) -> np.ndarray:
    n = max(1, int((len(x) - int(帧长 * sr)) // int(帧移 * sr) + 1))
    F0 = np.full(n, np.nan)
    for i in range(n):
        seg = x[i * int(帧移 * sr): i * int(帧移 * sr) + int(帧长 * sr)]
        if len(seg) < 100:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        if hi >= len(ac):
            continue
        段 = ac[lo:hi]
        if np.max(段) <= 0:
            continue
        lag = lo + int(np.argmax(段))
        if ac[lag] / (ac[0] + 1e-9) < 0.30:
            continue
        F0[i] = sr / lag
    return F0


def 从参考估(参考wav: str, 名: str = "定制", 备注: str = "") -> dict:
    """只估参数（F0/能量/谱重心/平坦度/亮度→ 8+4 维），不复制波形、不做拼接。"""
    sr, x = _读wav(参考wav)
    if len(x) < sr // 2:
        raise 契约错误(E1006, "参考录音太短：%s" % 参考wav)
    F0 = _F0轨迹(x)
    F0v = F0[np.isfinite(F0) & (F0 > 60) & (F0 < 500)]
    f0 = float(np.median(F0v)) if len(F0v) else 190.0
    seg = x[:min(len(x), sr * 3)]
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    频 = np.fft.rfftfreq(len(seg), 1.0 / sr)
    重心 = float(np.sum(频 * mag) / (np.sum(mag) + 1e-12))
    平坦 = float(np.exp(np.mean(np.log(mag + 1e-9))) / (np.mean(mag) + 1e-9))
    能量 = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    sha = hashlib.sha256(open(参考wav, "rb").read()).hexdigest()[:16] if os.path.getsize(参考wav) < 50_000_000 else "大文件未hash"
    参 = dict(
        基频=float(np.clip(f0, 70.0, 400.0)),
        语速=3.2,
        幅度=float(np.clip(0.30 + 4.0 * 能量, 0.3, 0.8)),
        气声度=float(np.clip(0.08 + 0.8 * 平坦, 0.05, 0.7)),
        紧喉度=float(np.clip(0.55 - 重心 / 7000.0, 0.05, 0.85)),
        粗糙度=0.05,
        送气量=float(np.clip(0.15 + 0.3 * 平坦, 0.1, 0.6)),
        声区=float(np.clip((f0 - 180.0) / 300.0, 0.0, 0.5)),
        共振峰缩放=(float(np.clip((重心 / 2500.0) ** 0.30, 0.80, 1.25)),) * 3,
        谱倾斜=float(np.clip((重心 - 2500.0) / 4000.0, -0.6, 0.6)),
        气声谱重心=float(np.clip(0.5 + 0.2 * 平坦, 0.2, 0.8)),
        抖动=0.04,
        增益=1.0,
        来源="外采·参考录音",
        备注=(备注 or "从参考录音估参数（参数化模仿，非克隆）") + "｜参考sha256:%s" % sha,
    )
    a = 声线档案(名=名, **参)
    a.校验()
    return dict(声线=a, 参考=dict(F0中位=round(f0, 1), 谱重心=round(重心, 1),
                                 平坦=round(平坦, 4), 能量=round(能量, 4),
                                 sha256=sha, 文件=os.path.abspath(参考wav)))


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for p in 导出():
        print("导出:", p)
    for 名, a in 全部().items():
        print("%-6s F0=%5.1f 共振峰=%.2f/%.2f/%.2f 倾斜=%+.2f 指纹=%s"
              % (名, a.基频, a.共振峰缩放[0], a.共振峰缩放[1], a.共振峰缩放[2],
                 a.谱倾斜, a.指纹()))
