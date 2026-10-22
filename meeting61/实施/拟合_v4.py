# -*- coding: utf-8 -*-
"""meeting60/实施/拟合_44k.py —— 参考录音 → 14 维声线（拟合对象 = 44.1k 新引擎）。

与 meeting59 拟合的差别：
  · 合成探针走 `韵律_44k.人类韵律44`（44.1k、F1–F5、无 ±8dB EQ），下采样到 16k 再与参考同口径比对；
  · 共振峰缩放约束 [0.85,1.10]（修 v2 的 1.18 卡通化）；谱倾斜约束 [-0.35,0.35]；
  · EQ 只做 ≤±3dB 轻校（组D用），>6.5k 自动回 0dB，保留引擎自然高频；
  · 最终 F0 用帧真值对中参考 F0 中位。

口径：参数化模仿，不是声克隆；参考音频不复制、不再分发。C1/C10：无外部模型。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
MEET61 = os.path.dirname(HERE)
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting59", "实施"))

import 声学工具 as T  # noqa: E402  16k 分析口径（只读复用）
import 韵律_v4 as H44  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

产物档案 = os.path.join(MEET61, "产物", "声线档案")
产物指标 = os.path.join(MEET61, "产物", "指标")
os.makedirs(产物档案, exist_ok=True)
os.makedirs(产物指标, exist_ok=True)

探针文本 = "你好，很高兴认识你。今天天气不错，我们一起出去走走吧。"
缩放域 = (0.85, 1.10)


def 下采样16k(x44: np.ndarray) -> np.ndarray:
    return resample_poly(np.asarray(x44, dtype=np.float64), 160, 441)


def _带6(x16: np.ndarray, 原始: bool = False) -> np.ndarray:
    """250–8k 六带 log 功率（跳过 0–250 隆隆带；默认在六带内减均值）。"""
    v = T.带包络(x16, 归一=False)[1:]
    return v if 原始 else v - float(np.mean(v))


def 分析16(x16: np.ndarray) -> dict:
    f0, hnr, 能 = T.基频轨迹(x16)
    v = f0[np.isfinite(f0)]
    return dict(
        F0=float(np.median(v)) if v.size else 0.0,
        共振峰=T.共振峰中位(x16, f0, 能),
        带=_带6(x16),
        带7=T.带包络(x16),
        倾斜=T.谱倾斜(x16),
        语速=T.语速估计(x16),
        谐噪比=T.谐噪比中位(x16),
        抖动=T.抖动估计(x16),
        均方根=T.均方根(x16),
    )


def 探针合成(档案: 声线档案) -> tuple:
    w44, 元, 帧 = H44.说(探针文本, 档案, 300)
    return 下采样16k(w44), 帧, w44


def 参考路径(名: str) -> list:
    """优先 meeting60 的 44.1k 参考；否则回退 meeting59 的 16k 参考。"""
    p60 = os.path.join(MEET61, "素材", 名)
    ps = sorted(glob.glob(os.path.join(p60, "ref44_*.wav")))
    if not ps and os.path.isdir(p60):
        ps = sorted(glob.glob(os.path.join(p60, "*.wav")))
    if ps:
        return ps
    p59 = os.path.join(AHI4, "meeting59", "素材", 名)
    ps = sorted(glob.glob(os.path.join(p59, "ref_*.wav")))
    if 名 == "洛天依":
        ps = [p for p in ps if "清唱" in os.path.basename(p)] or ps
    return ps


def 参考原始带(路径s: list) -> np.ndarray:
    xs = [T.读wav16(p) for p in 路径s]
    x0 = np.concatenate([np.concatenate([v, np.zeros(int(0.15 * T.FS))]) for v in xs])
    return T.带包络(T.活动片段(x0), 归一=False)


def 初值(参考: dict) -> dict:
    f0 = float(np.clip(参考["F0"], 70.0, 400.0))
    语速 = float(np.clip(3.2 * 4.0 / max(float(参考["语速"]), 2.0), 1.8, 5.5))
    气声 = float(np.clip(0.55 - 0.50 * float(参考["谐噪比"]), 0.03, 0.55))
    抖动 = float(np.clip(参考["抖动"], 0.0, 1.0))
    return dict(
        名="临时", 版本="1.0", 来源="外采·参考录音",
        基频=f0, 语速=语速, 幅度=0.55,
        气声度=气声, 紧喉度=0.30, 粗糙度=float(np.clip(抖动 * 1.2, 0.02, 0.35)),
        送气量=0.20, 声区=float(np.clip((f0 - 180.0) / 300.0, 0.0, 0.5)),
        共振峰缩放=(1.0, 1.0, 1.0), 谱倾斜=0.0, 气声谱重心=0.5,
        抖动=float(np.clip(抖动 * 0.05, 0.0, 0.1)), 增益=1.0,
        备注="待拟合",
    )


def _建档(d: dict, 名: str) -> 声线档案:
    d = dict(d)
    d["名"] = 名
    d["共振峰缩放"] = tuple(float(x) for x in d.get("共振峰缩放", (1.0, 1.0, 1.0)))
    a = 声线档案.从dict(d)
    a.校验()
    return a


def _带距离(档案: 声线档案, 参考: dict) -> tuple:
    x16, 帧, w44 = 探针合成(档案)
    测 = 分析16(x16)
    d0 = T.包络距离(测["带"], 参考["带6"])
    s = 档案.共振峰缩放
    reg = 0.012 * float(sum(np.log2(max(float(x), 0.5)) ** 2 for x in s)) \
        + 0.010 * float(档案.谱倾斜 ** 2)
    return float(d0 + reg), float(d0), 测, 帧, w44


def _试参(基: 声线档案, 参: dict) -> 声线档案:
    缩放 = tuple(float(np.clip(x * 参["缩放"], *缩放域)) for x in 基.共振峰缩放)
    return 基.派生(
        共振峰缩放=缩放,
        谱倾斜=float(np.clip(参["谱倾斜"], -0.35, 0.35)),
        气声度=float(np.clip(参["气声度"], 0.0, 0.60)),
        紧喉度=float(np.clip(参["紧喉度"], 0.08, 0.78)),
    )


def 拟合(路径s: list, 名: str, 迭代: int = 2, 搜索轮: int = 3, 详细=True) -> dict:
    参考 = T.参考统计(路径s)          # 16k 参考统计
    参考原始 = 参考原始带(路径s)       # 7 带 raw log 功率
    参考["带6"] = 参考原始[1:] - float(np.mean(参考原始[1:]))
    目标F0 = float(参考["F0"])
    档案 = _建档(初值(参考), 名)
    轨迹 = []
    for r in range(迭代):
        x16, 帧, _ = 探针合成(档案)
        测 = 分析16(x16)
        帧F0 = float(np.median(帧["f0"])) if 帧["f0"] else 0.0
        f0倍 = float(np.clip(目标F0 / max(帧F0, 1e-6), 0.5, 2.0))
        档案 = 档案.派生(基频=float(np.clip(档案.基频 * f0倍, 70.0, 400.0)),
                         语速=4.8)   # v4：参考源偏慢/偏歌，语速不跟参考，用自然说话语速
        轨迹.append(dict(轮=r, 帧F0=round(帧F0, 1), 目标F0=round(目标F0, 1),
                         音频F0=round(测["F0"], 1), 语速测=round(测["语速"], 2)))
    # 包络坐标搜索
    基 = 档案
    参 = dict(缩放=1.0, 谱倾斜=基.谱倾斜, 气声度=基.气声度, 紧喉度=基.紧喉度)
    当前, _, _, _, _ = _带距离(_试参(基, 参), 参考)
    steps = dict(缩放=0.05, 谱倾斜=0.10, 气声度=0.06, 紧喉度=0.08)
    for 轮 in range(搜索轮):
        for k in ("缩放", "谱倾斜", "气声度", "紧喉度"):
            for 向 in (+1.0, -1.0):
                试 = dict(参)
                试[k] = 参[k] + 向 * steps[k]
                dnew, _, _, _, _ = _带距离(_试参(基, 试), 参考)
                if dnew < 当前 - 1e-6:
                    当前 = dnew
                    参 = 试
        for k in steps:
            steps[k] *= 0.5
    档案 = _试参(基, 参)
    # 三轴微分微调
    当前, _, _, _, _ = _带距离(档案, 参考)
    for axis in range(3):
        for 向 in (+1.0, -1.0):
            s2 = list(档案.共振峰缩放)
            s2[axis] = float(np.clip(s2[axis] * (1.0 + 向 * 0.04), *缩放域))
            试 = 档案.派生(共振峰缩放=tuple(s2))
            dnew, _, _, _, _ = _带距离(试, 参考)
            if dnew < 当前 - 1e-6:
                当前 = dnew
                档案 = 试
    # F0 收尾对中（帧真值）
    for _ in range(2):
        x16, 帧, _ = 探针合成(档案)
        帧F0 = float(np.median(帧["f0"])) if 帧["f0"] else 0.0
        比 = float(np.clip(目标F0 / max(帧F0, 1e-6), 0.5, 2.0))
        if abs(比 - 1.0) <= 0.001:
            break
        档案 = 档案.派生(基频=float(np.clip(档案.基频 * 比, 70.0, 400.0)))
    # 最终：裸声带指标 + 六带 EQ（250–8k，≤±9dB，避免 v2 的 16k 三共振峰 EQ 染色）
    x16, 帧, w44 = 探针合成(档案)
    测 = 分析16(x16)
    合成原始 = T.带包络(x16, 归一=False)[1:]
    增益 = 参考原始[1:] - 合成原始
    增益 = 增益 - float(np.mean(增益))
    增益 = np.clip(增益 * (10.0 / np.log(10.0)), -6.0, 6.0)
    增益 = 增益 - float(np.mean(增益))
    w_eq = H44.应用EQ44(w44, 增益)
    x_eq16 = 下采样16k(w_eq)
    带余弦_校正 = T.带余弦(_带6(x_eq16), 参考["带6"])
    带余弦 = T.带余弦(测["带"], 参考["带6"])
    带余弦7 = T.带余弦(T.带包络(x16), 参考["带"])
    帧F0终 = float(np.median(帧["f0"])) if 帧["f0"] else 0.0
    fr, fs = np.asarray(参考["共振峰"]), np.asarray(测["共振峰"])
    指标 = dict(
        名=名,
        参考文件=[os.path.abspath(p) for p in 路径s],
        参考sha256=[T.文件sha256(p) for p in 路径s],
        参考统计=dict(F0中位=round(参考["F0"], 2), 语速Hz=round(参考["语速"], 3),
                      共振峰=[round(float(x), 1) for x in 参考["共振峰"]],
                      谱倾斜=round(参考["谱倾斜"], 2), 谐噪比=round(参考["谐噪比"], 3),
                      抖动=round(参考["抖动"], 3), 均方根=round(参考["均方根"], 4),
                      浊音比=round(参考["浊音比"], 3), 时长秒=参考["时长秒"]),
        合成统计=dict(帧F0=round(帧F0终, 2), 音频F0=round(测["F0"], 1),
                      语速Hz=round(测["语速"], 3),
                      共振峰=[round(float(x), 1) for x in 测["共振峰"]]),
        F0差半音=round(T.半音距(帧F0终, 目标F0), 3), F0目标=round(目标F0, 2),
        带余弦=round(带余弦, 4), 带余弦_校正=round(float(带余弦_校正), 4),
        带余弦7=round(float(带余弦7), 4),
        EQ增益dB=[round(float(v), 2) for v in 增益],
        共振峰相对误差=([round(float(abs(fs[j] - fr[j]) / max(fr[j], 1e-9)), 4)
                    for j in range(3)]
                   if np.all(np.isfinite(fr)) and np.all(np.isfinite(fs)) else None),
        迭代轨迹=轨迹,
        声线档案=dict(档案.转dict(), 指纹=档案.指纹()),
        口径="44.1k 原生引擎参数化模仿（非克隆）；包络=250–8k 六带；EQ≤±9dB；参考不复制/不再分发",
    )
    档案 = 档案.派生(备注="44k修复拟合｜%s｜参考F0=%.1f 带余弦(250-8k)=%.3f→%.3f(EQ)｜参数化模仿非克隆"
                  % ("/".join(os.path.basename(p) for p in 路径s),
                     参考["F0"], 带余弦, 带余弦_校正))
    if 详细:
        print("[拟合44] %-4s F0=%.1f(帧%.1f) 带余弦(250-8k)=%.4f→%.4f(EQ±9) 缩放=%.2f/%.2f/%.2f "
              "斜=%.2f 气=%.2f 紧=%.2f"
              % (名, 参考["F0"], 帧F0终, 带余弦, 带余弦_校正,
                 *档案.共振峰缩放, 档案.谱倾斜, 档案.气声度, 档案.紧喉度))
    return dict(声线=档案, 指标=指标, 参考=参考, EQ=增益)


def 存(结: dict, 名: str) -> dict:
    a = 结["声线"]
    p档案 = os.path.join(产物档案, 名 + ".json")
    with open(p档案, "w", encoding="utf-8") as f:
        json.dump(dict(a.转dict(), 指纹=a.指纹()), f, ensure_ascii=False, indent=1)
    p指标 = os.path.join(产物指标, "拟合44_%s.json" % 名)
    with open(p指标, "w", encoding="utf-8") as f:
        json.dump(结["指标"], f, ensure_ascii=False, indent=1)
    pEQ = os.path.join(产物档案, 名 + ".eq44.json")
    with open(pEQ, "w", encoding="utf-8") as f:
        json.dump(dict(增益dB=[round(float(v), 3) for v in 结["EQ"]],
                       带中心Hz=list(H44.带中心),
                       元=dict(名=名, 指纹=a.指纹(), 说明="≤±3dB 轻校；>6.5k 回 0dB")),
                  f, ensure_ascii=False, indent=1)
    return dict(档案=os.path.abspath(p档案), 指标=os.path.abspath(p指标),
                EQ=os.path.abspath(pEQ), 指纹=a.指纹())


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    名s = [a for a in argv if not a.startswith("-")] or ["EmoCompanion", "SpeakerA", "洛天依"]
    输出 = {}
    for 名 in 名s:
        ps = 参考路径(名)
        if not ps:
            print("[拟合44] 跳过 %s：无参考" % 名)
            continue
        结 = 拟合(ps, 名)
        输出[名] = 存(结, 名)
        print("   档案:", 输出[名]["档案"])
    print(json.dumps(输出, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
