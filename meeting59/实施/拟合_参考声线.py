# -*- coding: utf-8 -*-
"""meeting59/实施/拟合_参考声线.py —— 真实参考录音 → 14 维原生声线档案（参数化拟合）。

方法（全部本机经典 DSP，无任何神经网络模型）：
  1) 参考统计：F0 中位、倒谱包络 F1/F2/F3 中位（诊断）、7 带平均对数包络、谱倾斜、
     谐噪比、能量包络调制率（音节/秒）、F0 抖动、浊音比；
  2) 初值：F0→基频；调制率→语速；谐噪比→气声度；抖动→粗糙度；F0 位置→声区；
  3) 阶段1（2 轮）：用**最终交付同一条渲染链**（人类化韵律 + 原生声带）合成探针，
     按测量值校正 基频 / 语速，使合成声学量对齐参考；
  4) 阶段2（3 轮坐标搜索）：以 **7 带 log 包络余弦** 为主目标（对增益不变、跨元音稳定），
     搜索 全局共振峰缩放 / 谱倾斜 / 气声度 / 紧喉度，带轻正则防止极端漂移；
  5) 阶段3：F1/F2/F3 三轴 ±5% 微分微调（同一包络目标 + 正则）；
  6) 输出档案 JSON + 拟合指标 JSON（含参考 sha256、来源标签、诚实口径）。

为什么以 7 带包络为主目标：参考语音的元音分布未知，倒谱共振峰中位受元音分布与估计器
偏差影响大（E2 已登记），不能可靠反演个体声道；包络形状对元音分布更稳健，也直接对应听感音色。
倒谱共振峰仍作为诊断量记录，并用于三轴微调。

口径：**参数化模仿，不是声克隆**；参考音频不复制、不复用波形；固定种子两趟逐位同。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET59 = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import 声学工具 as T  # noqa: E402
import 韵律_人类 as 人类  # noqa: E402
import 谱包络_校正 as E  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

产物档案 = os.path.join(MEET59, "产物", "声线档案")
产物指标 = os.path.join(MEET59, "产物", "指标")
os.makedirs(产物档案, exist_ok=True)
os.makedirs(产物指标, exist_ok=True)

探针文本 = "你好，很高兴认识你。今天天气不错，我们一起出去走走吧。"


def 探针合成(档案: 声线档案) -> np.ndarray:
    """用最终交付同一条渲染链（人类化韵律 + 原生声带）合成探针。"""
    w, _, _ = 人类.说(探针文本, 档案, 300)
    return w


def 分析(x: np.ndarray) -> dict:
    f0, hnr, 能 = T.基频轨迹(x)
    v = f0[np.isfinite(f0)]
    return dict(
        F0=float(np.median(v)) if v.size else 0.0,
        共振峰=T.共振峰中位(x, f0, 能),
        带=T.带包络(x),
        倾斜=T.谱倾斜(x),
        语速=T.语速估计(x),
        谐噪比=T.谐噪比中位(x),
        抖动=T.抖动估计(x),
        均方根=T.均方根(x),
    )


def _参考原始带(路径s: list) -> np.ndarray:
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
        基频=f0, 语速=语速, 幅度=0.50,
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


def _距离带(档案: 声线档案, 参考: dict) -> tuple:
    """(带正则距离, 纯包络余弦, 波形)；主目标是 7 带 log 包络余弦。"""
    w = 探针合成(档案)
    带 = T.带包络(w)
    d0 = T.包络距离(带, 参考["带"])
    s = 档案.共振峰缩放
    reg = 0.020 * float(sum(np.log2(max(float(x), 0.5)) ** 2 for x in s)) \
        + 0.010 * float(档案.谱倾斜 ** 2)
    return float(d0 + reg), float(d0), w


def _试参(档案: 声线档案, 参: dict) -> 声线档案:
    基 = 参["基"]
    return 基.派生(
        共振峰缩放=tuple(float(np.clip(x * 参["缩放"], 0.72, 1.28))
                    for x in 基.共振峰缩放),
        谱倾斜=float(np.clip(参["谱倾斜"], -0.75, 0.75)),
        气声度=float(np.clip(参["气声度"], 0.0, 0.75)),
        紧喉度=float(np.clip(参["紧喉度"], 0.02, 0.92)),
    )


def 拟合(路径s: list, 名: str, 迭代: int = 2, 搜索轮: int = 3, 详细=True,
         F0目标: float | None = None) -> dict:
    参考 = T.参考统计(路径s)
    目标F0 = float(F0目标) if F0目标 else float(参考["F0"])
    档案 = _建档(初值(参考), 名)
    轨迹 = []
    # 阶段1：F0 / 语速 校正（基频与帧 F0 是严格线性关系，用帧真值校正，避免估计器高音域失效）
    for r in range(迭代):
        w, _, 帧 = 人类.说(探针文本, 档案, 300)
        测 = 分析(w)
        帧F0 = float(np.median(帧["f0"])) if 帧["f0"] else 0.0
        f0倍 = float(np.clip(目标F0 / max(帧F0, 1e-6), 0.5, 2.0))
        语速倍 = float(np.clip(参考["语速"] / max(测["语速"], 1e-6), 0.88, 1.14))
        档案 = 档案.派生(
            基频=float(np.clip(档案.基频 * f0倍, 70.0, 400.0)),
            语速=float(np.clip(档案.语速 * 语速倍, 1.5, 7.0)),
        )
        轨迹.append(dict(轮=r, 帧F0=round(帧F0, 1), F0参=round(目标F0, 1),
                         音频估计F0=round(测["F0"], 1),
                         语速测=round(测["语速"], 2), 语速参=round(参考["语速"], 2)))
    # 阶段2：7 带包络坐标搜索（缩放 + 倾斜 + 气声 + 紧喉）
    基 = 档案
    参 = dict(缩放=1.0, 谱倾斜=基.谱倾斜, 气声度=基.气声度,
              紧喉度=基.紧喉度, 基=基)
    当前, _, _ = _距离带(_试参(基, 参), 参考)
    steps = dict(缩放=0.07, 谱倾斜=0.14, 气声度=0.07, 紧喉度=0.09)
    for 轮 in range(搜索轮):
        for k in ("缩放", "谱倾斜", "气声度", "紧喉度"):
            for 向 in (+1.0, -1.0):
                试 = dict(参)
                试[k] = 参[k] + 向 * steps[k]
                dnew, _, _ = _距离带(_试参(基, 试), 参考)
                if dnew < 当前 - 1e-6:
                    当前 = dnew
                    参 = 试
        for k in steps:
            steps[k] *= 0.5
    档案 = _试参(基, 参)
    # 阶段3：三轴微分微调（每轴 ±5%，包络目标 + 正则）
    当前, _, _ = _距离带(档案, 参考)
    for axis in range(3):
        for 向 in (+1.0, -1.0):
            s2 = list(档案.共振峰缩放)
            s2[axis] = float(np.clip(s2[axis] * (1.0 + 向 * 0.05), 0.72, 1.28))
            试 = 档案.派生(共振峰缩放=tuple(s2))
            dnew, _, _ = _距离带(试, 参考)
            if dnew < 当前 - 1e-6:
                当前 = dnew
                档案 = 试
    # 阶段4：F0 收尾校正（用帧 F0 真值；基频→帧 F0 是严格线性缩放，避免高音域估计器失效）
    _, _, 帧4 = 人类.说(探针文本, 档案, 300)
    for _ in range(2):
        帧F0 = float(np.median(帧4["f0"])) if 帧4["f0"] else 0.0
        比 = float(np.clip(目标F0 / max(帧F0, 1e-6), 0.5, 2.0))
        if abs(比 - 1.0) <= 0.001:
            break
        档案 = 档案.派生(基频=float(np.clip(档案.基频 * 比, 70.0, 400.0)))
        _, _, 帧4 = 人类.说(探针文本, 档案, 300)
    # 阶段5：7 段静态谱包络校正（E2 建议2；纯 numpy 零相位 EQ；存独立 json，不进档案）
    w裸, _, 帧裸 = 人类.说(探针文本, 档案, 300)
    帧F0终 = float(np.median(帧裸["f0"])) if 帧裸["f0"] else 0.0
    参考原始带 = _参考原始带(路径s)
    增益 = E.设计(参考原始带, T.带包络(w裸, 归一=False))
    w_eq = E.应用(w裸, 增益)
    带余弦_校正 = T.带余弦(T.带包络(w_eq), 参考["带"])
    # 最终指标
    d终, d0终, w终 = _距离带(档案, 参考)
    测终 = 分析(w终)
    fr, fs = np.asarray(参考["共振峰"]), np.asarray(测终["共振峰"])
    指标 = dict(
        名=名,
        参考文件=[os.path.abspath(p) for p in 路径s],
        参考sha256=[T.文件sha256(p) for p in 路径s],
        参考统计=dict(F0中位=round(参考["F0"], 2), 语速Hz=round(参考["语速"], 3),
                      共振峰=[round(float(x), 1) for x in 参考["共振峰"]],
                      谱倾斜=round(参考["谱倾斜"], 2), 谐噪比=round(参考["谐噪比"], 3),
                      抖动=round(参考["抖动"], 3), 均方根=round(参考["均方根"], 4),
                      浊音比=round(参考["浊音比"], 3), 时长秒=参考["时长秒"]),
        合成统计=dict(F0中位=round(帧F0终, 2), 音频估计F0=round(测终["F0"], 1),
                      语速Hz=round(测终["语速"], 3),
                      共振峰=[round(float(x), 1) for x in 测终["共振峰"]],
                      谱倾斜=round(测终["倾斜"], 2)),
        F0差半音=round(T.半音距(帧F0终, 目标F0), 3),
        F0目标=round(目标F0, 2),
        带余弦=round(1.0 - d0终, 4),
        带余弦_校正=round(float(带余弦_校正), 4),
        EQ增益dB=[round(float(v), 2) for v in 增益],
        EQ说明="交付音频含 7 段静态谱包络校正（纯 numpy 零相位；meeting58 E2 建议2）；带余弦=裸声带值",
        共振峰相对误差=([round(float(abs(fs[j] - fr[j]) / max(fr[j], 1e-9)), 4)
                    for j in range(3)]
                   if np.all(np.isfinite(fr)) and np.all(np.isfinite(fs)) else None),
        迭代轨迹=轨迹,
        搜索距离=round(float(d终), 4),
        声线档案=dict(档案.转dict(), 指纹=档案.指纹()),
        口径="参数化模仿（从参考录音估参数），不是真克隆；参考音频不复制波形、不再分发",
    )
    档案 = 档案.派生(备注="参考拟合｜%s｜参考F0=%.1fHz 带余弦=%.3f→%.3f(EQ)｜参数化模仿非克隆"
                  % ("/".join(os.path.basename(p) for p in 路径s),
                     参考["F0"], 指标["带余弦"], 指标["带余弦_校正"]))
    if 详细:
        print("[拟合] %-4s F0=%.1f(帧%.1f) 带余弦=%.4f→%.4f(EQ) 缩放=%.2f/%.2f/%.2f 倾斜=%.2f "
              "气声=%.2f 紧喉=%.2f"
              % (名, 参考["F0"], 帧F0终, 指标["带余弦"], 指标["带余弦_校正"],
                 *档案.共振峰缩放, 档案.谱倾斜, 档案.气声度, 档案.紧喉度))
    return dict(声线=档案, 指标=指标, 参考=参考, EQ=增益)


def 存(结: dict, 名: str) -> dict:
    a = 结["声线"]
    p档案 = os.path.join(产物档案, 名 + ".json")
    with open(p档案, "w", encoding="utf-8") as f:
        json.dump(dict(a.转dict(), 指纹=a.指纹()), f, ensure_ascii=False, indent=1)
    p指标 = os.path.join(产物指标, "拟合_%s.json" % 名)
    with open(p指标, "w", encoding="utf-8") as f:
        json.dump(结["指标"], f, ensure_ascii=False, indent=1)
    pEQ = E.存(os.path.join(产物档案, 名 + ".eq.json"), 结["EQ"],
                dict(名=名, 档案指纹=a.指纹(),
                     说明="7 段静态谱包络校正（纯 numpy 零相位）；目标=参考录音活动帧平均 7 带 log 功率"))
    return dict(档案=os.path.abspath(p档案), 指标=os.path.abspath(p指标),
                EQ=os.path.abspath(pEQ), 指纹=a.指纹())


def 参考路径(名: str) -> list:
    目录 = os.path.join(MEET59, "素材", 名)
    if not os.path.isdir(目录):
        return []
    ps = sorted(glob.glob(os.path.join(目录, "ref_*.wav")))
    # 洛天依：只用第三方「清唱干声」（L-C，无 BGM、可直接代表该声线）；
    # 「语音」样本(non-清唱)仅作旁证，不参与拟合；带 BGM 的整段样本也不参与。
    if 名 == "洛天依":
        清唱 = [p for p in ps if "清唱" in os.path.basename(p)]
        if 清唱:
            ps = 清唱
    if not ps:
        ps = sorted(glob.glob(os.path.join(目录, "*.wav")))
    return ps


def _洛天依目标F0() -> float | None:
    """洛天依说话音高目标 = 清唱干声组的 F0 中位（说话组仅旁证，不混入）。"""
    目录 = os.path.join(MEET59, "素材", "洛天依")
    ps = sorted(glob.glob(os.path.join(目录, "ref_洛天依_清唱_*.wav")))
    if not ps:
        return None
    return float(T.参考统计(ps)["F0"])


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    名s = [a for a in argv if not a.startswith("-")] or ["EmoCompanion", "SpeakerA", "洛天依"]
    输出 = {}
    for 名 in 名s:
        ps = 参考路径(名)
        if not ps:
            print("[拟合] 跳过 %s：无参考片段（素材/%s/）" % (名, 名))
            continue
        F0目 = _洛天依目标F0() if 名 == "洛天依" else None
        结 = 拟合(ps, 名, F0目标=F0目)
        if F0目:
            结["指标"]["参考统计"]["F0语音目标"] = round(F0目, 2)
        输出[名] = 存(结, 名)
        print("   指标:", json.dumps(结["指标"]["参考统计"], ensure_ascii=False))
    print(json.dumps(输出, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
