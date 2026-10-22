# -*- coding: utf-8 -*-
"""meeting62/实施/声线覆盖层_v7.py —— 三角色「声线档案覆盖层」（N2 修复，纯新增，C14）。

问题（`诊断报告.md` §6.2 / 缺口 G3）：meeting61 的三个声线档案（洛天依/SpeakerA/EmoCompanion）14 维中
**7 维逐位恒同**（语速 4.8 / 幅度 0.55 / 紧喉 0.42 / 送气 0.2 / 谱倾斜 −0.175 / 气声谱重心 0.5 /
共振峰缩放第 1 轴 1.1），其中 `共振峰缩放` 在 **SpeakerA/EmoCompanion** 上完全同值 `[1.1,1.044,1.044]`
⇒ 引擎对三角色的产出可听差异主要只剩 F0（登–缘仅 0.196 半音），即所有者说的「换角色听不出」。
而**参考层本身是可分的**：`拟合44_*.json` 的 `参考统计.共振峰` F3 = 1492/1836/1617 Hz
（登–缘差 **13.5%**），说明差异**原本存在、只是被拟合压平**。

本覆盖层只把 **`共振峰缩放`（F1/F2/F3 三轴）** 与 **`谱倾斜`** 两维**从「参考录音实测统计」重标定**，
经**引擎真实合成路径**生效（`韵律_v7` → `引擎_v7` 的 Klatt 串联 F1–F5 / 带宽 / 谱倾斜一阶级联），
**不是事后 EQ**：覆盖值在**运行期**由参考统计确定性地派生，**不落盘改写任何档案**。

维度来源（全部是 meeting61 内**已固化的参考录音统计**，零外部音频/模型，C10 合规）：
  · `共振峰缩放` 三轴 = 现档案塌缩基点 (1.1, 1.044, 1.044) × (参考共振峰_该轴 ÷ 三角色该轴均值)
  · `谱倾斜`          = 现档案基点 −0.175 × (参考谱倾斜_角色 ÷ 三角色均值)
**未启用维度（如实登记，非「没做」）**：
  · `带宽缩放`：v7 引擎支持（`引擎_v7` 读 `音["带宽缩放"]`），但 meeting61 无**参考带宽统计**，
    无据可分 ⇒ 保持 1.0（若启用即属无据调参，违反「差异必须来自角色档案」）。
  · `送气量`：参考 `谐噪比` 三角色濒同（0.873/0.883/0.880）⇒ 无独立可分统计 ⇒ 保持档案原值 0.2。
  · `气声度/粗糙度/抖动`：档案层**本已随拟合不同**（非恒同维），不改。

只读 import meeting61 档案/指标 json + meeting58 契约；meeting61/58 一字不改（C14）。
纯算术，确定性（同 json ⇒ 同覆盖值）。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M62 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M62)
MEET61 = os.path.join(AHI4, "meeting61")
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

from 接口契约_角色声线 import 声线档案  # noqa: E402

角色 = ("洛天依", "SpeakerA", "EmoCompanion")
档案目录 = os.path.join(MEET61, "产物", "声线档案")
指标目录 = os.path.join(MEET61, "产物", "指标")

# 现档案的塌缩基点（三轴/斜率，取自 meeting61 档案层的恒同值）
基点缩放 = (1.1, 1.044, 1.044)      # F1/F2/F3 轴
基点倾斜 = -0.175                    # 谱倾斜

_底缓存 = None


def _读档案(名: str) -> 声线档案:
    with open(os.path.join(档案目录, 名 + ".json"), encoding="utf-8") as f:
        return 声线档案.从dict(json.load(f))


def _参考统计(名: str) -> dict:
    with open(os.path.join(指标目录, "拟合44_%s.json" % 名), encoding="utf-8") as f:
        return json.load(f)["参考统计"]


def _底() -> tuple:
    """三角色参考统计的两条均值基线：((F1,F2,F3) 均值, 谱倾斜均值)。"""
    global _底缓存
    if _底缓存 is None:
        重 = [_参考统计(n)["共振峰"] for n in 角色]
        坡 = [_参考统计(n)["谱倾斜"] for n in 角色]
        _底缓存 = (tuple(float(sum(r[k] for r in 重)) / len(角色) for k in range(3)),
                  float(sum(坡)) / len(坡))
    return _底缓存


def 覆盖字段(名: str) -> dict:
    """该角色的覆盖字段（仅这两维；与档案基点相乘，钳制在契约值域内）。"""
    参 = _参考统计(名)
    重 = 参["共振峰"]
    坡 = 参["谱倾斜"]
    底重, 底坡 = _底()
    return dict(
        共振峰缩放=tuple(round(float(min(max(基点缩放[k] * 重[k] / 底重[k], 0.70), 1.30)), 6)
                       for k in range(3)),
        谱倾斜=round(float(min(max(基点倾斜 * 坡 / 底坡, -0.5), 0.5)), 6),
    )


def 取(名: str) -> 声线档案:
    """读 meeting61 档案 + 在运行期派生覆盖字段；返回**契约合法**的 声线档案。"""
    a = _读档案(名)
    d = 覆盖字段(名)
    b = a.派生(**d)
    b.校验()
    return b


def 说明() -> dict:
    """覆盖层自描述（供判据读数登记：维度/来源/值/未启用维）。"""
    底重, 底坡 = _底()
    return dict(
        角色=list(角色),
        覆盖维=["共振峰缩放(F1/F2/F3三轴)", "谱倾斜"],
        来源="meeting61 产物/指标/拟合44_*.json 的 参考统计.共振峰 / 参考统计.谱倾斜（已固化，零外部源）",
        派生式="共振峰缩放轴k = (1.1,1.044,1.044)[k] × 参考共振峰[k]/三角色均值[k]"
              "; 谱倾斜 = −0.175 × 参考谱倾斜/三角色均值",
        三角色均值=dict(共振峰=[round(x, 2) for x in 底重], 谱倾斜=round(底坡, 4)),
        逐角色={n: dict(参考共振峰=_参考统计(n)["共振峰"],
                        参考谱倾斜=_参考统计(n)["谱倾斜"], **覆盖字段(n)) for n in 角色},
        未启用维=dict(带宽缩放="值保持 1.0：meeting61 无参考带宽统计，无据可分",
                    送气量="值保持档案原 0.2：参考 HNR 三角色濒同（0.873/0.883/0.880），无独立可分统计"),
        落盘="不改写任何档案 json（覆盖仅在运行期派生，C14）",
    )


def 恒同维诊断() -> dict:
    """登记档案层「三角色逐位恒同」的维度（N2 根因证据）。"""
    档s = {n: _读档案(n) for n in 角色}
    字段 = ("语速", "幅度", "紧喉度", "送气量", "谱倾斜", "气声谱重心")
    恒 = []
    for k in 字段:
        if len({round(float(getattr(档s[n], k)), 9) for n in 角色}) == 1:
            恒.append(k)
    缩 = [tuple(float(x) for x in 档s[n].共振峰缩放) for n in 角色]
    for k in range(3):
        if len({round(x[k], 9) for x in 缩}) == 1:
            恒.append("共振峰缩放轴%d" % (k + 1))
    return dict(恒同维=恒, 恒同维数=len(恒))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import pprint
    print("恒同维诊断:", 恒同维诊断())
    print("-" * 80)
    for n in 角色:
        a0, a1 = _读档案(n), 取(n)
        print("%-4s 原 缩放=%s 斜率=%+.4f → 覆盖 缩放=%s 斜率=%+.4f 指纹=%s"
              % (n, ["%.4f" % x for x in a0.共振峰缩放], a0.谱倾斜,
                 ["%.4f" % x for x in a1.共振峰缩放], a1.谱倾斜, a1.指纹()))
    print("-" * 80)
    pprint.pprint(说明(), sort_dicts=False)
