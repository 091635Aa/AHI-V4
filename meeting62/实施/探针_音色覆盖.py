# -*- coding: utf-8 -*-
"""meeting62/实施/探针_音色覆盖.py —— N2「音色身份可分」覆盖层候选值探针（只读 meeting61）。

只读 import 判据_发音_v7 的量法（_谱带/分段/_半音）；只写 stdout。
用于在落地 `声线覆盖层_v7.py` 前，实测候选覆盖层对 **登–缘 浊谱带 L1** 的影响。
口径与 N2 臂**逐位一致**（不改阈值）。

用法（先 $env:PYTHONIOENCODING="utf-8"）：
  & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" "<workspace>/\03_AHI\\AHI4\\meeting62\\实施\\探针_音色覆盖.py"
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M62 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M62)
MEET61 = os.path.join(AHI4, "meeting61")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(MEET61, "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 判据_发音_v7 as J  # noqa: E402
import 韵律_v7 as H7  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

角色 = ("洛天依", "SpeakerA", "EmoCompanion")
音色句 = "今天的风很轻，适合慢慢地说一会儿话。"


def 原档(名):
    with open(os.path.join(MEET61, "产物", "声线档案", 名 + ".json"), encoding="utf-8") as f:
        return 声线档案.从dict(json.load(f))


def 参考统计(名):
    with open(os.path.join(MEET61, "产物", "指标", "拟合44_%s.json" % 名), encoding="utf-8") as f:
        return json.load(f)["参考统计"]


def 测(名, 档):
    w, _, 帧 = H7.人类韵律_v5(档, 300).说(音色句)
    浊 = J.分段(w, 帧, {1})
    f0v = np.array([帧["f0"][i] for i, t in enumerate(帧["声源"]) if int(t) == 1])
    return dict(w=w, 浊=浊, F0=float(np.median(f0v)) if f0v.size else 0.0,
                浊谱带=J._谱带(浊))


def 两两(读数):
    out = {}
    for i in range(len(角色)):
        for j in range(i + 1, len(角色)):
            a, b = 角色[i], 角色[j]
            st = abs(J._半音(读数[a]["F0"], 读数[b]["F0"]))
            ba, bb = 读数[a]["浊谱带"], 读数[b]["浊谱带"]
            L1 = float(sum(abs(ba[k] - bb[k]) for k in ba if k in bb))
            out["%s-%s" % (a, b)] = (round(st, 3), round(L1, 4),
                                     "可分" if (st >= 1.0 or L1 >= 0.20) else "听不出")
    return out


def 建覆盖(模式: str):
    """按模式生成三角色覆盖档案（全部来自 meeting61 参考统计；不含任何外部源）。"""
    参 = {n: 参考统计(n) for n in 角色}
    重 = {n: 参[n]["共振峰"] for n in 角色}
    坡 = {n: 参[n]["谱倾斜"] for n in 角色}
    底重 = [float(np.mean([重[n][k] for n in 角色])) for k in range(3)]
    底坡 = float(np.mean([坡[n] for n in 角色]))
    基 = (1.1, 1.044, 1.044)          # 现档案的三轴值（塌缩基点）
    基斜 = -0.175
    出 = {}
    for n in 角色:
        a = 原档(n)
        d = {}
        if "重" in 模式:
            d["共振峰缩放"] = tuple(float(np.clip(基[k] * 重[n][k] / 底重[k], 0.70, 1.30))
                                 for k in range(3))
        if "斜" in 模式:
            d["谱倾斜"] = float(np.clip(基斜 * 坡[n] / 底坡, -0.5, 0.5))
        出[n] = a.派生(**d) if d else a
    return 出


def 主() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("=" * 100)
    for n in 角色:
        档 = 原档(n)
        参 = 参考统计(n)
        print("%-5s 档案 基频=%.1f 声区=%.3f 缩放=%s 斜率=%.3f | 参考 F0=%.1f 共振峰=%s 谱斜率=%.2f"
              % (n, 档.基频, 档.声区, ["%.3f" % x for x in 档.共振峰缩放], 档.谱倾斜,
                 参["F0中位"], 参["共振峰"], 参["谱倾斜"]))
    print("-" * 100)
    for 模式 in ("", "重", "重斜"):
        覆盖 = 建覆盖(模式)
        for n in 角色:
            print("  模式[%s] %-5s 缩放=%s 斜率=%.4f"
                  % (模式 or "原", n, ["%.4f" % x for x in 覆盖[n].共振峰缩放], 覆盖[n].谱倾斜))
        读数 = {n: 测(n, 覆盖[n]) for n in 角色}
        print("  模式[%s] 两两:" % (模式 or "原"))
        for k, v in 两两(读数).items():
            print("        %-14s F0半音=%.3f 浊谱带L1=%.4f  %s" % (k, v[0], v[1], v[2]))
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
