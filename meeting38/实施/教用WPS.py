# -*- coding: utf-8 -*-
"""meeting38/实施/教用WPS.py —— 步7 教学：教两三次就会（目标指认 → 寻找 → 双击运行 → 多巴胺）

━━━ 教学法（所有者 2026-10-06：「教两三次就会」「移动性偏置」）━━━
  ① 指认：主人给出图标位置 → 采该处主色（眼帧 5×5 中值）→ 写进 目标槽（"我在找这个颜色"）；
  ② 寻找：脑注意被目标色强吸引（分 +3×接近度，压过新异）→ 眼经 意图偏置 边看边靠近 →
     手：靠近 → 到达**停 0.25s** → 双击运行；
  ③ 多巴胺：世界大变=+1；**目标达成再 +3**（"找到了！"），按主色桶记进多巴胺账本——
     「这种颜色常出好东西」成为它自己的偏好（回报偏置 0.5×均值 反灌注意打分）；
  ④ 检验：清掉目标 → 看它是否**自发**再去找同色目标（教 2-3 遍后的留存，无后训练、零梯度）。

━━━ 模式 ━━━
  --沙盒：演练引擎 + numpy 沙盒世界（判据/彩排，零真机）；
  默认实机：物理机主屏真鼠标（所有者知情在场；ESC 急停；拖/打字仍关）。

用法：
  python 实施/教用WPS.py --图标 1480,540 --遍 3 --遍秒 120 --检验秒 90 [--沙盒]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from 生命体 import 生命体                                              # noqa: E402
from 实机探索 import _ESC按下                                         # noqa: E402


def _主色采样(帧, 点, 半径: int = 25) -> tuple:
    """眼帧上 点 周围 (2r+1)² 的 RGB 中值（教学指认的颜色快照）。
    半径默认 25：贴眼元素粒度（120px 格）下的块主色口径，真机桌面图标也可用。"""
    h, w = 帧.shape[:2]
    x, y = int(点[0]), int(点[1])
    x0, x1 = max(0, x - 半径), min(w, x + 半径 + 1)
    y0, y1 = max(0, y - 半径), min(h, y + 半径 + 1)
    块 = 帧[y0:y1, x0:x1].reshape(-1, 3)
    return tuple(int(v) for v in np.median(块, axis=0))


def 主(图标: str = "1480,540", 遍: int = 3, 遍秒: float = 120.0,
       检验秒: float = 90.0, 沙盒: bool = False, 发声: bool = False,
       名称: str = "WPS") -> dict:
    点 = tuple(int(v) for v in 图标.split(","))
    if 沙盒:
        from 沙盒 import 沙盒世界, 沙盒安全源
        世界 = 沙盒世界()
        体 = 生命体(用语音=发声, 演练=True, 眼帧源=世界.帧, 手世界=世界.应用,
                    脑允许双=True, 脑允许回报=True)
        体.脑._安全探 = 沙盒安全源()
        世界帧 = 世界.帧
    else:
        体 = 生命体(用语音=发声, 演练=False,
                    脑允许双=True, 脑允许回报=True)            # 实机：真鼠标，目标+多巴胺开
        世界帧 = lambda: 体.眼.最近帧                           # 教学指认用眼的最新实拍帧

    # ── 上线（先上线再采样：帧源要有画面）──
    体.上线()
    time.sleep(2.0)
    色 = _主色采样(世界帧(), 点)
    print("教学指认：(%d,%d) 主色=%s 名称=%s" % (点[0], 点[1], 色, 名称))

    停 = dict(esc=False)

    def _急停监视():
        while not 体.到点.is_set():
            if _ESC按下():
                停["esc"] = True
                体.到点.set()
                return
            time.sleep(0.05)

    threading.Thread(target=_急停监视, daemon=True).start()

    遍s = []
    for k in range(遍):
        if 停["esc"] or 体.到点.is_set():
            break
        体.态.写目标(主色=色, 心=点, 来源="教学", 名称=名称)    # ① 指认（色+位+名）
        基准 = 体.脑.达成数
        t0 = time.time()
        while (time.time() - t0 < 遍秒 and 体.脑.达成数 == 基准
               and not 停["esc"] and not 体.到点.is_set()):
            time.sleep(0.5)
        用时 = round(time.time() - t0, 1)
        达成 = 体.脑.达成数 > 基准
        遍s.append(dict(遍=k + 1, 达成=bool(达成), 用时s=用时))
        print("第 %d 遍：%s（%.0fs）｜多巴胺总=%.1f 簿=%d 条"
              % (k + 1, "达成✓" if 达成 else "未达成", 用时, 体.脑.多巴胺总, len(体.脑.簿)))
        time.sleep(1.0)

    # ④ 检验：清目标 → 自发直达（图标簿：形状认得 → 直接知道在哪里）
    体.态.写目标(清=True)
    基准双击 = _自发双击计数(体)
    基准直达 = _直达计数(体)
    t0 = time.time()
    while (time.time() - t0 < 检验秒 and not 停["esc"] and not 体.到点.is_set()):
        time.sleep(0.5)
    自发 = _自发双击计数(体) - 基准双击
    直达 = _直达计数(体) - 基准直达
    print("检验（无目标 %.0fs）：自发双击 %d 次（其中**直达** %d 次——直接知道在哪里）｜"
          "多巴胺总=%.1f 达成=%d 簇=%d"
          % (检验秒, 自发, 直达, 体.脑.多巴胺总, 体.脑.达成数, len(体.脑.簿)))
    体.到点.set()
    time.sleep(1.0)

    果 = dict(程序="步8 教用WPS（教两三次就会：指认→寻找→停→双击→多巴胺→图标簿→自发直达）",
              模式=("沙盒" if 沙盒 else "实机"), 图标=点, 主色=list(色), 名称=名称,
              遍s=遍s, 检验自发双击=int(自发), 检验直达=int(直达),
              图标簿=[dict(名称=t.get("名称", ""), 心=list(t.get("心") or ()),
                           成功=t.get("成功", 0), 大变=t.get("大变", 0))
                      for t in 体.脑.簿.values()],
              多巴胺=dict(总量=round(体.脑.多巴胺总, 1), 达成数=体.脑.达成数),
              ESC=停["esc"])
    out = os.path.join(_HERE, "步7_教用WPS结果.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print("结果落盘：%s" % out)
    return 果


def _自发双击计数(体) -> int:
    """从台账数：出招 键=「双」的次数（教学/检验期均计数，差值=检验期自发量）。"""
    n = 0
    try:
        with open(os.path.join(_ROOT, "生命体台账.jsonl"), encoding="utf-8") as f:
            for ln in f:
                if '"键": "双"' in ln and '"文": "出招"' in ln:
                    n += 1
    except Exception:
        pass
    return n


def _直达计数(体) -> int:
    """从台账数：直达出招次数（直接知道在哪里——无漫游首招即中）。台账行 文="直达"。"""
    n = 0
    try:
        with open(os.path.join(_ROOT, "生命体台账.jsonl"), encoding="utf-8") as f:
            for ln in f:
                if '"文": "直达"' in ln:
                    n += 1
    except Exception:
        pass
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--图标", type=str, default="1480,540", help="被教图标的主屏坐标 x,y")
    ap.add_argument("--名称", type=str, default="WPS")
    ap.add_argument("--遍", type=int, default=3)
    ap.add_argument("--遍秒", type=float, default=120.0)
    ap.add_argument("--检验秒", type=float, default=90.0)
    ap.add_argument("--沙盒", action="store_true")
    ap.add_argument("--发声", action="store_true")
    a = ap.parse_args()
    主(图标=a.图标, 遍=a.遍, 遍秒=a.遍秒, 检验秒=a.检验秒, 沙盒=a.沙盒,
       发声=a.发声, 名称=a.名称)
