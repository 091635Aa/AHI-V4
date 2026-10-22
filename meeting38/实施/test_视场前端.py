# -*- coding: utf-8 -*-
"""meeting38/实施/test_视场前端.py —— 步2 判据：像人-1~5（会议纪要 §5；可真失败）。

跑法：python test_视场前端.py    （纯合成帧，不碰真屏；确定性）
产物：步2_视场判据.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1\meeting38")

from 共存态 import 共存态                     # noqa: E402
from 视场前端 import 视场前端, 格X, 格Y        # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

H, W = 144, 192                                # 合成帧（小块，快且确定性）


def _帧(种子: int = 1, h: int = H, w: int = W) -> np.ndarray:
    """确定性合成帧：低对比底 + 若干纹理块。"""
    rng = np.random.default_rng(种子)
    图 = np.full((h, w, 3), 120.0, dtype=np.float32)
    图 += rng.normal(0, 3.0, (h, w, 3))        # 微噪声
    for i in range(4):                          # 4 个纹理块
        y0, x0 = int(6 + i * 32), int(10 + i * 40)
        图[y0:y0 + 20, x0:x0 + 24] += rng.normal(0, 40.0, (20, 24, 3))
    return np.clip(图, 0, 255).astype(np.uint8)


def _局部改(帧: np.ndarray, bx: int, by: int) -> np.ndarray:
    """在格 (bx,by) 区域注入明显结构变化。"""
    w = 帧.copy()
    h, wd = w.shape[:2]
    y0, y1 = int(by * h / 格Y), int((by + 1) * h / 格Y)
    x0, x1 = int(bx * wd / 格X), int((bx + 1) * wd / 格X)
    rng = np.random.default_rng(9000 + bx * 31 + by)
    补 = rng.integers(0, 256, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
    w[y0:y1, x0:x1] = 补
    return w


def _明度(帧: np.ndarray, 系数: float) -> np.ndarray:
    return np.clip(帧.astype(np.float32) * 系数, 0, 255).astype(np.uint8)


def _auc(正: list, 负: list) -> float:
    """秩 AUC（正>负 的比例；并列计 0.5）。"""
    n = 0.0
    for a in 正:
        for b in 负:
            n += 1.0 if a > b else (0.5 if a == b else 0.0)
    return n / max(len(正) * len(负), 1)


def _判(name, value, thr, passed, note=""):
    return dict(name=name, value=value, threshold=thr, passed=bool(passed), note=note)


# ══════════ 像人-1 单帧决定性：输出必须依赖注视历史 ══════════
def test_1():
    f = _帧(3)
    前端 = 视场前端(共存态(种子=3800, 用gpu=False))
    前端.一拍(f)                                # 暖机，建立上帧块码
    # 同一帧、两种历史：A 刚看左（抑制左），B 刚看右（抑制右）
    A = 视场前端(共存态(种子=3800, 用gpu=False)); A.一拍(f)
    B = 视场前端(共存态(种子=3800, 用gpu=False)); B.一拍(f)
    A.注视, B.注视 = (0, 3), (格X - 1, 3)
    A.抑制[:] = 0; B.抑制[:] = 0
    A.抑制[3 * 格X + 0] = 1.0
    B.抑制[3 * 格X + (格X - 1)] = 1.0
    逐格 = A.误差场(A.粗码s(f))[1]
    gA = A.下一注视(逐格)                        # A 的抑制在左 → 不往左
    gB = B.下一注视(逐格)
    位移列 = abs(gA[0] - gB[0])
    场A = A.视场码(A.粗码s(f), A.细码s(f)); 场B = B.视场码(B.粗码s(f), B.细码s(f))
    cos = float(场A @ 场B / (np.linalg.norm(场A) * np.linalg.norm(场B) + 1e-9))
    过 = (位移列 >= 0.25 * 格X) or (cos <= 0.8)
    c = [_判("单帧决定性（注视历史改变下一注视）", "Δ列=%d, cos=%.3f" % (位移列, cos),
             "Δ列≥2 或 cos≤0.8", 过, "AI 式看：输出由单帧决定，此处必失败")]
    return dict(route="像人-1", passed=过, criteria=c)


# ══════════ 像人-2 变化优先 + 适应（全局例外 vs 局部） ══════════
def test_2():
    base = _帧(5)
    前端A = 视场前端(共存态(种子=3800, 用gpu=False))
    前端A.一拍(_明度(base, 1.00))
    意外A = []
    for t in range(1, 21):                      # 全局明度渐变（可预测）
        意外A.append(前端A.一拍(_明度(base, 1.00 + 0.01 * t))["意外度"])
    跳A = 前端A.眼跳数
    前端B = 视场前端(共存态(种子=3800, 用gpu=False))
    前端B.一拍(base)
    目标 = (5, 2)
    f2 = _局部改(base, *目标)
    前端B.注视 = 目标
    前端B.注视停留 = 0
    前端B.抑制[:] = 0
    逐格B = 前端B.误差场(前端B.粗码s(f2))[1]
    g = 前端B.下一注视(逐格B)
    命中 = int(g == 目标)
    峰 = float(逐格B.max())
    适应过 = (float(np.mean(意外A)) <= 0.2) and (跳A <= 2)     # 可预测的全局变化不应诱发眼跳
    意外峰过 = 峰 >= 0.5
    局过 = 命中 == 1
    c = [
        _判("适应：全局渐变几乎不产生意外/眼跳", "意外均值=%.4f 眼跳=%d" % (float(np.mean(意外A)), 跳A),
            "意外≤0.2 且 跳≤2", 适应过, "光适应（局部归一化）+注视保持律"),
        _判("变化优先：局部移动产生意外峰", round(峰, 4), "≥0.5", 意外峰过),
        _判("变化优先：下一注视落到变化处", "命中" if 局过 else "未命中", "命中", 局过),
    ]
    return dict(route="像人-2", passed=all(x["passed"] for x in c), criteria=c)


# ══════════ 像人-3 不依赖物体列表 ══════════
def test_3():
    态 = 共存态(种子=3800, 用gpu=False)
    前端 = 视场前端(态)
    态.写视觉([])                                # 视觉槽元素s 保持空
    base = _帧(7)
    前端.一拍(base)
    无变 = [前端.一拍(base)["最意外"] for _ in range(6)]          # 连续无变化 → 视而不见
    有变 = []
    for i in range(6):                                           # 每次回到基准，再注入一处变化
        前端.一拍(base)
        有变.append(前端.一拍(_局部改(base, i, i % 格Y))["最意外"])
    链 = 态.取视场()["注视链"]
    元素s = 态.取视觉().get("元素s")
    峰有变, 峰无变 = float(np.mean(有变)), float(np.mean(无变))
    过 = (峰有变 >= 0.5) and (峰无变 <= 0.15) and (len(链) >= 8) and (len(元素s) == 0)
    c = [
        _判("有变化帧 峰值意外", round(峰有变, 4), "≥0.5", 峰有变 >= 0.5),
        _判("无变化帧 峰值意外（视而不见）", round(峰无变, 4), "≤0.15", 峰无变 <= 0.15),
        _判("注视链持续增长", len(链), "≥8", len(链) >= 8),
        _判("视觉槽.元素s 仍为空（看 不依赖它）", len(元素s), "==0", len(元素s) == 0),
    ]
    return dict(route="像人-3", passed=过, criteria=c)


# ══════════ 像人-4 误差驱动注视命中率 ══════════
def test_4():
    基 = _帧(11)
    前端 = 视场前端(共存态(种子=3800, 用gpu=False))
    前端.一拍(基)
    命中 = 0
    试验 = 50
    rng = np.random.default_rng(404)
    for i in range(试验):
        bx = int(rng.integers(0, 格X)); by = int(rng.integers(0, 格Y))
        前端.注视 = (int(rng.integers(0, 格X)), int(rng.integers(0, 格Y)))
        前端.抑制[:] = 0
        f = _局部改(基, bx, by)
        逐格 = 前端.误差场(前端.粗码s(f))[1]
        g = 前端.下一注视(逐格)
        命中 += int(g == (bx, by))
    p = 命中 / 试验
    随机 = 1.0 / (格X * 格Y)
    c = [_判("误差驱动注视命中率", round(p, 4), "≥0.80（随机≈%.3f）" % 随机, p >= 0.80)]
    return dict(route="像人-4", passed=p >= 0.80, criteria=c)


# ══════════ 像人-5 预测收敛 + 新旧可分 ══════════
def test_5():
    结构 = _帧(13)
    前端 = 视场前端(共存态(种子=3800, 用gpu=False))
    前端.一拍(结构)
    负 = [前端.一拍(结构)["意外度"] for _ in range(20)]           # 反复看同一结构 → 意外≈0
    正 = []
    for i in range(10):                                          # 10 个"首次见到"的新结构
        f = _局部改(结构, int(4 + i % 4), int(i % 6))
        正.append(前端.一拍(f)["意外度"])
    收敛比 = float(np.mean(负[-5:]) / (np.mean(正[:3]) + 1e-9))   # 末尾熟悉度 / 首次新异
    新vs旧 = _auc(正, 负[-10:])
    过 = (收敛比 <= 0.4) and (新vs旧 >= 0.90)
    c = [
        _判("预测收敛比（熟悉后/首次新异）", round(收敛比, 4), "≤0.4", 收敛比 <= 0.4,
            "熟悉末=%.4f  首次新异=%.4f" % (float(np.mean(负[-5:])), float(np.mean(正[:3])))),
        _判("新旧结构可分 AUC", round(新vs旧, 4), "≥0.90", 新vs旧 >= 0.90),
    ]
    return dict(route="像人-5", passed=过, criteria=c)


def main():
    组 = [("像人-1", test_1), ("像人-2", test_2), ("像人-3", test_3),
          ("像人-4", test_4), ("像人-5", test_5)]
    结果, 全过 = {}, True
    for 名, fn in 组:
        try:
            r = fn()
        except Exception as e:
            import traceback
            r = dict(route=名, passed=False, criteria=[],
                     error="%s: %s" % (type(e).__name__, e), tb=traceback.format_exc())
        结果[名] = r
        全过 &= bool(r.get("passed"))
        print("\n[%s] -> %s" % (名, "PASS" if r.get("passed") else "FAIL"))
        for c in r.get("criteria", []):
            print("  %s %-34s = %-16s 要求 %s %s" %
                  ("PASS" if c["passed"] else "FAIL", c["name"], c["value"],
                   c["threshold"], c.get("note", "")))
        if r.get("error"):
            print("  ERROR:", r["error"]); print(r.get("tb", ""))
    print("\n总计：%s" % ("全部通过" if 全过 else "存在失败"))
    json.dump(结果, open(os.path.join(HERE, "步2_视场判据.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    return 0 if 全过 else 1


if __name__ == "__main__":
    sys.exit(main())