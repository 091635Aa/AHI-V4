# -*- coding: utf-8 -*-
"""test_步3_接线.py —— 步3 接线判据：器官_眼 接视场前端 + 元素s 双通道涌现 + 下线审计。

跑法：python test_步3_接线.py   （合成帧注入 _截屏，不碰真屏；确定性）
产物：步3_接线判据.json

判据（预登记于 步0_基线台账 §11.4 的"下一步"与 会议_视觉 00_会议纪要 §7 步3）：
  J1 视场槽每拍更新（注视链增长；视觉槽帧号/变化率/屏幕保字段）
  J2 元素s 保字段（句柄/框/心/字/主色/形状/置信；字恒空=C9；框心在屏内）
  J3 惊讶通道：局部变化 → 出现高置信结构，且与变化格重叠
  J4 确认通道：静帧注视确认 → 在场者出现且含嵌套对（语义.容器树 最大深度≥2）
  J5 句柄跨帧稳定（静帧连续三拍句柄集不变）
  J6 确定性：双实例同帧序列 → 视场/元素逐位一致
  J7 下线审计：meeting38 根目录 *.py 代码 token 内无 ultralytics/rapidocr/yolo/
     _检测视网膜/_补文/_普查（注释/docstring 剥离后扫描）
  J8 性能登记（不设门）：前端单拍 1080p 合成帧 均值/P95
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
根 = r"<workspace>/03_AHI\AHI2.1\meeting38"
sys.path.insert(0, HERE)
sys.path.insert(0, 根)

from 共存态 import 共存态                  # noqa: E402
from 器官_眼 import 眼器                    # noqa: E402
from 语义 import 语义图                     # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

H, W = 144, 192
格X, 格Y = 8, 6


def _帧(种子: int = 1, h: int = H, w: int = W) -> np.ndarray:
    rng = np.random.default_rng(种子)
    图 = np.full((h, w, 3), 120.0, dtype=np.float32)
    图 += rng.normal(0, 3.0, (h, w, 3))
    for i in range(4):
        y0, x0 = int(6 + i * 32), int(10 + i * 40)
        图[y0:y0 + 20, x0:x0 + 24] += rng.normal(0, 40.0, (20, 24, 3))
    return np.clip(图, 0, 255).astype(np.uint8)


def _改格(帧: np.ndarray, bx: int, by: int) -> np.ndarray:
    w = 帧.copy()
    h, wd = w.shape[:2]
    y0, y1 = int(by * h / 格Y), int((by + 1) * h / 格Y)
    x0, x1 = int(bx * wd / 格X), int((bx + 1) * wd / 格X)
    rng = np.random.default_rng(7000 + bx * 13 + by)
    w[y0:y1, x0:x1] = rng.integers(0, 256, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
    return w


class 帧源:
    def __init__(self, 帧s):
        self.帧s = list(帧s)
        self.i = 0

    def 下一个(self):
        f = self.帧s[min(self.i, len(self.帧s) - 1)]
        self.i += 1
        return f, (int(f.shape[1]), int(f.shape[0]))


def _新眼(帧s, 种子: int = 3800):
    态 = 共存态(种子=3800, 用gpu=False)
    器 = 眼器(态, None, 种子=种子)
    器._截屏 = 帧源(帧s).下一个
    return 态, 器


def _判(name, value, thr, passed, note=""):
    return dict(name=name, value=value, threshold=thr, passed=bool(passed), note=note)


# ── J1 视场槽每拍更新 + 视觉槽保字段 ──
def t1():
    静 = _帧(3)
    态, 器 = _新眼([静] * 6)
    for _ in range(6):
        器.一轮()
    视 = 态.取视觉()
    场 = 态.取视场()
    c = [
        _判("视场槽.注视链随拍增长", len(场["注视链"]), "≥6", len(场["注视链"]) >= 6),
        _判("视场槽.注视已写入", str(场["注视"]), "非空", 场["注视"] is not None),
        _判("视觉槽帧号=拍数", 视["帧号"], "==6", 视["帧号"] == 6),
        _判("视觉槽变化率/屏幕保字段", "%.3f/%s" % (视["变化率"], str(视["屏幕"])),
            "浮点/(W,H)", isinstance(视["变化率"], float) and tuple(视["屏幕"]) == (W, H)),
    ]
    return dict(route="J1", passed=all(x["passed"] for x in c), criteria=c)


# ── J2 元素s 保字段 + 字恒空 ──
def t2():
    静 = _帧(5)
    态, 器 = _新眼([静] * 5)
    for _ in range(5):
        器.一轮()
    元素s = 态.取视觉()["元素s"]
    需键 = {"句柄", "框", "心", "字", "主色", "形状", "置信"}
    齐 = all(需键 <= set(e) for e in 元素s)
    字空 = all(e.get("字") == "" for e in 元素s)
    在屏 = all(0 <= e["框"][0] and 0 <= e["框"][1] and e["框"][0] + e["框"][2] <= W
               and e["框"][1] + e["框"][3] <= H for e in 元素s)
    心对 = all(abs(e["心"][0] - (e["框"][0] + e["框"][2] // 2)) <= 1 and
               abs(e["心"][1] - (e["框"][1] + e["框"][3] // 2)) <= 1 for e in 元素s)
    c = [
        _判("元素非空（静帧确认通道）", len(元素s), "≥1", len(元素s) >= 1),
        _判("字段齐全", 齐, "全元素含7键", 齐),
        _判("字恒空（C9 禁 OCR）", 字空, "全空", 字空),
        _判("框/心在屏内且心=框中心", "在屏=%s 心对=%s" % (在屏, 心对), "均真",
            bool(在屏 and 心对)),
    ]
    return dict(route="J2", passed=all(x["passed"] for x in c), criteria=c)


# ── J3 惊讶通道：局部变化 → 高置信结构 + 与变化格重叠 ──
def t3():
    基 = _帧(7)
    目标 = (5, 2)
    帧s = [基, 基, _改格(基, *目标), 基]
    态, 器 = _新眼(帧s)
    最大置信, 命中 = 0.0, False
    for _ in range(4):
        器.一轮()
        元素s = 态.取视觉()["元素s"]
        万x0, 万y0 = int(目标[0] * W / 格X), int(目标[1] * H / 格Y)
        万x1, 万y1 = int((目标[0] + 1) * W / 格X), int((目标[1] + 1) * H / 格Y)
        for e in 元素s:
            x, y, w, h = e["框"]
            交 = max(0, min(x + w, 万x1) - max(x, 万x0)) * max(0, min(y + h, 万y1) - max(y, 万y0))
            if 交 > 0.3 * (w * h) and e["置信"] > 0.5:
                最大置信, 命中 = max(最大置信, e["置信"]), True
    c = [
        _判("变化拍出现高置信结构（惊讶区）", round(最大置信, 3), ">0.5", 最大置信 > 0.5),
        _判("结构与变化格重叠", 命中, "真", 命中),
    ]
    return dict(route="J3", passed=all(x["passed"] for x in c), criteria=c)


# ── J4 确认通道：静帧 → 在场者 + 嵌套对（容器树深度≥2） ──
def t4():
    静 = _帧(11)
    态, 器 = _新眼([静] * 5)
    for _ in range(5):
        器.一轮()
    元素s = 态.取视觉()["元素s"]
    嵌套 = False
    for a in 元素s:
        for b in 元素s:
            if a is b:
                continue
            wa, ha = a["框"][2], a["框"][3]
            wb, hb = b["框"][2], b["框"][3]
            if wa * ha <= wb * hb:
                continue
            if (a["框"][0] <= b["心"][0] <= a["框"][0] + wa
                    and a["框"][1] <= b["心"][1] <= a["框"][1] + ha):
                嵌套 = True
                break
        if 嵌套:
            break
    语 = 语义图(态)
    树 = 语.建容器树(元素s)
    深度 = 语.判据()["最大深度"]
    c = [
        _判("静帧在场者（确认通道）", len(元素s), "≥2", len(元素s) >= 2),
        _判("含嵌套对（窗⊃格）", 嵌套, "真", 嵌套),
        _判("语义容器树最大深度", 深度, "≥2（W4 前提）", 深度 >= 2),
    ]
    return dict(route="J4", passed=all(x["passed"] for x in c), criteria=c)


# ── J5 句柄跨帧稳定（静帧） ──
def t5():
    静 = _帧(13)
    态, 器 = _新眼([静] * 5)
    账 = []
    for _ in range(5):
        器.一轮()
        账.append([(e["句柄"], e["框"]) for e in 态.取视觉()["元素s"]])
    稳 = all(set(账[i]) == set(账[i + 1]) for i in (2, 3))
    c = [_判("静帧句柄/框集连续两拍不变", 稳, "真", 稳,
             "拍3→4、拍4→5 逐元素一致")]
    return dict(route="J5", passed=稳, criteria=c)


# ── J6 确定性：双实例逐位一致 ──
def t6():
    基 = _帧(17)
    帧s = [基, 基, _改格(基, 3, 4), 基, 基]
    态A, 器A = _新眼(帧s)
    态B, 器B = _新眼(帧s)
    链A, 链B = [], []
    for _ in range(5):
        器A.一轮()
        器B.一轮()
        链A.append(list(态A.取视场()["注视链"])[-1])
        链B.append(list(态B.取视场()["注视链"])[-1])
    eA = [(e["句柄"], e["框"], e["字"], e["置信"]) for e in 态A.取视觉()["元素s"]]
    eB = [(e["句柄"], e["框"], e["字"], e["置信"]) for e in 态B.取视觉()["元素s"]]
    ok = (链A == 链B) and (eA == eB)
    c = [_判("视场注视链 + 元素表逐位一致", ok, "真", ok)]
    return dict(route="J6", passed=ok, criteria=c)


# ── J7 下线审计（代码 token 扫描） ──
def _剥注释字符串(源码: str) -> str:
    import io
    import tokenize
    out = []
    try:
        for tk in tokenize.generate_tokens(io.StringIO(源码).readline):
            if tk.type in (tokenize.COMMENT, tokenize.STRING):
                out.append(" ")
            else:
                out.append(tk.string)
    except Exception:
        return 源码
    return " ".join(out)


def t7():
    禁式 = r"(?i)ultralytics|rapidocr|yolo|_检测视网膜|_补文|_普查"
    命中 = []
    for f in sorted(os.listdir(根)):
        if not f.endswith(".py"):
            continue
        with open(os.path.join(根, f), encoding="utf-8") as fh:
            码 = _剥注释字符串(fh.read())
        if re.search(禁式, 码):
            命中.append(f)
    # 接线反证：器官_眼 必须真的引到视场前端（不是靠删干净冒充）
    with open(os.path.join(根, "器官_眼.py"), encoding="utf-8") as fh:
        眼码 = fh.read()
    接线 = ("视场前端" in 眼码) and ("写视觉" in 眼码) and ("torch" not in _剥注释字符串(眼码))
    c = [
        _判("根目录 *.py 无 yolo/ocr/旧检测残留（token 级）", 命中, "空", not 命中),
        _判("器官_眼 已接视场前端且无 torch", 接线, "真", 接线),
    ]
    return dict(route="J7", passed=all(x["passed"] for x in c), criteria=c)


# ── J8 性能登记（不设门） ──
def t8():
    态 = 共存态(种子=3800, 用gpu=False)
    from 视场前端 import 视场前端
    前端 = 视场前端(态, 种子=3800)
    大 = _帧(23, h=1080, w=1920)
    前端.一拍(大, (1920, 1080))
    时s = []
    import time as _t
    for _ in range(6):
        t0 = _t.perf_counter()
        前端.一拍(大, (1920, 1080))
        时s.append((_t.perf_counter() - t0) * 1000)
    均, p95 = float(np.mean(时s)), float(np.percentile(时s, 95))
    过 = p95 <= 200.0                     # 仅防爆表；正式门槛（33/67ms）由 步3 台账登记
    c = [_判("前端单拍 1080p（登记）", "均值%.1fms P95=%.1fms" % (均, p95),
             "≤200（登记项）", 过)]
    return dict(route="J8", passed=过, criteria=c)


def main():
    组 = [("J1视场槽/视觉槽", t1), ("J2元素保字段", t2), ("J3惊讶通道", t3),
          ("J4确认通道+嵌套", t4), ("J5句柄稳定", t5), ("J6确定性", t6),
          ("J7下线审计", t7), ("J8性能登记", t8)]
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
            print("  %s %-36s = %-28s 要求 %s" %
                  ("PASS" if c["passed"] else "FAIL", c["name"], c["value"], c["threshold"]))
        if r.get("error"):
            print("  ERROR:", r["error"]); print(r.get("tb", "")[:1200])
    print("\n总计：%s" % ("全部通过" if 全过 else "存在失败"))
    json.dump(结果, open(os.path.join(HERE, "步3_接线判据.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    return 0 if 全过 else 1


if __name__ == "__main__":
    sys.exit(main())