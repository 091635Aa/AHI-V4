# -*- coding: utf-8 -*-
"""test_模块六_画面.py —— V3 判据：第六层「画面 / 视觉想象」（所有者点名的那一件）。

验的是**机制真接线且真能读出**，不是"文件在不在"：

  N6-1 装配（第六层真接上）：`内核宿主.造桥()` 的 MRO 里有 `顶覆盖_综合体H`（链顶），
       且**五层仍在**（不破既有）；`画面开关=True` 时 `画面核` 就位。
  N6-2 写码（真屏/N6 帧 → 码）：一屏能选出结构件并建成物体码；**件数 ≤ 容量上限**。
  N6-3 回放（从码里解出画面）：位置地址解码的保真 ≥ 判据，且**身份正确**（argmin 指向自己）；
       并给出**空位虚警**对照（空位最高码似必须低于判据）。
  N6-4 想象（组合，不是回放）：把记忆里的物件**环面搬家**到它没待过的格位后，
       ① 仍认得出（码似 ≥ 判据）；② 新旧格位组合确实没出现过（新组合格数 > 0）。
  N6-5 想象不污染（provenance）：想象/回忆**不入画面库**——库大小只随"实感"增；
       且想象喂自模型时 `更新=False` ⇒ 自模型 W 逐位不变（自编的东西不算经历）。
  N6-6 画面记忆有界：‖W画‖ 恒为上限（与 `W_self` 同律），方向仍随屏变。
  N6-7 学习≠记忆 + C13 确定性：画面迹/库不在 `codec._段`；同 seed 同帧两遍**逐位同**。

跑法：`python test_模块六_画面.py`（不设 PYTHONIOENCODING 也应 exit 0）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))       # meeting38/实施
M38 = os.path.dirname(HERE)
AHI3 = os.path.dirname(M38)
for p in (HERE, M38, AHI3, os.path.join(AHI3, "meeting31"), os.path.join(AHI3, "meeting39", "实施")):
    if p not in sys.path:
        sys.path.insert(0, p)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _载实现():
    """按文件路径加载实现版 `V3主程序`（AHI3 根还有一个同名薄壳启动器）。"""
    路径 = os.path.join(HERE, "V3主程序.py")
    for k in [k for k in sys.modules if k == "V3主程序"]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location("V3主程序_实现", 路径)
    模 = importlib.util.module_from_spec(spec)
    sys.modules["V3主程序_实现"] = 模
    spec.loader.exec_module(模)
    return 模


V = _载实现()
import 内核宿主 as HZ                                            # noqa: E402
from 内核_画面 import 判据_检测, 上限件                            # noqa: E402


def 造屏(种子: int = 7, H: int = 144, W: int = 256, 位移: int = 0):
    """合成屏：7 件**彼此分离、彼此不同**的结构（可整体平移做"新场景"）。"""
    rng = np.random.default_rng(种子)
    图 = np.full((H, W, 3), 40, np.float32)
    件 = [(16, 16, 32, 32, (230, 60, 60)), (80, 16, 32, 32, (60, 200, 90)),
          (144, 16, 32, 32, (80, 120, 240)), (208, 16, 32, 32, (240, 220, 40)),
          (16, 96, 64, 16, (250, 250, 250)), (112, 96, 32, 48, (20, 20, 20)),
          (192, 96, 48, 32, (150, 90, 200))]
    for x, y, w, h, c in 件:
        x2 = int(np.clip(x + 位移, 0, W - w))
        图[y:y + h, x2:x2 + w] = np.asarray(c, np.float32)
    return 图 + rng.normal(0, 3, 图.shape).astype(np.float32)


def 取真屏():
    try:
        import mss
        with mss.mss() as sct:
            return np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]
    except Exception:
        return None


def 真外观(核, 帧, 位s):
    """该格的光适应块（与编码同源）→ 单位化，当"真外观"对照。"""
    块 = 核.块s(帧, 带对比=True)
    出 = []
    for k in 位s:
        x = 块[k][0]
        n = float(np.linalg.norm(x))
        出.append(x / (n if n > 1e-9 else 1.0))
    return np.stack(出, 0)


def 造屏2(种子: int = 50, H: int = 144, W: int = 256):
    """两件**同瞥**（互为冗余，会被去冗余合并）+ 若干分离件 ⇒ 必然产出多瞥多场码。
    用于验"多瞥场码与联想阵"（单瞥单件的屏场范数恒为 1，会让断言失去鉴别力）。"""
    rng = np.random.default_rng(种子)
    图 = np.full((H, W, 3), 40, np.float32)
    件 = [(8, 8, 16, 16, (230, 60, 60)), (40, 8, 16, 16, (230, 60, 60)),
          (100, 20, 24, 24, (60, 200, 90)), (180, 30, 24, 24, (80, 120, 240)),
          (16, 100, 32, 24, (240, 220, 40))]
    for x, y, w, h, c in 件:
        图[y:y + h, x:x + w] = np.asarray(c, np.float32)
    return 图 + rng.normal(0, 2, 图.shape).astype(np.float32)


def main():
    报 = {}

    # ── N6-1 装配：第六层在链顶，五层仍在 ──
    命 = V.V3生命(种子=310)
    ok起 = 命.起综合体()
    桥 = 命.桥
    mro名 = [c.__name__ for c in type(桥).__mro__]
    五层 = ["顶覆盖_综合体W", "顶覆盖_综合体Z", "顶覆盖_综合体L", "顶覆盖_综合体T", "顶覆盖_综合体"]
    六层 = "顶覆盖_综合体H"
    核 = getattr(桥, "画面核", None)
    N6_1 = bool(ok起 and 六层 in mro名 and mro名[0] == 六层 and all(x in mro名 for x in 五层)
                and 核 is not None and type(核).__module__ == "内核_画面")
    接帧 = 命.接画面()

    # ── 用真帧（真屏优先；取不到用合成屏）建一屏 ──
    帧 = 取真屏()
    来源 = "真屏"
    if 帧 is None:
        帧 = 造屏()
        来源 = "合成屏"
    记 = 核.记一屏(帧, 拍=1)
    位s = [tuple(x) for x in 记.get("位s", [])]

    # ── N6-2 写码：能选出件且不超容量 ──
    N6_2 = bool(记.get("记") and 0 < int(记.get("物体数", 0)) <= int(上限件))

    # ── N6-3 回放：保真 + 身份 + 空位虚警 ──
    回 = 核.回放() if 记.get("记") else dict(命中数=0, 重建={}, 码似={})
    真 = 真外观(核, 帧, 位s) if 位s else np.zeros((0, 12), np.float32)
    外 = np.asarray(核.库[-1]["外观s"]) if 核.库 else np.zeros((0, 12), np.float32)
    保, 对, 真码 = [], 0, []
    for i, k in enumerate(位s):
        rec = 回["重建"][k][1]
        rn = float(np.linalg.norm(rec))
        保.append(float(真[i] @ rec / (rn + 1e-12)))
        真码.append(float(回["每格码似"][k]))
        if 外.shape[0] > 0:
            对 += int(int(np.argmax(外 @ rec / (np.linalg.norm(外, axis=1) * rn + 1e-12))) == i)
    瞥_位 = 核.分瞥() if 记.get("记") else {}
    空 = [k for k in [(bx, by) for by in range(核.格y) for bx in range(核.格x)]
          if k not in set(位s) and (not 瞥_位 or 瞥_位[k] in 核.库[-1]["场s"])] if 核.库 else []
    r2 = (核.读出(核.库[-1]["场s"], 码库=核.库[-1]["码s"], 位置s=空,
                 外观库=核.库[-1].get("外观s")) if 空 else dict(每格码似={}))
    空高 = max([float(r2["每格码似"][k]) for k in 空], default=0.0)
    N6_3 = bool(位s and float(np.mean(保)) >= 0.9 and 对 == len(位s)
                and float(np.min(真码)) >= float(判据_检测) and 空高 < float(判据_检测))

    # ── N6-4 想象（环面搬家）：认得出 ∧ 是新组合 ──
    im = 核.想象码(位移=(3, 2)) if 核.库 else dict(想象=False)
    认, 想低 = 0, 0.0
    if im.get("想象"):
        r3 = 核.读出(im["场s"], 码库=核.库[-1]["码s"], 位置s=list(im["位s"]),
                     外观库=核.库[-1].get("外观s"), 瞥号s=list(im["瞥号s"]))
        想s = [float(r3["每格码似"][k]) for k in im["位s"]]
        认 = sum(1 for x in 想s if x >= float(判据_检测))
        想低 = float(np.min(想s)) if 想s else 0.0
    N6_4 = bool(im.get("想象") and 认 == len(im.get("位s", [])) and im.get("新组合格数", 0) > 0
                and not set(im.get("位s", [])) == set(位s))

    # 单件搬家（"把这一件搬到它从没待过的地方"）
    单 = 核.想象(0) if 核.库 else dict(想象=False)
    N6_4b = bool(单.get("想象") and 单.get("认得") and 单.get("新组合"))

    # ── N6-5 想象不污染：库只随实感增；想象喂自模型不改 W ──
    库0 = len(核.库)
    for _ in range(3):
        核.想象码(位移=(1, 1))
        核.想象(0)
    核.回放()
    库1 = len(核.库)
    N6_5a = bool(库1 == 库0)
    zm = getattr(桥, "自模型", None)
    # 先喂几拍让 ĝ 有 W
    HZ.设来源(桥, 自源=True)
    for s in ("先看这一屏", "再想它一下", "记住这几件"):
        桥.步_向量(命._感知向量(s), 原文=None)
    喂 = dict(喂=False)
    W0 = None
    if zm is not None and getattr(zm, "W", None) is not None:
        W0 = np.array(zm.W, copy=True)
        喂 = 桥.喂想象给自模型(位移=(3, 2), 更新=False)
    W差 = 0.0
    if W0 is not None and getattr(zm, "W", None) is not None:
        W差 = float(np.max(np.abs(np.asarray(zm.W) - W0)))
    N6_5b = bool(喂.get("喂") and W差 == 0.0)

    # ── N6-6 画面联想阵：默认**不背**（写W画=False）；开成实验臂时有界且方向仍变 ──
    #   为什么默认关：实测它①不在读出路径上（每瞥 46.6ms×9 ≈ 373ms/屏 纯亏）
    #   ②归一到单位 Frobenius 后 ‖W·v‖ 比 ‖v‖ 小 23 倍 ⇒ 联想读出 cos 0.002（负增益）。
    #   见 `测试记录/_预检_画面联想阵.py`、`_诊断_画面联想阵.py`。
    范s, 向s = [], []
    for i in range(4):
        核.记一屏(造屏2(种子=20 + i), 拍=10 + i)
    N6_6a = bool(float(np.linalg.norm(核.W画)) == 0.0 and not bool(getattr(核, "写W画", True)))
    核2 = type(核)(D=1024, 种子=3800, 写W画_=True)
    for i in range(4):
        核2.记一屏(造屏2(种子=30 + i), 拍=i)
        n6 = float(np.linalg.norm(核2.W画))
        范s.append(n6)
        向s.append(np.array(核2.W画 / max(n6, 1e-12), copy=True))
    有界 = all(abs(x - 1.0) <= 1e-6 for x in 范s)
    向变 = max((float(np.abs(向s[i + 1] - 向s[i]).max()) for i in range(len(向s) - 1)),
              default=0.0)
    N6_6 = bool(N6_6a and 有界 and 向变 > 0.0)

    # ── N6-8 写码开销（性能判据；曾因逐格重算 + 每瞥 1024² 外积掉到 683ms/屏）──
    import time as _t
    帧8 = 造屏(种子=77)
    核3 = type(核)(D=1024, 种子=3800)
    _t0 = _t.time()
    for _ in range(3):
        核3.记一屏(帧8, 拍=1)
    记ms = (_t.time() - _t0) / 3 * 1000
    N6_8 = bool(记ms <= 200.0)

    # ── N6-7 学习≠记忆 + 确定性 ──
    段0 = len(桥.codec._段)
    核.记一屏(造屏(种子=33), 拍=99, 来路="想象")
    段1 = len(桥.codec._段)
    N6_7a = bool(段0 == 段1)
    核A = type(核)(D=1024, 种子=3800)
    核B = type(核)(D=1024, 种子=3800)
    帧A = 造屏(种子=41)
    rA = 核A.记一屏(帧A, 拍=1)
    rB = 核B.记一屏(帧A, 拍=1)
    同 = bool(rA.get("记") and rB.get("记")
              and list(rA["位s"]) == list(rB["位s"])
              and float(np.max(np.abs(np.asarray(rA["外观s"]) - np.asarray(rB["外观s"])))) == 0.0)
    W同 = bool(np.max(np.abs(核A.W画 - 核B.W画)) == 0.0)
    N6_7b = bool(同 and W同)

    态 = 命.内核态()
    报 = dict(
        N6_1_装配第六层=dict(过=bool(N6_1), 链顶=mro名[0], 五层全在=all(x in mro名 for x in 五层),
                          画面核=type(核).__name__, 帧源=bool(接帧)),
        N6_2_写码=dict(过=bool(N6_2), 来源=来源, 帧形=tuple(np.asarray(帧).shape),
                      件数=int(记.get("物体数", 0)), 瞥数=int(记.get("瞥数", 0)),
                      上限件=int(上限件)),
        N6_3_回放=dict(过=bool(N6_3), 件数=len(位s),
                      保真均=(round(float(np.mean(保)), 4) if 保 else None),
                      保真低=(round(float(np.min(保)), 4) if 保 else None),
                      身份对=对, 真位码似低=(round(float(np.min(真码)), 4) if 真码 else None),
                      空位码似高=round(空高, 4), 判据=float(判据_检测),
                      回放命中=回.get("命中数")),
        N6_4_想象=dict(过=bool(N6_4 and N6_4b), 整体搬家认=("%d/%d" % (认, len(im.get("位s", [])))),
                      想象码似低=round(想低, 4), 新组合格数=im.get("新组合格数"),
                      单件认得=单.get("认得"), 单件新组合=单.get("新组合"),
                      单件cos=(None if 单.get("cos") is None else round(float(单["cos"]), 4)),
                      位移=[3, 2]),
        N6_5_想象不污染=dict(过=bool(N6_5a and N6_5b), 库前=库0, 库后=库1,
                          喂自模型=喂.get("喂"), 自模型W_maxΔ=W差,
                          ε_self=(None if 喂.get("ε_self") is None else round(float(喂["ε_self"]), 4))),
        N6_6_画面联想阵=dict(过=bool(N6_6), 默认不背W画=N6_6a,
                         开臂范数s=[round(x, 6) for x in 范s], 方向仍变=round(向变, 6),
                         说明="默认关：实测它不在读出路径上且联想增益为负（见文件头）"),
        N6_8_写码开销=dict(过=bool(N6_8), 记一屏ms=round(记ms, 1), 门限ms=200.0),
        N6_7_学习非记忆与确定性=dict(过=bool(N6_7a and N6_7b), 段前=段0, 段后=段1,
                             同seed同帧逐位同=同, W画逐位同=W同),
        内核态=态,
    )
    全过 = all(报[k]["过"] for k in 报 if k.startswith("N6_"))
    报["全过"] = bool(全过)
    with open(os.path.join(HERE, "test_模块六_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(报, ensure_ascii=False, indent=1, default=str))
    return 全过


def _短(x):
    return x


if __name__ == "__main__":
    ok = main()
    print("\n[模块六·画面/视觉想象] %s" % ("全过 ✅" if ok else "未全过 ❌"))
