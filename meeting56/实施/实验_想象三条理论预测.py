# -*- coding: utf-8 -*-
"""实验_想象三条理论预测.py —— 把「人脑想象画面＝离线建构」的预测加工解释，落到 AHI4 现有
`内核_画面`（HDC 生成式画面核）上，做三条**可直接证伪**的实测。

背景（为什么要做这个实验）：
  人脑想象不是"抽一丝视觉映射到某个地方"，而是①**重用同一套视觉网络**（视觉皮层被自上而下
  信号重新激活）、②比真实看见**更弱、更不稳**、③**没有屏幕/投影位置**（分布式表征，位置只是
  读取地址）、④是**组合/离线建构**而不是回放。`内核_画面` 恰好把这四件事都做成了可测量的量：
  编码 P / 反投影 Pᵀ、位置当读地址、组合想象（环面搬家）、`来路` provenance。

本实验**只新建文件、不改任何既有件**（C14）；全部用**合成屏**（不用真屏截图），确保 C13 可复现。

三条实测臂（设计见同目录 ../00_应用_想象画面_从人脑到AHI4.md）：
  P1 同一网络复用：感知解码 vs 想象解码走**同一** P/Pᵀ/码库；比较逐件码似。
  P2 精度（信噪比）→ 可辨识：给想象场码按比例加噪，扫"想象强度"，找认出阈（心盲症类比）。
  P3 地址 ≠ 内容：同一想象场码在**正确地址**解得 0.9+，在**错误地址**≈0 ⇒ 位置是读取键，
     不是"画面被投影到的那一格"；全局画面不存在（空位解不出内容）。
  P4（对照，复用既有定义）组合想象：搬家后仍认得 ∧ 组合确实是新的。
  P5 源监控：同一物体在"实感场"与"想象场"里用的是**同一个码** ⇒ 来路不在内容里，
     必须由 provenance / 两区室 A 差分给出（结构化事实，不设 PASS/FAIL）。

跑法：`python 实验_想象三条理论预测.py`（约 10–20 秒，无网络、无外挂模型、固定种子）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                 # AHI4/meeting56_想象画面/实施
AHI4 = os.path.dirname(os.path.dirname(HERE))                    # AHI4
M39 = os.path.join(AHI4, "meeting39", "实施")                     # AHI4/meeting39/实施
if M39 not in sys.path:
    sys.path.insert(0, M39)

from 内核_画面 import 画面核, 判据_检测, 光适应                      # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

种子_核 = 3800
种子_屏 = 7
噪声种子 = 5601


def 造屏(种子: int = 种子_屏, H: int = 144, W: int = 256) -> np.ndarray:
    """合成屏：7 件彼此分离、彼此不同颜色的结构（与 test_模块六 同源；确定性）。"""
    rng = np.random.default_rng(种子)
    图 = np.full((H, W, 3), 40, np.float32)
    件 = [(16, 16, 32, 32, (230, 60, 60)), (80, 16, 32, 32, (60, 200, 90)),
          (144, 16, 32, 32, (80, 120, 240)), (208, 16, 32, 32, (240, 220, 40)),
          (16, 96, 64, 16, (250, 250, 250)), (112, 96, 32, 48, (20, 20, 20)),
          (192, 96, 48, 32, (150, 90, 200))]
    for x, y, w, h, c in 件:
        图[y:y + h, x:x + w] = np.asarray(c, np.float32)
    return 图 + rng.normal(0, 3, 图.shape).astype(np.float32)


def _r(x, n: int = 4):
    return None if x is None else round(float(x), n)


def _逐件码似(核, 场s, 位s, 瞥号s, 码s, 外观s):
    r = 核.读出(场s, 码库=码s, 位置s=list(位s), 外观库=外观s,
               判据=判据_检测, 瞥号s=(list(瞥号s) if 瞥号s is not None else None))
    return [float(r["每格码似"][k]) for k in 位s]


def 主跑():
    报 = {"设置": dict(D=1024, 种子_核=种子_核, 种子_屏=种子_屏, 判据_检测=float(判据_检测),
                     屏="合成屏 144x256（不用真屏，保证 C13）")}
    核 = 画面核(D=1024, 种子=种子_核)
    记 = 核.记一屏(造屏(), 拍=1)
    屏 = 核.库[-1]
    位s = [tuple(x) for x in 屏["位s"]]
    码s, 外观s, 瞥号s = 屏["码s"], 屏["外观s"], list(屏["瞥号s"])
    件数 = len(位s)

    # ── P1 同一网络复用：感知 vs 想象（同一 P/Pᵀ/同一码库）──
    #    ★注意（实测发现的既有件陷阱）：`记一屏` 返回的 `屏["瞥号s"]` 是**按瞥分组后的顺序**，
    #      与 `屏["位s"]` **不逐项对齐**（实测 16 件：用分瞥正确 0.9010、误用屏瞥号 0.4609）。
    #      所以感知解码**不能**把 屏["瞥号s"] 当逐项对齐参数传；想象解码用 `想象码` 自带的
    #      `im["瞥号s"]`（那一路是逐项对齐的）。
    感 = _逐件码似(核, 屏["场s"], 位s, None, 码s, 外观s)          # 走 分瞥()，即编码时的瞥
    im = 核.想象读出(位移=(3, 2))
    想 = list(im["逐件码似"])
    位置对齐检验 = dict(
        按分瞥=_r(np.mean(感)), 按屏瞥号s=_r(np.mean(_逐件码似(
            核, 屏["场s"], 位s, list(屏["瞥号s"]), 码s, 外观s))),
        说明="屏['瞥号s'] 与 屏['位s'] 不同序；逐项对齐必须用 想象码 返回的 瞥号s")
    P1 = dict(
        件数=件数,
        感知_均=_r(np.mean(感)), 感知_低=_r(np.min(感)),
        想象_均=_r(np.mean(想)), 想象_低=_r(np.min(想)),
        差值_均=_r(np.mean(np.asarray(感) - np.asarray(想))),
        感知认出率=_r(np.mean(np.asarray(感) >= 判据_检测)),
        想象认出率=_r(np.mean(np.asarray(想) >= 判据_检测)),
        位置对齐检验=位置对齐检验,
        同一网络=dict(编码="P（固定随机投影，未训练）", 解码="Pᵀ/√D",
                      码库="同一个库码（想象不新建物体码）", 说明="想象没有自己的视觉网络"),
    )

    # ── P2 精度扫描：给场码按比例加噪 → 还认不认得出（感知与想象两条曲线）──
    def 加噪读出(场s0, 位0, 瞥0, c, rng):
        场s = {}
        for g, v in 场s0.items():
            n = rng.normal(size=v.shape).astype(np.float32)
            n = n / (float(np.linalg.norm(n)) + 1e-12) * float(c) * float(np.linalg.norm(v))
            场s[g] = (v + n).astype(np.float32)
        return _逐件码似(核, 场s, 位0, 瞥0, 码s, 外观s)

    def 翻位读出(场s0, 位0, 瞥0, p, rng):
        """HDC 天然的精度退化：按比例翻转场码的位（与 联去想噪 的"带噪回忆"同口径）。"""
        场s = {}
        for g, v in 场s0.items():
            v2 = np.asarray(v, dtype=np.float32).copy()
            if p > 0:
                掩 = rng.random(v2.shape[0]) < float(p)
                v2[掩] *= -1.0
            场s[g] = v2
        return _逐件码似(核, 场s, 位0, 瞥0, 码s, 外观s)

    rng = np.random.default_rng(噪声种子)
    扫 = []
    for c in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
        似感 = 加噪读出(屏["场s"], 位s, None, c, rng)
        似想 = 加噪读出(im["场s"], list(im["位s"]), list(im["瞥号s"]), c, rng)
        扫.append(dict(噪比=float(c),
                       感知_均=_r(np.mean(似感)), 感知认出率=_r(np.mean(np.asarray(似感) >= 判据_检测)),
                       想象_均=_r(np.mean(似想)), 想象认出率=_r(np.mean(np.asarray(似想) >= 判据_检测))))
    扫翻 = []
    rngF = np.random.default_rng(噪声种子 + 1)
    for p in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        似感 = 翻位读出(屏["场s"], 位s, None, p, rngF)
        似想 = 翻位读出(im["场s"], list(im["位s"]), list(im["瞥号s"]), p, rngF)
        扫翻.append(dict(翻位比=float(p),
                         感知_均=_r(np.mean(似感)), 感知认出率=_r(np.mean(np.asarray(似感) >= 判据_检测)),
                         想象_均=_r(np.mean(似想)), 想象认出率=_r(np.mean(np.asarray(似想) >= 判据_检测))))
    门G = next((x for x in 扫 if x["想象_均"] < float(判据_检测)), None)
    门F = next((x for x in 扫翻 if x["想象_均"] < float(判据_检测)), None)
    P2 = dict(高斯扫描=扫, 翻位扫描=扫翻,
              高斯认不出阈=(门G["噪比"] if 门G else None),
              翻位认不出阈=(门F["翻位比"] if 门F else None),
              说明="高斯噪比 = ‖注入噪声‖/‖场码范数‖；翻位比 = 被翻转的码位比例。"
                   "两条曲线形状同 → 同一解码网络；想象起点更低 → 同样精度下想象更弱"
                   "（心盲症类比：精度不足则'没有画面'）")

    # ── P3 地址 ≠ 内容（位置是读取键，不是投影屏幕）──
    单 = 核.想象(0, 新键=核.新位置键("P3正"))
    正确 = float(核.读出(单["场"], 码库=屏["码s"][:1], 位置s=[单["键"]],
                         外观库=屏["外观s"][:1])["每格码似"][单["键"]])
    错 = []
    for i in range(5):
        k2 = 核.新位置键("P3错%d" % i)
        r2 = 核.读出(单["场"], 码库=屏["码s"][:1], 位置s=[k2], 外观库=屏["外观s"][:1])
        错.append(float(r2["每格码似"][k2]))
    # 空位（格内、该屏没东西）——整屏场码也解不出"那里有什么"
    空位 = [(bx, by) for by in range(核.格y) for bx in range(核.格x) if (bx, by) not in set(位s)]
    r3 = 核.读出(屏["场s"], 码库=屏["码s"], 位置s=空位[:40], 外观库=屏["外观s"],
                 判据=判据_检测, 瞥号_位=核.分瞥())
    空似 = [float(r3["每格码似"][k]) for k in 空位[:40]]
    P3 = dict(正确地址码似=_r(正确), 错误地址码似均=_r(np.mean(错)), 错误地址码似高=_r(np.max(错)),
              空位最高码似=_r(np.max(空似) if 空似 else None),
              说明="同一场码换个地址就解不出 ⇒ 位置只是读地址；空位解不出 ⇒ 没有'铺开的屏幕'")

    # ── P4 组合想象（离线建构，非回放）──
    P4 = dict(认得件数=im.get("认得件数"), 总件数=im.get("总件数"),
              码似最低=_r(im.get("码似最低")), 新组合格数=im.get("新组合格数"),
              位移=list(im.get("位移", ())), 全认得=bool(im.get("认得")),
              判据=float(判据_检测),
              说明="搬家后每件都得重新解出来：认得 ⇒ 离线重建成功；新组合>0 ⇒ 不是回放")

    # ── P5 源监控：同一物体、同一个码；来路不在内容里 ──
    实感场 = np.asarray(核.原子(单["原键"]) * 屏["码s"][0], dtype=np.float32)
    想象场 = np.asarray(单["场"], dtype=np.float32)
    同码 = bool(np.array_equal(屏["码s"][0], 屏["码s"][单["物体下标"]]))
    P5 = dict(同一物体码=同码,
              实感场范数=_r(np.linalg.norm(实感场)), 想象场范数=_r(np.linalg.norm(想象场)),
              两场码域cos=_r(float(实感场 @ 想象场) / (np.linalg.norm(实感场) * np.linalg.norm(想象场) + 1e-12)),
              位置原子cos=_r(float(核.原子(单["原键"]) @ 核.原子(单["键"])) / 1024.0),
              说明="两种场用的是同一个物体码，只有位置原子不同 ⇒ 内容不携带'来路'；"
                   "源监控只能来自 provenance（来路）或两区室 A/B 差分")

    # ── 确定性：同种子两趟逐位同 ──
    核B = 画面核(D=1024, 种子=种子_核)
    记B = 核B.记一屏(造屏(), 拍=1)
    同 = dict(件数同=bool(len(位s) == len(记B["位s"]) and list(位s) == [tuple(x) for x in 记B["位s"]]),
              场s逐位同=bool(all(np.array_equal(屏["场s"][g], 核B.库[-1]["场s"][g])
                                for g in 屏["场s"])),
              码s逐位同=bool(np.array_equal(屏["码s"], 核B.库[-1]["码s"])))
    报["臂"] = dict(P1_同一网络=dict(**P1), P2_精度扫描=P2, P3_地址非内容=P3,
                   P4_组合想象=P4, P5_源监控=P5, 确定性=同)

    # ── 产物：回放屏 / 想象屏（人眼核对"脑子里那块"）──
    产物 = os.path.join(os.path.dirname(HERE), "产物")
    os.makedirs(产物, exist_ok=True)
    报["产物"] = dict(
        回放屏=核.造png(路径=os.path.join(产物, "回放屏.png")),
        想象屏=核.造想象png(位移=(3, 2), 路径=os.path.join(产物, "想象屏.png")),
    )
    return 报


def main():
    报 = 主跑()
    out = os.path.join(HERE, "实验_想象三条理论预测结果.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(报, ensure_ascii=False, indent=1, default=str))
    print("\n[想象三条理论预测] 产物：%s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
