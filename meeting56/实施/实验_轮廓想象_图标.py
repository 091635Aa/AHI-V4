# -*- coding: utf-8 -*-
"""实验_轮廓想象_图标.py —— meeting56 三轮：轮廓型想象画面 + "有没有用"的判据。

【为什么重做】所有者（语音原话）：「人脑的想象画面是边框啊之类的一些虚线啊之类的……
哪怕是一个非常模糊的虚影，或者是边框，或者是一些碎片……至少来说能幻想到文件夹一些轮廓……
你这个彩色方块我没有看出有什么联想性……你这个验证方式可能有点小问题……至少来说对他有用的思考，对他有用才行。」

所以本轮不用 PSNR 当主判据（PSNR 只说明像素像不像，不说明想得对不对、有没有用）。
主判据 = **①认得出来 ②碎片下还认得 ③认得出还能用来行动 ④反例会失败**。

臂：
  C1 轮廓可读：想出来的轮廓画出来（人眼核对：边框/虚线/碎片，不是彩色方块）
  C2 想象→识别：生动度扫描（1.0/0.7/0.45/0.3/0.2/0.12），top-1 是否 = 记忆里那个图标
  C3 联想/泛化：换一套**没见过的实例**（不同尺寸/间距/位置）→ 还能认出是"文件夹/文件/…"
  C4 有用：想象给出 **身份 + 位置** → 生成"去那里"的计划；位置误差、目标是否对
  C5 反例（must-fail）：形状码打乱 → top-1 必须掉到机会水平（1/5）
  C6 空白屏不虚构：空屏 → 0 个图标
  C7 确定性：两趟画面/结果逐位同

跑法：python 实验_轮廓想象_图标.py （合成线画图标屏、固定种子、无网络）
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
M39 = os.path.join(AHI4, "meeting39", "实施")
for p in (HERE, M39):
    if p not in sys.path:
        sys.path.insert(0, p)

import 画面_轮廓想象 as LO                                   # noqa: E402

产物 = os.path.join(os.path.dirname(HERE), "产物")
名s = ("文件夹", "文件", "窗口", "箭头", "齿轮")
活度s = (1.0, 0.7, 0.45, 0.3, 0.2, 0.12)


def _放大(图, k: int = 2) -> np.ndarray:
    图 = np.clip(图, 0, 255).astype(np.uint8)
    return np.repeat(np.repeat(图, k, 0), k, 1)


def _存(路径, 图):
    LO.写png(_放大(图, 2), 路径)


def main():
    os.makedirs(产物, exist_ok=True)
    报 = {"设置": dict(图标=名s, 屏="合成线画图标屏 144×256（固定种子 2026）",
                       生动度=活度s, 轮廓="三阈值结构图的边界并集；补块边框不算轮廓")}

    # ── 记忆：看一遍原型屏（变体0）──
    帧0 = LO.造图标屏(变体=0)
    真值轮廓0 = np.full(帧0.shape, 28.0, np.float32)
    g = LO.灰度(帧0)
    真值轮廓0[g > float(np.median(g)) + 30.0] = 235.0
    _存(os.path.join(产物, "轮廓_真值屏.png"), 帧0)
    _存(os.path.join(产物, "轮廓_真值轮廓.png"), 真值轮廓0)
    核 = LO.轮廓想象()
    记 = 核.记一屏(帧0, 名s=名s)
    名序 = [条.get("名") for 条 in 核.库]
    报["记忆"] = dict(图标数=记["图标数"], 名序=名序,
                      框s=[条["框"] for 条 in 核.库], 位置s=[条["位置"] for 条 in 核.库],
                      轮廓像素=[int(条["轮廓"].sum()) for 条 in 核.库])

    # ── C1 轮廓可读（产物）──
    for 活 in (1.0, 活度s[2], 0.2):
        画 = LO.空画布()
        for i in range(len(核.库)):
            核.想象(i, 活, 画布=画)
        _存(os.path.join(产物, "轮廓_想象_%s.png" % 活), 画)
    # 单图标三联（补块 / 轮廓 / 碎片0.45）
    条0 = 核.库[0]
    稀疏0 = 核._稀疏(条0["轮廓"].ravel().astype(np.float32), 0.45).reshape(64, 64)
    for 名, 图 in (("补块", 条0["补"]), ("轮廓", 条0["轮廓"] * 235.0), ("碎片", 稀疏0 * 235.0)):
        LO.写png(np.repeat(np.repeat(np.clip(图, 0, 255).astype(np.uint8), 4, 0), 4, 1),
                 os.path.join(产物, "轮廓_文件夹_%s.png" % 名))
    C1 = dict(产物=["轮廓_真值屏.png", "轮廓_真值轮廓.png", "轮廓_想象_1.0.png",
                   "轮廓_想象_0.45.png", "轮廓_想象_0.2.png",
                   "轮廓_文件夹_补块.png", "轮廓_文件夹_轮廓.png", "轮廓_文件夹_碎片.png"],
              说明="人眼核对：框/虚线/碎片；单图标三联看'看到→轮廓→碎片'")

    # ── C2 想象→识别（生动度扫描）──
    C2 = {}
    for 活 in 活度s:
        对, 下 = 0, []
        for i, 条 in enumerate(核.库):
            稀疏 = 核._稀疏(条["轮廓"].ravel().astype(np.float32), 活).reshape(64, 64)
            r = 核.识别(稀疏)
            对 += int(r["名"] == 条["名"])
            下.append(dict(真=条["名"], 认=r["名"], 相似=r["相似"], 次佳=r["次佳"]))
        C2[str(活)] = dict(top1="%d/%d" % (对, len(核.库)), 准确率=round(对 / len(核.库), 3), 明细=下)

    # ── C3 联想/泛化：没见过的实例（变体1）──
    帧1 = LO.造图标屏(变体=1)
    _存(os.path.join(产物, "轮廓_真值屏_变体.png"), 帧1)
    核1 = LO.轮廓想象()
    记1 = 核1.记一屏(帧1, 名s=[])                    # 不依赖名：只取框/补块
    对3, 明3 = 0, []
    for j in range(len(核1.库)):
        补 = 核1.库[j]["补"]
        轮 = LO.轮廓图(补)
        稀疏 = 核1._稀疏(轮.ravel().astype(np.float32), 0.45).reshape(64, 64)
        r = 核.识别(稀疏)                          # 问**原型记忆**：这是什么
        真名 = 名s[j] if j < len(名s) else None
        对3 += int(r["名"] == 真名)
        明3.append(dict(实例=真名, 认=r["名"], 相似=r["相似"], 次佳=r["次佳"]))
    画 = LO.空画布()
    for j in range(len(核1.库)):
        轮 = LO.轮廓图(核1.库[j]["补"])
        稀疏 = 核1._稀疏(轮.ravel().astype(np.float32), 0.45).reshape(64, 64)
        r = 核.识别(稀疏)
        x0, y0, w, h = 核1.库[j]["框"]
        核1._贴(画, 稀疏, x0 - LO.框边距, y0 - LO.框边距, w + 2 * LO.框边距, h + 2 * LO.框边距)
    _存(os.path.join(产物, "轮廓_想象_变体.png"), 画)
    C3 = dict(实例数=len(核1.库), top1="%d/%d" % (对3, len(核1.库)),
              准确率=round(对3 / max(1, len(核1.库)), 3), 明细=明3,
              说明="记忆里只有变体0；变体1 的尺寸/间距/位置都没见过 ⇒ 认出才算'联想'不是'背像素'")

    # ── C4 有用：想象 → 身份+位置 → 计划 ──
    计划 = []
    for i, 条 in enumerate(核.库):
        稀疏 = 核._稀疏(条["轮廓"].ravel().astype(np.float32), 0.45).reshape(64, 64)
        r = 核.识别(稀疏)
        计划.append(dict(想=条["名"], 认出=r["名"], 目标位置=r["位置"],
                         记忆位置=条["位置"],
                         位置误差格=int(abs(r["位置"][0] - 条["位置"][0]) + abs(r["位置"][1] - 条["位置"][1])),
                         动作="去 %s 处点一下" % (r["位置"],)))
    想错 = sum(1 for x in 计划 if x["想"] != x["认出"])
    C4 = dict(计划=计划, 身份错数=想错,
              位置误差和=int(sum(x["位置误差格"] for x in 计划)),
              说明="想象给出'是什么+在哪'；位置来自记忆的空间参考帧（不是像素投影），可直接当行动目标")

    # ── C5 反例 must-fail：码打乱（多个种子平均，避免 n=5 的偶然）──
    对s = []
    for 反例种子 in range(2026, 2036):
        对 = 0
        for i, 条 in enumerate(核.库):
            稀疏 = 核._稀疏(条["轮廓"].ravel().astype(np.float32), 0.45).reshape(64, 64)
            r = 核.识别(稀疏, 反例种子=反例种子)
            对 += int(r["名"] == 条["名"])
        对s.append(对 / len(核.库))
    C5 = dict(打乱后top1_多种子=[round(x, 3) for x in 对s],
              准确率均=round(float(np.mean(对s)), 3),
              准确率最高=round(float(np.max(对s)), 3),
              机会水平=round(1 / len(核.库), 3),
              判定=("通过" if float(np.mean(对s)) <= 0.4 else "未过"),
              说明="同范数打乱坐标 ⇒ 形状信息被毁；若仍 >=0.8 说明识别靠了颜色/亮度等捷径")

    # ── C8 想象当先验：想象全屏 vs 真实轮廓的 IoU（预测校验用）──
    画8 = LO.空画布()
    for i in range(len(核.库)):
        核.想象(i, 1.0, 画布=画8)
    想轮 = (画8 > 60.0)
    真轮 = (真值轮廓0 > 60.0)

    def _胀(m, 次=2):
        m = np.asarray(m).astype(bool)
        for _ in range(次):
            r = np.pad(m, 1, constant_values=False)
            m = (m | r[2:, 1:-1] | r[:-2, 1:-1] | r[1:-1, 2:] | r[1:-1, :-2])
        return m

    交 = float((想轮 & 真轮).sum())
    并 = float((想轮 | 真轮).sum())
    交T = float((_胀(想轮) & _胀(真轮)).sum())
    并T = float((_胀(想轮) | _胀(真轮)).sum())
    C8 = dict(轮廓IoU=round(交 / max(并, 1.0), 3),
              容差2px_IoU=round(交T / max(并T, 1.0), 3),
              交=交, 并=并,
              说明="想象图可直接当'预期'与真实轮廓比；严格 IoU 低是因为 1px 虚线对细线极敏感，"
                   "容差 2px 的 IoU 才反映'轮廓对上了没有'。IoU 低 = '我记的和看到的不一样'（预测误差）")

    # ── C6 空白屏不虚构 ──
    空 = LO.空画布()
    核6 = LO.轮廓想象()
    记6 = 核6.记一屏(空 + np.random.default_rng(3).normal(0, 3, 空.shape).astype(np.float32))
    C6 = dict(空白屏图标数=记6["图标数"], 说明="没有东西就不该'想'出东西")

    # ── C7 确定性：两趟画面字节逐位同 ──
    画A = LO.空画布()
    画B = LO.空画布()
    for i in range(len(核.库)):
        核.想象(i, 0.45, 画布=画A)
    核B = LO.轮廓想象()
    核B.记一屏(帧0, 名s=名s)
    for i in range(len(核B.库)):
        核B.想象(i, 0.45, 画布=画B)
    C7 = dict(两趟画面逐位同=bool(np.array_equal(画A, 画B)),
              sha256=hashlib.sha256(np.clip(画A, 0, 255).astype(np.uint8).tobytes()).hexdigest()[:16])

    报["臂"] = dict(C1_轮廓可读=C1, C2_想象识别=C2, C3_泛化=C3, C4_有用=C4,
                   C5_反例=C5, C6_空白=C6, C7_确定性=C7, C8_先验校验=C8)
    out = os.path.join(HERE, "实验_轮廓想象_图标结果.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(报, ensure_ascii=False, indent=1, default=str))
    print("\n[轮廓想象] 产物目录：%s\n结果：%s" % (产物, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
