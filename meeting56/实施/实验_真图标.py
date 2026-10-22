# -*- coding: utf-8 -*-
"""实验_真图标.py —— meeting56 四轮：**真桌面图标**的检出 / 轮廓原型 / 接形状束·图标簿 / 识别门重标。

素材：`meeting56/素材/真屏/桌面_A|B|C_20261018.png`（所有者 2026-10-18 真桌面截图；文件哈希入库）。
口径：A、B、C 是同一张桌面的三个局部/全域截图（B 与 C 逐位相同；A 严格相同像素约 99.6–99.7%）——
      因此"跨屏重认"是**平凡**的，本实验据实当基线报；真正的泛化用**退化探针**（原样/缩放/平移/JPEG/噪声/模糊）。
      规模：151 个真图标 × 10 个退化探针 × 3 种表示（972 轮廓码 / 束(边缘) / 束(原始像素)）。

臂（沿用 甲/乙/丙/丁/戊 命名 + must-fail 纪律）：
  R0 编码          151 个真图标 → 边缘/轮廓 → 972 轮廓码 / 1024 既有形状束（边缘 与 原始像素两路）；同字形重复清点
  R1 检出          桌面栅格 76×100；检出格数、精确率（人工标注"非图标"）、跨裁剪一致率
  R2 实例重认      151 真图标 × 10 退化探针；留一最近邻；三种表示对照
  R3 类别识别      文件/文件夹/系统类（7 类）；留一 1-NN；混淆矩阵；三种表示对照
  R4 识别门重标    genuine/impostor → τ；与既有 `识别门=0.75` 同口径对比（实例级 + 类别级）
  R5 反例 must-fail 码打乱 10 种子 → 必须掉到机会水平；壁纸误检格 → 不应被认成图标
  R6 轮廓想象      真图标 → 轮廓原型 → 虚线/碎片（产物）；想象→识别（生动度扫描）；
                   想象=预期 vs 真实轮廓 → 预测误差（IoU/容差 IoU）
  R7 确定性        两趟逐位同 + 输入/结果 SHA256
  R8 想为用        类 → 动作 + 栅格位置 → 像素坐标 → "去那里双击"计划
  R9 接图标簿      真 器官_脑.脑器.簿 + 既有 形状束 + 既有 识别门（实例属性遮蔽，不改源码）

跑法：python 实验_真图标.py   （无网络、无外挂模型、无 OCR；固定种子；本机约 25 秒）
铁律：C1 禁 Transformer ｜ C9 禁 OCR ｜ C10 禁外挂模型当器官 ｜ C13 确定性 ｜ C14 不破既有。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
M56 = os.path.dirname(HERE)
产物根 = os.path.join(M56, "产物")
产物 = os.path.join(产物根, "真图标")
素材 = os.path.join(M56, "素材", "真屏")
for p in (HERE, os.path.join(AHI4, "meeting38"), os.path.join(AHI4, "meeting38", "实施"),
          os.path.join(AHI4, "meeting39", "实施")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2                                                       # noqa: E402
import 画面_真图标 as ZI                                         # noqa: E402

屏s = [("A", "桌面_A_20261018.png", 0), ("B", "桌面_B_20261018.png", 0),
       ("C", "桌面_C_20261018.png", 48)]                        # 排除底：整屏才有任务栏
活度s = (1.0, 0.7, 0.45, 0.3, 0.2, 0.12)
类臂 = ("文件夹", "压缩包", "白页", "Word", "Excel", "PDF", "系统")
动作表 = {"文件夹": "双击打开文件夹", "压缩包": "双击解压/打开", "白页": "双击打开文档",
          "Word": "双击打开文档", "Excel": "双击打开表格", "PDF": "双击打开 PDF",
          "演示": "双击放映/打开", "图片": "双击查看图片", "系统": "双击打开系统项",
          "应用": "双击运行程序", "其他": "双击打开", "非图标": "不动"}


def _f(x, n=4):
    return None if x is None else round(float(x), n)


def _json(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _码(补块, 编码):
    """补块 → 三种表示。

    - `码972`：972 维轮廓码（对**边缘图**做 3 阈值 × 16/8/2 池化）；
    - `束`  ：1024 维既有形状束，输入同为**边缘图** ⇒ 与 972 码是**同一特征的随机投影符号版**
      （复核实测 cos≈0.9997），不是独立表示；
    - `束原始`：1024 维既有形状束，输入是**原始 RGB 补块**（器官_脑 的原生用法）⇒ 与上面两者特征不同，才是真对照。
    """
    E = ZI.边缘色差图(补块)
    return dict(边缘=E, 轮廓=ZI.轮廓图(E, "边缘"),
                码972=ZI.码_972(E), 束=ZI.码_形状束(E, 编码),
                束原始=ZI.码_形状束(补块, 编码))


def _横排(图s, 路径, 放大=4):
    """若干等尺寸灰度/彩色小图 → 横排 PNG（人眼核对）。"""
    rows = []
    for a in 图s:
        a = np.asarray(a, np.float32)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        rows.append(np.clip(a, 0, 255).astype(np.uint8))
    h = max(r.shape[0] for r in rows)
    out = np.full((h, sum(r.shape[1] for r in rows) + 4 * (len(rows) - 1), 3), 16, np.uint8)
    x = 0
    for r in rows:
        out[:r.shape[0], x:x + r.shape[1]] = r
        x += r.shape[1] + 4
    out = np.repeat(np.repeat(out, 放大, 0), 放大, 1)
    ZI.写图(out, 路径)
    return 路径


def main():
    os.makedirs(产物, exist_ok=True)
    报 = {"设置": dict(
        素材="meeting56/素材/真屏/桌面_A|B|C_20261018.png（所有者 2026-10-18 真桌面截图，哈希见 输入）",
        口径="同一桌面的三个截图；真屏=演示通道（文件固定、哈希入库），判据不依赖实时屏幕",
        栅格="76×100（实测 ACF 峰）；字形带=格内纵向 [8,64)；标签带不进入任何编码（C9）",
        形状="边缘色差图（Lab 三通道 DoG 合成）→ 972 轮廓码 / 1024 既有形状束（1024 维双极）",
        退化=dict(类型=["原样", "缩0.6", "缩0.8", "缩1.25", "缩1.5", "移-3", "移3", "jpeg60", "噪5", "糊1"],
                  说明="真实桌面会遇到的退化（DPI 缩放/重采样/压缩/噪声/失焦）；不是全部真实扰动"),
        旧识别门=ZI.旧识别门,
        标注="实施/标注_真图标.json（人工标注：人看字形+可读标签；不来自系统 OCR/模型）",
        类臂=list(类臂))}

    # ───────────── 输入 / R1 检出 ─────────────
    输入 = {}
    检 = {}
    for tag, fn, 排除底 in 屏s:
        path = os.path.join(素材, fn)
        输入[tag] = dict(文件=fn, sha256=ZI.文件sha(path), 字节=os.path.getsize(path))
        img = ZI.读图(path)
        det = ZI.检出(img, 排除底=排除底)
        检[tag] = dict(尺寸=[img.shape[1], img.shape[0]], 栅格=det["栅格"],
                       图标s=det["图标s"], 图=img)
        ZI.写图(ZI.外形标记(img, det["图标s"]), os.path.join(产物, "检出_%s.png" % tag))
    素材关系 = {}
    try:
        base = 检["C"]["图"]
        for tag in ("B", "A"):
            small = 检[tag]["图"]
            res = cv2.matchTemplate(base, small, cv2.TM_SQDIFF)
            mn, _mx, loc, _ml = cv2.minMaxLoc(res)
            x0, y0 = int(loc[0]), int(loc[1])
            sub = base[y0:y0 + small.shape[0], x0:x0 + small.shape[1]]
            if sub.shape == small.shape:
                d = np.abs(sub.astype(np.int16) - small.astype(np.int16))
                素材关系[tag] = dict(对C偏移=[x0, y0],
                                    逐位全等=bool(int(d.max()) == 0),
                                    严格相同像素比=_f(float((d == 0).mean()), 4),
                                    平均差=_f(float(d.mean())),
                                    差大于8的像素比=_f(float((d > 8).mean()), 4),
                                    SQDIFF=float(mn))
    except Exception as e:
        素材关系["错误"] = "%s: %s" % (type(e).__name__, e)
    报["输入"] = dict(文件=输入, 关系=素材关系)

    ann = json.load(open(os.path.join(HERE, "标注_真图标.json"), encoding="utf-8"))
    assert [dict(行=g["行"], 列=g["列"]) for g in ann["格s"]] == \
           [dict(行=i["行"], 列=i["列"]) for i in 检["C"]["图标s"]], "标注与检出顺序不一致"
    类 = [g["类"] for g in ann["格s"]]
    索引s = [i for i in range(len(类)) if 类[i] != "非图标"]
    非图标s = [i for i in range(len(类)) if 类[i] == "非图标"]
    类计数 = {c: int(sum(1 for x in 类 if x == c)) for c in sorted(set(类))}

    # 人工标注页（人眼核对：格 crop + 类名）
    imgC = 检["C"]["图"]
    tiles = []
    for k, it in enumerate(检["C"]["图标s"]):
        x, y, w, h = it["格框"]
        patch = np.full((100, 76, 3), 15, np.uint8)
        xa, ya = max(0, x), max(0, y)
        xb, yb = min(imgC.shape[1], x + 76), min(imgC.shape[0], y + 100)
        patch[ya - y:yb - y, xa - x:xb - x] = imgC[ya:yb, xa:xb]
        tiles.append((k, it, patch, 类[k]))
    for pi in range(0, len(tiles), 40):
        chunk = tiles[pi:pi + 40]
        rm = (len(chunk) + 7) // 8
        M = np.full((rm * 224, 8 * 160, 3), 15, np.uint8)
        for j, (k, it, patch, cls) in enumerate(chunk):
            M[j // 8 * 224:j // 8 * 224 + 200, j % 8 * 160:j % 8 * 160 + 152] = \
                np.repeat(np.repeat(patch, 2, 0), 2, 1)
            cv2.putText(M, "%d" % k, (j % 8 * 160 + 2, j // 8 * 224 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(M, "r%dc%d %s" % (it["行"], it["列"], cls), (j % 8 * 160 + 44, j // 8 * 224 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 0), 1, cv2.LINE_AA)
        ZI.写图(M, os.path.join(产物, "标注页_%d.png" % (pi // 40)))

    # 跨裁剪一致率：B 与 C 同一像素（逐位相同）⇒ 检出差异只来自裁边/敏感性
    一致 = None
    try:
        base = 检["C"]["图"]
        small = 检["B"]["图"]
        res = cv2.matchTemplate(base, small, cv2.TM_SQDIFF)
        b0 = cv2.minMaxLoc(res)[2]                                # (x0,y0)
        C中心 = set()
        for it in 检["C"]["图标s"]:
            x, y, w, h = it["格框"]
            cx, cy = x + w // 2, y + h // 2
            if b0[0] - 20 <= cx <= b0[0] + small.shape[1] + 20 and \
               b0[1] - 20 <= cy <= b0[1] + small.shape[0] + 20:
                C中心.add((cx, cy))
        B中心 = set()
        for it in 检["B"]["图标s"]:
            x, y, w, h = it["格框"]
            B中心.add((x + w // 2 + b0[0], y + h // 2 + b0[1]))
        匹 = sum(1 for p in C中心 if any(abs(p[0] - q[0]) <= 38 and abs(p[1] - q[1]) <= 50 for q in B中心))
        一致 = dict(C区域内格=len(C中心), B检出=len(B中心), 可对上=匹,
                    一致率=_f(匹 / max(1, len(C中心)), 3))
    except Exception as e:
        一致 = dict(错误="%s: %s" % (type(e).__name__, e))

    报["R1_检出"] = dict(
        各屏={tag: dict(尺寸=检[tag]["尺寸"], 图标数=len(检[tag]["图标s"]), 栅格=检[tag]["栅格"]) for tag in 检},
        C人工标注=dict(图标=len(索引s), 非图标=len(非图标s),
                       精确率=_f(len(索引s) / max(1, len(索引s) + len(非图标s)), 3)),
        跨裁剪一致率=一致,
        产物=["检出_A.png", "检出_B.png", "检出_C.png"] + ["标注页_%d.png" % i for i in range(5)],
        说明="检出=桌面栅格 + 字形带密度/紧凑度启发（非误差场涌现）；非图标=人工标注的壁纸误检格。"
             "未被检出的图标无从标注 ⇒ 本臂给精确率与跨裁剪一致率，不给完整 recall（诚实边界）")

    # ───────────── R0 编码 ─────────────
    import 共存态
    态 = 共存态.共存态(种子=3800, 用gpu=False)
    编码 = 态.编码
    补 = {}
    码 = {}
    for i in 索引s:
        补[i] = ZI.补块(imgC, 检["C"]["图标s"][i])
        码[i] = _码(补[i], 编码)
    相似M = np.zeros((len(索引s), len(索引s)), np.float32)
    for a in range(len(索引s)):
        for b in range(a + 1, len(索引s)):
            相似M[a, b] = 相似M[b, a] = float(码[索引s[a]]["束"] @ 码[索引s[b]]["束"])
    同形 = {}
    for a, i in enumerate(索引s):
        双 = [(索引s[b], 类[索引s[b]], _f(相似M[a, b], 3)) for b in range(len(索引s))
              if b != a and 相似M[a, b] >= 0.999]
        if 双:
            同形[i] = 双
    报["R0_编码"] = dict(图标数=len(索引s), 非图标数=len(非图标s), 类计数=类计数,
                         表示=dict(轮廓码972="972 维实数单位向量（三轮同律：3 阈值 × 16/8/2 池化）",
                                  形状束1024="既有 形状束.py：1024 维双极码；识别门 0.75 的原生表示"),
                         同字形重复=dict(格数=len(同形),
                                       明细={"%s#%d" % (类[i], i): 同形[i] for i in sorted(同形)}),
                         说明="同字形重复=同一图标资源被多个文件复用（真桌面事实）；形状路天生分不开，不算算法错")

    # ───────────── 退化探针 ─────────────
    探 = {}
    for i in 索引s:
        for 名, p in ZI.退化探针(补[i]).items():
            探[(i, 名)] = _码(p, 编码)
    探名s = sorted(set(k[1] for k in 探))

    # ───────────── R2 实例重认（留一：探针 → 149 原型最近邻）─────────────
    def _实例臂(rep):
        原型 = np.stack([码[i][rep] for i in 索引s])
        序 = {i: k for k, i in enumerate(索引s)}
        逐 = {}
        全真, 全假 = [], []
        for tn in 探名s:
            对 = 0
            for i in 索引s:
                v = 探[(i, tn)][rep]
                sims = 原型 @ v
                k = int(np.argmax(sims))
                对 += int(索引s[k] == i)
                全真.append(float(sims[序[i]]))
                假 = sims.copy()
                假[序[i]] = -2.0
                全假.append(float(假.max()))
            逐[tn] = _f(对 / len(索引s), 4)
        non_shift = [v for k, v in 逐.items() if not k.startswith("移")]
        return dict(逐退化=逐, 平均top1=_f(np.mean(list(逐.values())), 4),
                    平均_剔除移3=_f(np.mean(non_shift), 4),
                    平均_只算退化=_f(np.mean([v for k, v in 逐.items() if k != "原样"]), 4),
                    最差=min(逐.items(), key=lambda t: t[1]),
                    真s=[_f(x) for x in 全真], 假s=[_f(x) for x in 全假],
                    真s均=_f(np.mean(全真)), 假s均=_f(np.mean(全假)))

    R2 = dict(轮廓码972=_实例臂("码972"), 形状束1024_边缘=_实例臂("束"),
          形状束1024_原始像素=_实例臂("束原始"))
    R2["说明"] = ("留一：每个退化探针在 %d 个原型里找最近邻（原型含自己，但探针是退化版）。" % len(索引s) +
                  "同字形重复格天然不可分，会拉低头行（见 R0）")
    报["R2_实例重认"] = R2

    # ───────────── R3 类别识别（7 类；留一 1-NN）─────────────
    类臂索引 = [i for i in 索引s if 类[i] in 类臂]

    def _类别臂(rep):
        原型 = np.stack([码[i][rep] for i in 类臂索引])
        位置 = {i: k for k, i in enumerate(类臂索引)}
        逐, 混淆 = {}, {}
        for tn in 探名s:
            对 = 0
            for i in 类臂索引:
                v = 探[(i, tn)][rep]
                sims = 原型 @ v
                sims[位置[i]] = -2.0
                k = int(np.argmax(sims))
                真, 预 = 类[i], 类[类臂索引[k]]
                key = "%s→%s" % (真, 预)
                混淆[key] = 混淆.get(key, 0) + 1
                对 += int(预 == 真)
            逐[tn] = _f(对 / len(类臂索引), 4)
        逐类 = {}
        for c in 类臂:
            子 = [i for i in 类臂索引 if 类[i] == c]
            if len(子) < 2:
                逐类[c] = dict(实例=len(子), 说明="单实例：留一无法测（排除）")
                continue
            对 = 0
            for i in 子:
                sims = 原型 @ 原型[位置[i]]
                sims[位置[i]] = -2.0
                k = int(np.argmax(sims))
                对 += int(类[类臂索引[k]] == c)
            逐类[c] = dict(实例=len(子), 留一自识别=_f(对 / len(子), 3))
        return dict(逐退化=逐, 平均=_f(np.mean(list(逐.values())), 4),
                    混淆=dict(sorted(混淆.items(), key=lambda t: -t[1])), 逐类=逐类)

    R3 = dict(轮廓码972=_类别臂("码972"), 形状束1024_边缘=_类别臂("束"),
          形状束1024_原始像素=_类别臂("束原始"))
    R3["说明"] = "留一 1-NN：排除自己，看最近邻的类；类臂=文件夹/压缩包/白页/Word/Excel/PDF/系统"
    报["R3_类别识别"] = R3

    # ───────────── R4 识别门重标 ─────────────
    def _门(rep):
        原型 = np.stack([码[i][rep] for i in 索引s])
        序 = {i: k for k, i in enumerate(索引s)}
        T, F = [], []
        for tn in 探名s:
            for i in 索引s:
                sims = 原型 @ 探[(i, tn)][rep]
                T.append(float(sims[序[i]]))
                假 = sims.copy()
                假[序[i]] = -2.0
                F.append(float(假.max()))
        标 = ZI.标定门(T, F, ZI.旧识别门)
        for 门 in (ZI.旧识别门, 标["τ"]):
            接 = 正 = 0
            for tn in 探名s:
                for i in 索引s:
                    sims = 原型 @ 探[(i, tn)][rep]
                    k = int(np.argmax(sims))
                    if sims[k] >= 门:
                        接 += 1
                        正 += int(索引s[k] == i)
            标["门%.3f" % 门] = dict(接受率=_f(接 / (len(探名s) * len(索引s)), 4),
                                     接受中正确率=_f(正 / max(1, 接), 4))
        return 标

    def _类门(rep):
        原型 = np.stack([码[i][rep] for i in 类臂索引])
        位置 = {i: k for k, i in enumerate(类臂索引)}
        T, F = [], []
        for tn in 探名s:
            for i in 类臂索引:
                sims = 原型 @ 探[(i, tn)][rep]
                同, 异 = [], []
                for j in 类臂索引:
                    if j == i:
                        continue
                    (同 if 类[j] == 类[i] else 异).append(float(sims[位置[j]]))
                if 同:
                    T.append(max(同))
                if 异:
                    F.append(max(异))
        return ZI.标定门(T, F, ZI.旧识别门) if T and F else dict(说明="样本不足")

    报["R4_识别门重标"] = dict(
        实例级=dict(轮廓码972=_门("码972"), 形状束1024_边缘=_门("束"),
                   形状束1024_原始像素=_门("束原始")),
        类别级=dict(轮廓码972=_类门("码972"), 形状束1024_边缘=_类门("束"),
                   形状束1024_原始像素=_类门("束原始")),
        说明="τ 由 Youden J=TPR−FPR 在 0.30..0.99 网格（步长 0.005）上取；同分取较大 τ（保守）。"
             "旧门=0.75 是 器官_脑.py:409 的既有值（沙盒图标标定），本臂用真图标重标")

    # ───────────── R5 反例 must-fail ─────────────
    乱 = []
    for 种子 in range(20261018, 20261028):
        rr = np.random.default_rng(int(种子))
        for rep in ("码972", "束"):
            原型 = np.stack([码[i][rep] for i in 索引s])
            对 = 0
            for i in 索引s:
                v = 码[i][rep][rr.permutation(len(码[i][rep]))]        # 同范数、毁形状
                sims = 原型 @ v
                k = int(np.argmax(sims))
                对 += int(索引s[k] == i)
            乱.append(dict(种子=int(种子), 表示=rep, top1=_f(对 / len(索引s), 4)))
    原型束 = np.stack([码[i]["束"] for i in 索引s])
    壁纸测 = []
    τ新 = 报["R4_识别门重标"]["实例级"]["形状束1024_边缘"]["τ"]
    for i in 非图标s:
        z = _码(ZI.补块(imgC, 检["C"]["图标s"][i]), 编码)
        sims = 原型束 @ z["束"]
        k = int(np.argmax(sims))
        壁纸测.append(dict(格=(检["C"]["图标s"][i]["行"], 检["C"]["图标s"][i]["列"]),
                          最高相似=_f(float(sims[k]), 3), 最近类=类[索引s[k]],
                          过旧门=bool(sims[k] >= ZI.旧识别门), 过新门=bool(sims[k] >= τ新)))
    报["R5_反例"] = dict(
        打乱码=dict(逐次=乱, 均top1=_f(np.mean([x["top1"] for x in 乱]), 4),
                    机会水平=_f(1.0 / len(索引s), 4),
                    判定=("通过" if float(np.mean([x["top1"] for x in 乱])) <= 0.05 else "未过")),
        壁纸误检格=dict(格数=len(壁纸测), 明细=壁纸测,
                        过旧门数=int(sum(x["过旧门"] for x in 壁纸测)),
                        过新门数=int(sum(x["过新门"] for x in 壁纸测))),
        说明="打乱码必须掉回机会水平（形状信息被毁）；壁纸格不应被认成图标（不虚构）")

    # ───────────── R6 轮廓想象 ─────────────
    类原型 = {}
    for c in sorted(set(类[i] for i in 索引s)):
        子 = [i for i in 索引s if 类[i] == c]
        best, bi = -2.0, 子[0]
        for i in 子:
            s = np.mean([float(码[i]["束"] @ 码[j]["束"]) for j in 子 if j != i]) if len(子) > 1 else 1.0
            if s > best:
                best, bi = float(s), i
        类原型[c] = bi
    书 = ZI.图标簿(门=ZI.旧识别门)
    原型轮廓 = {}
    for c, i in 类原型.items():
        原型轮廓[c] = 码[i]["轮廓"]
        书.记(名=c, 码=ZI.码_972(码[i]["轮廓"], 已轮廓=True), 类=c,
              位置=(检["C"]["图标s"][i]["行"], 检["C"]["图标s"][i]["列"]),
              束=码[i]["束"], 成员s=[索引s.index(x) for x in 索引s if 类[x] == c])
    for c, i in 类原型.items():
        轮 = 原型轮廓[c].astype(np.float32) * 255.0
        图s = [补[i], 轮] + [np.clip(ZI.想象(原型轮廓[c], a), 0, 255) for a in (1.0, 0.45, 0.2)]
        _横排(图s, os.path.join(产物, "想象_%s.png" % c), 放大=4)
    C2 = {}
    for 活 in 活度s:
        对, 明 = 0, []
        for c, i in 类原型.items():
            稀 = ZI.想象(原型轮廓[c], 活)
            r = 书.识(ZI.码_972(稀, 已轮廓=True), 门=1.1)      # 门=1.1 ⇒ 只取最近邻，不拒绝
            对 += int(r["名"] == c)
            明.append(dict(真=c, 认=r["名"], 相似=r["相似"], 次佳=r["次佳"]))
        C2[str(活)] = dict(top1="%d/%d" % (对, len(类原型)), 准确率=_f(对 / len(类原型), 3), 明细=明)
    IoUs = {c: [] for c in 类原型}
    for i in 索引s:
        if 类[i] not in 类原型:
            continue
        IoUs[类[i]].append(ZI.预测误差IoU(原型轮廓[类[i]], 码[i]["轮廓"], 容差=2))
    全IoU = [x for v in IoUs.values() for x in v]
    画 = np.full(imgC.shape[:2], 20, np.float32)
    碎 = np.full(imgC.shape[:2], 20, np.float32)
    for k, it in enumerate(检["C"]["图标s"]):
        c = 类[k]
        if c not in 类原型:
            continue
        x, y, w, h = it["格框"]
        xa, ya = max(0, x), max(0, y + 8)
        if xa + 76 > imgC.shape[1] or ya + 56 > imgC.shape[0]:
            continue
        区 = 画[ya:ya + 56, xa:xa + 76]
        轮贴 = cv2.resize(原型轮廓[c].astype(np.float32), (76, 56), interpolation=cv2.INTER_NEAREST)
        画[ya:ya + 56, xa:xa + 76] = np.maximum(区, 轮贴 * 235.0)
        区2 = 碎[ya:ya + 56, xa:xa + 76]
        碎贴 = cv2.resize(ZI.想象(原型轮廓[c], 0.45), (76, 56), interpolation=cv2.INTER_NEAREST)
        碎[ya:ya + 56, xa:xa + 76] = np.maximum(区2, 碎贴 * 235.0)
    ZI.写图(画, os.path.join(产物, "想象_全屏_轮廓.png"))
    ZI.写图(碎, os.path.join(产物, "想象_全屏_碎片0.45.png"))
    想象实例 = []
    for c, i in 类原型.items():
        码稀 = ZI.码_972(ZI.想象(原型轮廓[c], 0.45), 已轮廓=True)
        拟 = [(float(码稀 @ ZI.码_972(码[j]["轮廓"], 已轮廓=True)), j) for j in 索引s if j != i]
        拟.sort(key=lambda t: -t[0])
        预 = 类[拟[0][1]]
        想象实例.append(dict(原型=c, 最近实例类=预, 对=bool(预 == c), 相似=_f(拟[0][0]),
                            最近格=[int(检["C"]["图标s"][拟[0][1]]["行"]), int(检["C"]["图标s"][拟[0][1]]["列"])]))
    报["R6_轮廓想象"] = dict(
        类原型={"%s#%d" % (c, i): dict(类=c, 格=(检["C"]["图标s"][i]["行"], 检["C"]["图标s"][i]["列"]))
                for c, i in 类原型.items()},
        想象识别=C2,
        想象识别_实例留一=dict(
            说明="类原型想象（0.45 生动度）→ 在**其余 %d 个实例**轮廓码里最近邻（排除原型自己），看是否落回同类。" % (len(索引s) - 1) +
                 "这是对'想象=闭集自洽'的加测（复核指出 11/11 是闭集：想象图取自类原型自己）。",
            明细=想象实例,
            top1="%d/%d" % (sum(1 for x in 想象实例 if x["对"]), len(想象实例))),
        预测误差=dict(
            整体=dict(严格IoU均=_f(np.mean([x["IoU"] for x in 全IoU])),
                      容差2px_IoU均=_f(np.mean([x["容差IoU"] for x in 全IoU])),
                      严格IoU范围=[_f(np.min([x["IoU"] for x in 全IoU])),
                               _f(np.max([x["IoU"] for x in 全IoU]))],
                      容差IoU范围=[_f(np.min([x["容差IoU"] for x in 全IoU])),
                               _f(np.max([x["容差IoU"] for x in 全IoU]))],
                      样本=len(全IoU)),
            逐类={c: dict(样本=len(v), 严格IoU均=_f(np.mean([x["IoU"] for x in v])),
                        容差IoU均=_f(np.mean([x["容差IoU"] for x in v])))
                  for c, v in IoUs.items() if v}),
        产物=["想象_%s.png" % c for c in 类原型] + ["想象_全屏_轮廓.png", "想象_全屏_碎片0.45.png"],
        说明="想象=类原型轮廓按生动度稀疏化（虚线/碎片）；预测误差=原型轮廓 vs 该实例真实轮廓的 IoU；"
             "严格 IoU 低是 1px 线对细线极敏感（与三轮 C8 同一病灶），容差 2px 才反映'大体对上'")

    # ───────────── R7 确定性 ─────────────
    imgC2 = ZI.读图(os.path.join(素材, "桌面_C_20261018.png"))
    det2 = ZI.检出(imgC2, 排除底=48)
    同 = len(det2["图标s"]) == len(检["C"]["图标s"])
    if 同:
        同 = all(np.array_equal(det2["图标s"][i]["字框"], 检["C"]["图标s"][i]["字框"])
                for i in range(len(det2["图标s"])))
    码2 = {i: _码(ZI.补块(imgC2, det2["图标s"][i]), 编码) for i in 索引s}
    同 = 同 and all(np.array_equal(码2[i]["束"], 码[i]["束"])
                   and np.array_equal(码2[i]["束原始"], 码[i]["束原始"])
                   and np.array_equal(码2[i]["码972"], 码[i]["码972"]) for i in 索引s)
    报["R7_确定性"] = dict(两趟检出与码逐位同=bool(同),
                          输入sha256={k: v["sha256"] for k, v in 输入.items()})

    # ───────────── R8 想为用 ─────────────
    计划 = []
    for k, it in enumerate(检["C"]["图标s"]):
        c = 类[k]
        if c == "非图标":
            continue
        x, y, w, h = it["格框"]
        计划.append(dict(类=c, 格=(int(it["行"]), int(it["列"])),
                        像素=(int(x + 38), int(y + 42)), 动作=动作表.get(c, "看")))
    with open(os.path.join(产物, "行动计划_真图标.json"), "w", encoding="utf-8") as f:
        json.dump(计划, f, ensure_ascii=False, indent=1)
    报["R8_想为用"] = dict(计划数=len(计划), 示例=计划[:8],
                        计划文件="产物/真图标/行动计划_真图标.json",
                        说明="类 → 动作 + 栅格位置 → 像素坐标（位置由桌面栅格键给出，不是像素投影）")

    # ───────────── R9 接图标簿（真 脑器.簿）─────────────
    try:
        from 器官_脑 import 脑器
        脑 = 脑器(态, 允许双=True, 允许回报=True)
        簿书 = ZI.图标簿(门=ZI.旧识别门)
        for k, i in enumerate(索引s):
            簿书.记(名="%s#%d" % (类[i], i), 码=码[i]["束"], 类=类[i],
                    位置=(检["C"]["图标s"][i]["行"], 检["C"]["图标s"][i]["列"]),
                    束=码[i]["束"], 成员s=[k])
        灌 = 簿书.接图标簿(脑)
        测 = {}
        for 门 in (ZI.旧识别门, τ新):
            脑.识别门 = float(门)
            对 = 接 = 0
            for tn in 探名s:
                for i in 索引s:
                    条 = 脑._簿识束(探[(i, tn)]["束"])
                    if 条 is None:
                        continue
                    接 += 1
                    对 += int(条["名称"] == "%s#%d" % (类[i], i))
            测["门%.3f" % 门] = dict(识别率=_f(对 / (len(探名s) * len(索引s)), 4),
                                     覆盖率=_f(接 / (len(探名s) * len(索引s)), 4),
                                     接受中正确率=_f(对 / max(1, 接), 4))
        报["R9_接图标簿"] = dict(成功=True, 簿条目=灌, 形状描可用=bool(脑.形状描 is not None),
                             门=测, 说明="真的构造了 器官_脑.脑器，把形状束灌进 脑.簿，用既有 _簿识束"
                                         "（门=脑.识别门）做识别；只改实例属性，源码未动（C14）")
    except Exception as e:
        报["R9_接图标簿"] = dict(成功=False, 原因="%s: %s" % (type(e).__name__, e))

    # ───────────── 落盘 ─────────────
    out = os.path.join(HERE, "实验_真图标结果.json")
    报["R7_确定性"]["结果内容sha256"] = hashlib.sha256(
        json.dumps(报, ensure_ascii=False, indent=1, default=_json).encode("utf-8")).hexdigest()
    txt = json.dumps(报, ensure_ascii=False, indent=1, default=_json)
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(json.dumps({k: 报[k] for k in ("R1_检出", "R2_实例重认", "R4_识别门重标", "R5_反例")},
                     ensure_ascii=True, indent=1, default=_json)[:4000])
    print("\n结果：%s\n产物：%s" % (out, 产物))
    return 0


if __name__ == "__main__":
    sys.exit(main())
