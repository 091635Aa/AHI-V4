# -*- coding: utf-8 -*-
"""实验_真图标_改进.py —— meeting56 五轮：**对齐补强 / 背景分离 / 跨 DPI 模拟**（接四轮 §13）。

四轮（§13）留下的三个最大缺口（§13.6）：
  ① 对齐敏感 —— 移 ±3px 就把 151 图标留一重认打到 .30–.53（最弱臂）；
  ② 背景干扰 —— `束(原始像素)` 的"原样"只有 .914（壁纸把不同图标拉成同码）；
  ③ 跨 DPI 未测 —— A/B/C 是同一桌面的裁剪，换 DPI 的行为没有数据。

本轮（零新算子、零训练、零 OCR、不改既有件）：
  R10 **对齐消融**：`对齐补块`（按字形质心平移到格带中心，确定性 warpAffine）× 表示 × 10 退化探针的留一重认；
       `前景对比图`（与局部背景色的 Lab 距离，替代原像素当"形状地形"）也一起消融（含负结果）。
  R11 **对齐后的识别门重标**：genuine/impostor → τ（对照四轮 τ 与既有 `识别门=0.75`）。
  R12 **跨 DPI 模拟**：整张 C 重采样到 0.75/1.25/1.5（栅距/相位同比缩放），测检出/重叠/重认；
       **这是模拟**（同一壁纸/布局），不是真换 DPI。
  R13 反例/确定性：码打乱、壁纸格、两趟逐位同。
  R14 产物：对齐示意（原样/移3/移3+对齐）、跨 DPI 检出框。

跑法：python 实验_真图标_改进.py   （无网络、固定种子；本机约 40 秒）
铁律：C1 禁 Transformer ｜ C9 禁 OCR ｜ C10 禁外挂模型当器官 ｜ C13 确定性 ｜ C14 不破既有。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
M56 = os.path.dirname(HERE)
产物 = os.path.join(M56, "产物", "真图标_改进")
素材 = os.path.join(M56, "素材", "真屏")
for p in (HERE, os.path.join(AHI4, "meeting38"), os.path.join(AHI4, "meeting38", "实施"),
          os.path.join(AHI4, "meeting39", "实施")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2                                                       # noqa: E402
import 画面_真图标 as ZI                                         # noqa: E402

表示s = ("边缘972", "前景972", "束原始", "束前景")


def _f(x, n=4):
    return None if x is None else round(float(x), n)


def _json(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def 码集(补块, 编码, 对齐: bool) -> dict:
    """→ dict(边缘972, 前景972, 束原始, 束前景)。对齐=False 走四轮原口径。"""
    if 对齐:
        a = ZI.对齐补块(补块)
        补块2, 前景 = a["补块"], a["前景"]
    else:
        补块2, 前景 = 补块, ZI.前景对比图(补块)
    return dict(边缘972=ZI.码_972(ZI.边缘色差图(补块2)),
                前景972=ZI.码_972(前景),
                束原始=ZI.码_形状束(补块2, 编码),
                束前景=ZI.码_形状束(前景, 编码))


def main():
    os.makedirs(产物, exist_ok=True)
    t0 = time.time()
    报 = {"设置": dict(
        素材="meeting56/素材/真屏/桌面_C_20261018.png（四轮同一素材）",
        口径="对齐=按字形质心平移到格带中心（warpAffine，确定性）；前景=与局部背景色(Lab)的距离图；"
             "跨 DPI=整屏重采样（模拟，不是真换 DPI）",
        表示=list(表示s),
        探针="原样/缩0.6/缩0.8/缩1.25/缩1.5/移-3/移+3/jpeg60/噪5/糊1（同四轮）",
        四轮基线="实例留一重认：边缘972=.7060（原样 1.0）、束原始=.6861（原样 .9139）；τ：边缘972=.90、束原始=.84")}

    img = ZI.读图(os.path.join(素材, "桌面_C_20261018.png"))
    H, W = img.shape[:2]
    det = ZI.检出(img, 排除底=48)
    ann = json.load(open(os.path.join(HERE, "标注_真图标.json"), encoding="utf-8"))
    assert [dict(行=g["行"], 列=g["列"]) for g in ann["格s"]] == \
           [dict(行=i["行"], 列=i["列"]) for i in det["图标s"]]
    类 = [g["类"] for g in ann["格s"]]
    idx = [i for i in range(len(类)) if 类[i] != "非图标"]
    非图标s = [i for i in range(len(类)) if 类[i] == "非图标"]

    import 共存态
    编码 = 共存态.共存态(种子=3800, 用gpu=False).编码
    补 = {i: ZI.补块(img, det["图标s"][i]) for i in idx}
    探s = list(ZI.退化探针(补[idx[0]]).keys())
    print("[setup] %d 图标 / %d 探针 / %.1fs" % (len(idx), len(探s), time.time() - t0), flush=True)

    # ── 码缓存：原型（对齐/不对齐）+ 全部退化探针（对齐/不对齐）──
    原型码 = {对齐: {i: 码集(补[i], 编码, 对齐) for i in idx} for 对齐 in (False, True)}
    探码 = {对齐: {(i, tn): 码集(ZI.退化探针(补[i])[tn], 编码, 对齐)
                 for i in idx for tn in 探s} for 对齐 in (False, True)}
    print("[codes] %.1fs" % (time.time() - t0), flush=True)

    原型堆 = {对齐: {rep: np.stack([原型码[对齐][i][rep] for i in idx]) for rep in 表示s}
            for 对齐 in (False, True)}
    序 = {i: k for k, i in enumerate(idx)}

    # ── R10 对齐消融 ──
    R10 = {"说明": "留一：每个退化探针在 151 原型里找最近邻；对齐=探针与原型都按字形质心对齐"}
    for 对齐, 名 in ((False, "原"), (True, "对齐")):
        结果 = {}
        for rep in 表示s:
            逐 = {}
            for tn in 探s:
                对 = sum(int(np.argmax(原型堆[对齐][rep] @ 探码[对齐][(i, tn)][rep]) == 序[i]) for i in idx)
                逐[tn] = _f(对 / len(idx))
            结果[rep] = dict(逐退化=逐, 均=_f(np.mean(list(逐.values()))),
                            均_剔除移3=_f(np.mean([v for k, v in 逐.items() if not k.startswith("移")])),
                            均_只算退化=_f(np.mean([v for k, v in 逐.items() if k != "原样"])))
        R10[名] = 结果
    报["R10_对齐消融"] = R10
    print("[R10] %.1fs" % (time.time() - t0), flush=True)

    # ── R11 对齐后的门重标 ──
    门 = {}
    for 对齐, 名 in ((False, "原"), (True, "对齐")):
        门[名] = {}
        for rep in ("边缘972", "束原始"):
            T, F = [], []
            for tn in 探s:
                for i in idx:
                    sims = 原型堆[对齐][rep] @ 探码[对齐][(i, tn)][rep]
                    T.append(float(sims[序[i]]))
                    假 = sims.copy()
                    假[序[i]] = -2.0
                    F.append(float(假.max()))
            标 = ZI.标定门(T, F, ZI.旧识别门)
            for 门值 in (ZI.旧识别门, 标["τ"]):
                接 = 正 = 0
                for tn in 探s:
                    for i in idx:
                        sims = 原型堆[对齐][rep] @ 探码[对齐][(i, tn)][rep]
                        k = int(np.argmax(sims))
                        if sims[k] >= 门值:
                            接 += 1
                            正 += int(idx[k] == i)
                标["门%.3f" % 门值] = dict(接受率=_f(接 / (len(idx) * len(探s))),
                                          接受中正确率=_f(正 / max(1, 接)))
            门[名][rep] = 标
    报["R11_门重标_对齐"] = dict(门=门,
                            说明="τ 由 Youden J 在 0.30..0.99 网格取；对照四轮 τ（边缘972=.90 / 束原始=.84）与既有 识别门=0.75")

    # ── R12 跨 DPI 模拟 ──
    格2i = {(it["行"], it["列"]): i for i, it in enumerate(det["图标s"])}
    基格 = set(格2i)
    跨 = {}
    for s in (0.75, 1.25, 1.5):
        W2, H2 = int(round(W * s)), int(round(H * s))
        im2 = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        det2 = ZI.检出(im2, 排除底=int(round(48 * s)),
                      列距_=int(round(ZI.列距 * s)), 行距_=int(round(ZI.行距 * s)))
        格2 = {(it["行"], it["列"]): it for it in det2["图标s"]}
        重叠 = set(格2) & 基格
        真对 = [(格2i[k], ZI.补块(im2, 格2[k])) for k in 重叠 if 类[格2i[k]] != "非图标"]
        准确 = {}
        for 对齐, 名 in ((False, "原"), (True, "对齐")):
            堆 = 原型堆[对齐]
            探码s = {i: 码集(p, 编码, 对齐) for i, p in 真对}
            for rep in ("边缘972", "束原始"):
                对 = sum(int(np.argmax(堆[rep] @ 探码s[i][rep]) == 序[i]) for i, _ in 真对)
                准确["%s_%s" % (rep, 名)] = _f(对 / max(1, len(真对)))
        跨[str(s)] = dict(尺寸=[W2, H2], 栅格=dict(列距=det2["栅格"]["列距"], 行距=det2["栅格"]["行距"],
                                                 列相=det2["栅格"]["列相"], 行相=det2["栅格"]["行相"]),
                         检出=len(det2["图标s"]), 与基准格重叠=len(重叠),
                         重叠中真图标=len(真对),
                         检出中真图标=len(真对),
                         精确率=_f(len(真对) / max(1, len(det2["图标s"]))),
                         基准格recall=_f(len(重叠) / max(1, len(基格))),
                         重认=准确)
        ZI.写图(ZI.外形标记(im2, det2["图标s"]), os.path.join(产物, "跨DPI_检出_%s.png" % s))
        print("[R12] s=%s 检出=%d 重叠=%d 真=%d" % (s, len(det2["图标s"]), len(重叠), len(真对)), flush=True)
    报["R12_跨DPI模拟"] = dict(
        逐尺度=跨, 基准=dict(检出=len(det["图标s"]), 真图标=len(idx), 非图标=len(非图标s)),
        说明="整屏重采样（0.75=INTER_AREA，1.25/1.5=INTER_LINEAR）+ 栅距/相位同比缩放；**模拟**，"
             "壁纸/布局/主题与基准相同，不等于真换 DPI。重叠=重采样后同一(行,列)格也被检出。")

    # ── R13 反例 / 确定性 ──
    堆 = 原型堆[True]
    打乱 = []
    for 种子 in range(20261018, 20261028):
        rr = np.random.default_rng(int(种子))
        对 = sum(int(np.argmax(堆["边缘972"] @ 原型码[True][i]["边缘972"][
            rr.permutation(len(原型码[True][i]["边缘972"]))]) == 序[i]) for i in idx)
        打乱.append(_f(对 / len(idx)))
    壁纸 = []
    for i in 非图标s:
        z = 码集(ZI.补块(img, det["图标s"][i]), 编码, True)
        sims = 堆["束原始"] @ z["束原始"]
        k = int(np.argmax(sims))
        壁纸.append(dict(格=[det["图标s"][i]["行"], det["图标s"][i]["列"]],
                        最高相似=_f(float(sims[k])), 最近类=类[idx[k]],
                        过旧门=bool(sims[k] >= 0.75),
                        过τ=bool(sims[k] >= 报["R11_门重标_对齐"]["门"]["对齐"]["束原始"]["τ"])))
    原型B = {i: 码集(ZI.补块(img, det["图标s"][i]), 编码, True) for i in idx}
    同 = all(np.array_equal(原型B[i][r], 原型码[True][i][r]) for i in idx for r in 表示s)
    报["R13_反例确定性"] = dict(
        打乱码=dict(逐种子=打乱, 均=_f(np.mean(打乱)), 机会=_f(1 / len(idx)),
                    判定=("通过" if float(np.mean(打乱)) <= 0.05 else "未过")),
        壁纸格=dict(格数=len(壁纸), 过旧门=int(sum(x["过旧门"] for x in 壁纸)),
                    过τ=int(sum(x["过τ"] for x in 壁纸)), 明细=壁纸),
        两趟原型码逐位同=bool(同),
        说明="打乱码必须掉回机会水平；壁纸格不应被认成图标（对齐后同口径）")

    # ── R14 产物：对齐示意（原样 / 移+3 / 移+3后对齐）──
    图s = []
    for i in idx[:6]:
        p = 补[i]
        移 = ZI.退化探针(p)["移3"]
        a = ZI.对齐补块(移)
        图s.append(np.concatenate([p, 移, a["补块"]], 1))
    ZI.写图(np.concatenate(图s, 0), os.path.join(产物, "对齐示意_移3.png"))
    报["R14_产物"] = ["跨DPI_检出_0.75.png", "跨DPI_检出_1.25.png", "跨DPI_检出_1.5.png", "对齐示意_移3.png"]

    # ── 落盘 ──
    out = os.path.join(HERE, "实验_真图标_改进结果.json")
    # 内容指纹**不含** 计时秒（计时每跑必变；四轮结果也把"内容"与"计时"分开）
    指纹源 = json.dumps({k: v for k, v in 报.items() if k != "计时秒"},
                        ensure_ascii=False, indent=1, default=_json)
    报["内容sha256"] = hashlib.sha256(指纹源.encode("utf-8")).hexdigest()
    报["计时秒"] = _f(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(报, ensure_ascii=False, indent=1, default=_json))
    print(json.dumps({k: 报[k] for k in ("R10_对齐消融", "R11_门重标_对齐", "R12_跨DPI模拟",
                                        "R13_反例确定性", "计时秒")},
                     ensure_ascii=True, indent=1, default=_json)[:6000])
    print("\n结果：%s\n产物：%s" % (out, 产物))
    return 0


if __name__ == "__main__":
    sys.exit(main())
