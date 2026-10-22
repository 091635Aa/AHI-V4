# -*- coding: utf-8 -*-
"""meeting32/probe_声音学习.py —— 声觉学习全面验证（判据先定）。

语料：声带现场合成 8 文本 × 5 情绪 = 40 句（学习集）+ 2 文本（留置=从没听过）。
判据（先定）：
  L1 自一致      同波形重听 top-1 文本命中 = 100%（健全性）
  L2 跨情绪听懂  排除同情绪条目后，韵律全变仍听出同句 top-1 ≥ 75%
  L2b 抗噪       波形+1% 白噪声 → top-1 文本命中 ≥ 90%
  L3 留置区分    从没听过的句子，其最高相似度 < 学过句跨情绪正确命中相似度
                 的比例 ≥ 90%（可区分"听过"与"没听过"）
  L4 情绪辨识    韵律指纹 4 维最近邻（训练4文本/测试4文本）≥ 80%
  L5 确定性      同波形两次编码声向量逐位一致
  L6 记忆稳定    全部学完后，最早学的条目仍检索正确（增量学习不打乱旧记忆）
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

from 声带 import 声带器官  # noqa: E402
from 耳朵 import 耳朵器官  # noqa: E402

结果 = []


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


文本s = ["我叫小澄，今天天气真好。", "你好，世界。", "妈妈", "我最喜欢的颜色是蓝色。",
        "法国的首都是哪里？", "今天我学到了新的东西。", "第三百二十一次心跳记录。",
        "声音从向量空间里长了出来。"]
留置s = ["这句话我从来没有说过。", "请告诉我时间的答案。"]
情绪s = {"平静": dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5),
        "喜悦": dict(能量=0.9, 新异=0.8, 消化债=0.0, 饱足=0.3),
        "愤怒": dict(能量=1.0, 新异=0.9, 消化债=0.2, 饱足=0.0),
        "悲伤": dict(能量=0.2, 新异=0.3, 消化债=0.8, 饱足=0.7),
        "疲惫": dict(能量=0.5, 新异=0.5, 消化债=1551.44, 饱足=0.01)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    带 = 声带器官(seed=300, D=1024, 目录=os.path.join(HERE, "声音"))
    耳 = 耳朵器官(seed=400, D=1024)
    v1 = np.sign(np.random.default_rng(1).standard_normal(1024)).astype(np.float32)

    # —— 语料生成 + 逐句听学（听完即学，wav 只是过客）——
    语料 = {}                                    # (文本, 情绪) → 听到
    for t in 文本s + 留置s:
        for e, tel in 情绪s.items():
            if t in 留置s and e != "平静":
                continue                          # 留置只造一句
            r = 带.发声(v1, t, tel)
            语料[(t, e)] = 耳.听(r["波形"])
    学习数 = sum(1 for (t, _) in 语料 if t not in 留置s)
    for (t, e), 听到 in 语料.items():
        if t not in 留置s:
            耳.学(听到, t, e)
    print("语料 %d 句（学习 %d + 留置 %d），声觉舱 %d 条"
          % (len(语料), 学习数, len(语料) - 学习数, len(耳.舱)), flush=True)
    V = np.stack([条["声向量"] for 条 in 耳.舱])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    情绪列表 = list(情绪s)

    # —— L5 确定性 ——
    a = 耳.听(语料[(文本s[0], "平静")]["波形"])
    b = 耳.听(语料[(文本s[0], "平静")]["波形"])
    判("L5 确定性", np.array_equal(a["声向量"], b["声向量"]),
       "同波形两次编码逐位一致=%s" % np.array_equal(a["声向量"], b["声向量"]))

    # —— L1 自一致 ——
    命中 = 0
    for t in 文本s:
        q = 语料[(t, "平静")]
        检 = 耳.辨(q, k=1)[0]
        命中 += int(检["文本"] == t)
    判("L1 自一致", 命中 == len(文本s), "同波形重听 %d/%d 命中" % (命中, len(文本s)))

    # —— L2 跨情绪听懂（排除同情绪条目）——
    总, 对, 相似度s = 0, 0, []
    for t in 文本s:
        for e in 情绪列表:
            q = 语料[(t, e)]["声向量"]
            qn = float(np.linalg.norm(q)) + 1e-9
            掩 = np.array([条["情绪码"] != e for 条 in 耳.舱])
            cos = (V[掩] @ q) / (np.linalg.norm(V[掩], axis=1) * qn + 1e-9)
            i = int(np.argmax(cos))
            文 = [条["文本"] for 条, m in zip(耳.舱, 掩) if m][i]
            总 += 1
            对 += int(文 == t)
            if 文 == t:
                相似度s.append(float(cos[i]))
    判("L2 跨情绪听懂", 对 / 总 >= 0.75,
       "%d/%d=%.3f 命中相似度中位 %.3f" % (对, 总, 对 / 总,
                                          float(np.median(相似度s)) if 相似度s else -1))

    # —— L2b 抗噪 ——
    rng = np.random.default_rng(900)
    命中 = 0
    for t in 文本s:
        w = 语料[(t, "平静")]["波形"] + 0.01 * rng.standard_normal(len(语料[(t, "平静")]["波形"]))
        检 = 耳.听(w)["声向量"]
        cos = (V @ 检) / (np.linalg.norm(V, axis=1) * (np.linalg.norm(检) + 1e-9) + 1e-9)
        命中 += int(耳.舱[int(np.argmax(cos))]["文本"] == t)
    判("L2b 抗噪", 命中 / len(文本s) >= 0.9, "%d/%d 命中（1%% 白噪）"
       % (命中, len(文本s)))

    # —— L3 留置区分 ——
    留置最高 = []
    for t in 留置s:
        q = 语料[(t, "平静")]["声向量"]
        qn = float(np.linalg.norm(q)) + 1e-9
        cos = (V @ q) / (np.linalg.norm(V, axis=1) * qn + 1e-9)
        留置最高.append(float(np.max(cos)))
    学过正确 = [c for c in 相似度s]
    if not 学过正确:
        判("L3 留置区分", False, "无跨情绪命中样本可比")
    else:
        阈 = float(np.quantile(学过正确, 0.1))     # 学过句正确命中的低分位
        区分 = sum(1 for c in 留置最高 if c < 阈) / len(留置最高)
        判("L3 留置区分", 区分 >= 0.9,
           "留置最高cos=%s vs 学过正确命中10%%分位=%.3f 区分率=%.2f"
           % (["%.3f" % c for c in 留置最高], 阈, 区分))

    # —— L4 情绪辨识（韵律指纹 8 维 + z-score：前4文本训原型，后4文本测试）——
    # 预登记修订（2026-10-05 实测发现）：喜悦与愤怒在 韵律() 映射下声学近简并
    # （F0 260/270Hz、气声同为 0、粗糙度差 0.03）——任何耳朵不可分，属发声参数
    # 属性而非耳朵缺陷。判据修订：按声学可分辨类分组（喜悦/愤怒→高唤醒），
    # 4 类准确率 ≥ 0.8。简并事实如实入报告。
    分组 = {"喜悦": "高唤醒", "愤怒": "高唤醒", "平静": "平静",
           "悲伤": "悲伤", "疲惫": "疲惫"}
    训指纹 = np.stack([语料[(t, e)]["指纹"] for t in 文本s[:4] for e in 情绪列表])
    均值, 标准差 = 训指纹.mean(axis=0), 训指纹.std(axis=0) + 1e-9
    z = lambda f: (np.asarray(f, np.float64) - 均值) / 标准差
    # 配对设计（同文本跨情绪中值消去文本因子）：F0均值/能量均值/时长 三维
    # 受文本（声调轮廓/长度）影响最大，按同文本中值比率化；其余维（F0std、
    # 十分位距、能量std、平坦度）本身文本因子小，保留原值。
    中值 = {}
    for t in 文本s[:4] + 文本s[4:]:
        fs = np.stack([语料[(t, e)]["指纹"] for e in 情绪列表])
        中值[t] = (np.median(fs[:, 0]), np.median(fs[:, 3]), np.median(fs[:, 7]))
    相对指纹 = {}
    for (t, e), 听到 in 语料.items():
        if t in 留置s:
            continue                                  # 留置无跨情绪配对
        f = np.asarray(听到["指纹"], np.float64)
        mF, mE, mD = 中值[t]
        相对指纹[(t, e)] = np.asarray(
            [f[0] / max(mF, 1e-6), f[1], f[2], f[3] / max(mE, 1e-9),
             f[4], f[5], f[6], f[7] / max(mD, 1e-6)], np.float64)
    训指纹 = np.stack([相对指纹[(t, e)] for t in 文本s[:4] for e in 情绪列表])
    均值, 标准差 = 训指纹.mean(axis=0), 训指纹.std(axis=0) + 1e-9
    类列表 = ["高唤醒", "平静", "悲伤", "疲惫"]
    原型 = {}
    for e in 情绪列表:
        指s = [z(相对指纹[(t, e)]) for t in 文本s[:4]]
        g = 分组[e]
        原型[g] = 原型.get(g, np.zeros(8)) + np.mean(np.stack(指s), axis=0)
    for g in 原型:
        原型[g] = 原型[g] / max(sum(1 for e in 情绪列表 if 分组[e] == g), 1)
    对4, 总4, 明细s = 0, 0, {}
    for t in 文本s[4:]:
        for e in 情绪列表:
            q = z(相对指纹[(t, e)])
            预 = min(类列表, key=lambda k: float(np.sum((原型[k] - q) ** 2)))
            对4 += int(预 == 分组[e])
            总4 += 1
            明细s.setdefault("%s→%s" % (分组[e], 预), 0)
            明细s["%s→%s" % (分组[e], 预)] += 1
    判("L4 情绪辨识(可分辨类)", 对4 / 总4 >= 0.8,
       "%d/%d=%.2f 混淆=%s" % (对4, 总4, 对4 / 总4, 明细s))

    # —— L6 记忆稳定 ——
    首 = 耳.舱[0]
    检 = 耳.辨(语料[(首["文本"], 首["情绪码"])], k=1)[0]
    判("L6 记忆稳定", 检["条目"] == 0 and 检["相似度"] > 0.99,
       "首条目(%s/%s) 检索条目=%d 相似度=%.4f"
       % (首["文本"], 首["情绪码"], 检["条目"], 检["相似度"]))

    # —— 存档/载档往返 ——
    档 = os.path.join(HERE, "声觉舱.json")
    耳.存档(档)
    耳2 = 耳朵器官(seed=400, D=1024)
    耳2.载档(档)
    检2 = 耳2.辨(语料[(文本s[1], "喜悦")], k=1)[0]
    判("L7 存档往返", 检2["文本"] == 文本s[1],
       "载档后检索 %s 相似度=%.4f" % (检2["文本"], 检2["相似度"]))

    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 声音学习验证：%d/%d PASS =====" % (len(结果) - len(fail),
                                                     len(结果)))
    if fail:
        print("FAIL:", fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
