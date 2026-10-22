# -*- coding: utf-8 -*-
"""meeting53 探针7：手感权重扫描（结果/动作绑定权重 × 内心词权重）。"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import V5_双通道 as V5
import 判据_V5_双通道 as J

np.seterr(all="ignore")
种子 = V5.种子默认
for w结, w动, w内 in [(0.5, 0.5, 1.2), (0.25, 0.25, 1.0), (0.2, 0.2, 0.9), (0.15, 0.15, 0.8)]:
    V5.w结果, V5.w动作, V5.w内词 = w结, w动, w内
    体 = V5.跑单脑(拍数=400, 测拍=0, 种子=种子)["体"]
    rec = J.跑测(体, 种子, 测拍数=200)
    对 = 错 = 0
    for x in rec:
        s = [w for w in x["词"] if w in V5.侧词]
        if s:
            对 += int(V5.侧词[x["观测"]] == s[0]); 错 += int(V5.侧词[x["观测"]] != s[0])
    一致 = 对 / (对 + 错 + 1e-9)
    # 冻结臂
    体2 = copy.deepcopy(体)
    rec2 = J.跑测(体2, 种子, 内心覆盖="冻结", 测拍数=200)
    冻 = np.mean([tuple(rec2[i]["词"]) != tuple(rec[i]["词"]) for i in range(200)])
    # 置换状态
    o列 = np.array([x["o用"] for x in rec]); 内列 = np.array([x["内心"] for x in rec])
    替 = np.roll(o列, -1, axis=0)
    置 = np.mean([J._生成(体, 替[i], 内列[i]) != tuple(rec[i]["词"]) for i in range(200)])
    # 解码
    样 = [x for x in rec if any(w in V5.侧词 for w in x["词"])]
    X = np.zeros((len(样), 24)); y = np.array([x["观测"] for x in 样])
    for i, x in enumerate(样):
        for w in x["词"]:
            X[i, V5.词id[w]] += 1
    半 = len(X) // 2
    for c in range(5):
        m = X[:半][y[:半] == c]
    def _质(c):
        m = X[:半][y[:半] == c]
        return m.mean(axis=0) if len(m) else np.zeros(24)
    质 = {c: _质(c) for c in range(5)}
    预 = [min(range(5), key=lambda c: float(np.linalg.norm(X[i] - 质[c]))) for i in range(半, len(X))]
    acc = np.mean([预[i - 半] == y[i] for i in range(半, len(X))])
    print("w结=%.2f w动=%.2f w内=%.2f 侧词一致=%.3f 冻结改变=%.3f 置换改变=%.3f 解码=%.3f"
          % (w结, w动, w内, 一致, 冻, 置, acc))
