# -*- coding: utf-8 -*-
"""meeting69/实施/判据_S2A.py —— S2-A 音频独臂判据（预注册 `24`；先写后跑；零训练）。

只换输入音频，前端/阈值/sham 与 S1 v6 冻结口径逐字相同：
  拍 100 ms；焦点=ε 单独格级 argmax+窗居中；q下限 1.0；意外通道=原值适应；尺度 0.08；A 计权。
读数落盘：meeting69/产物/S2A/结果.json
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
S2A = M69 / "产物" / "S2A"
S1真 = M69 / "产物" / "S1" / "真实"
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


前 = 载("听觉前端", HERE / "听觉前端.py")
耳 = 载("耳蜗前端_S0_2", HERE / "耳蜗前端_S0_2.py")
F32 = 前.场.带心32(耳.中心频率())
_载 = {}


def 取(路):
    if 路 not in _载:
        r = 前.载片段(路)
        _载[路] = (r, np.log(np.asarray(r["场未适"], np.float64) + 1e-3))
    return _载[路]


def 跑(路, 模式="主臂"):
    r, L = 取(路)
    return 前.听觉前端(拍ms=100).拍(r, 模式=模式, L=L, 焦点选择="ε", q下限=1.0,
                                   意外通道="原值适应", 意外尺度=0.08, 听感权=True)


def 带索引(hz):
    return int(np.argmin(np.abs(F32 - float(hz))))


def a4命中(rr, 配对="同拍"):
    命中 = 0; 总 = 0
    mx = rr["maxε"]; fo = rr["焦点"]; kk = rr["拍"]
    for i in range(len(kk)):
        if 配对 == "同拍":
            m = mx[i]
        else:
            if i == 0:
                continue
            m = mx[i - 1]
        b = m // 32; sl = m % 32
        b0, s0 = fo[i]
        if b0 <= b <= b0 + 3 and s0 <= sl <= s0 + 7:
            命中 += 1
        总 += 1
    return (命中 / 总) if 总 else 0.0, 命中, 总


def 格集(链, 前K=None):
    g = set()
    for e in (链[:前K] if 前K else 链):
        for db in range(4):
            for ds in range(8):
                g.add((e["b0"] + db, e["s0"] + ds))
    return g


def 链熵(链):
    计 = np.bincount([带索引(e["带心Hz"]) for e in 链], minlength=32).astype(float)
    p = 计[计 > 0] / 计.sum()
    return float(-np.sum(p * np.log2(p)))


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    自声 = {名: S2A / "自声" / f"{名}.wav" for 名 in ("平静", "高兴", "低落")}
    真人 = {"(5)@10": S1真 / "(5)@10.wav", "(64)@10": S1真 / "(64)@10.wav"}
    退化 = S2A / "自声" / "平静_噪5dB.wav"
    报 = dict(工具="判据_S2A.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              预注册="meeting69/24_预注册_S2A_音频独臂.md", 来源="产物/S2A/结果.json",
              口径="S1 v6 冻结口径（只换输入音频）", 声源清单=json.loads((S2A / "声源清单.json").read_text(encoding="utf-8")))

    # ── 主臂：3 条自声 + 2 条真人参考 ──
    记 = {}
    for 名, 路 in 自声.items():
        记[名] = 跑(str(路))
    for 名, 路 in 真人.items():
        记[名] = 跑(str(路))
    记["平静_噪5dB"] = 跑(str(退化))

    # ── S2A-1 自声入耳非平凡（3 条合并） ──
    s全 = np.concatenate([np.array(记[k]["s"]) for k in 自声])
    链全 = [e for k in 自声 for e in 记[k]["链"]]
    s方 = float(np.var(s全)); 熵 = 链熵(链全)
    S1 = bool(s方 > 0.01 and 熵 >= 1.0)
    报["S2A-1_自声非平凡"] = dict(s方差=round(s方, 5), 带心熵bit=round(熵, 4), 拍数=int(len(s全)),
                                判="PASS" if S1 else "FAIL")

    # ── S2A-2 自声可跟踪（每条 A-4 + 三 sham） ──
    a2 = []
    for 名 in 自声:
        主臂 = 记[名]
        r主, h主, t主 = a4命中(主臂)
        sham = {}
        for 模式 in ("随机焦点", "仅显著", "固定焦点"):
            r, h, t = a4命中(跑(str(自声[名]), 模式=模式))
            sham[模式] = round(r, 4)
        过 = bool(r主 >= 0.80 and sham["随机焦点"] <= 0.15
                  and sham["仅显著"] <= r主 - 0.10 and sham["固定焦点"] <= 0.12)
        a2.append(dict(情绪=名, 命中率=round(r主, 4), 命中=h主, 总=t主, sham=sham, 过=过))
    S2 = bool(all(x["过"] for x in a2))
    报["S2A-2_自声可跟踪"] = dict(细节=a2, 判="PASS" if S2 else "FAIL")

    # ── S2A-3 自声状态依赖（两两 Jaccard / 带心位移） ──
    a3 = []
    名单 = list(自声)
    for i in range(len(名单)):
        for j in range(i + 1, len(名单)):
            A = 记[名单[i]]["链"][4:9]; B = 记[名单[j]]["链"][4:9]
            n = min(len(A), len(B))
            交 = len(格集(A) & 格集(B)); 并 = len(格集(A) | 格集(B))
            jac = 交 / max(1, 并)
            位移 = float(np.mean([abs(math.log2(A[k]["带心Hz"] / B[k]["带心Hz"])) for k in range(n)])) if n else 0.0
            a3.append(dict(对=f"{名单[i]}-{名单[j]}", Jaccard=round(jac, 4),
                           带心位移倍频程=round(位移, 4), 过=bool(jac <= 0.6 or 位移 >= 1 / 3)))
    S3 = bool(sum(1 for x in a3 if x["过"]) == len(a3))
    报["S2A-3_自声状态依赖"] = dict(细节=a3, 判="PASS" if S3 else "FAIL")

    # ── S2A-4 真人参考同判 ──
    a4 = []
    for 名, rr in 记.items():
        if 名 in 自声 or 名 == "平静_噪5dB":
            continue
        r, h, t = a4命中(rr)
        a4.append(dict(片段=名, 命中率=round(r, 4), 命中=h, 总=t, 过=bool(r >= 0.80)))
    S4 = bool(all(x["过"] for x in a4))
    报["S2A-4_真人参考"] = dict(细节=a4, 判="PASS" if S4 else "FAIL")

    # ── S2A-5 决定性（自声-平静 两趟） ──
    再 = 跑(str(自声["平静"])); 首 = 记["平静"]
    同 = bool(首["sha_听场"] == 再["sha_听场"] and 首["sha_焦点链"] == 再["sha_焦点链"]
              and 首["sha_遥测"] == 再["sha_遥测"])
    S5 = 同
    报["S2A-5_决定性"] = dict(两趟sha同=同, sha=dict(听场=[首["sha_听场"], 再["sha_听场"]],
                                                 焦点链=[首["sha_焦点链"], 再["sha_焦点链"]]),
                               判="PASS" if S5 else "FAIL")

    # ── S2A-6 工程 ──
    全cost = []; 最大内 = 0.0
    for 名, rr in 记.items():
        全cost += rr["cost"]
        路 = str(自声.get(名) or 真人.get(名) or 退化)
        r, L = 取(路)
        内 = L.nbytes + r["场"].nbytes + r["场未适"].nbytes + 1024 * 4 * 2 + 192 * 4
        最大内 = max(最大内, 内)
    全cost = np.array(全cost)
    P95 = float(np.percentile(全cost, 95)) * 1000; 中位 = float(np.median(全cost)) * 1000
    S6 = bool(P95 <= 33.0 and 中位 <= 12.0 and 最大内 / 1024 / 1024 <= 10.0)
    报["S2A-6_工程"] = dict(每拍P95ms=round(P95, 3), 每拍中位ms=round(中位, 3),
                            拍数=int(len(全cost)), 最大净资源MB=round(最大内 / 1024 / 1024, 3),
                            判="PASS" if S6 else "FAIL")

    # ── 信息臂 ──
    退 = 记["平静_噪5dB"]
    退r, _, _ = a4命中(退)
    smed = {名: round(float(np.median(rr["s"])), 4) for 名, rr in 记.items()}
    事件率 = {}
    for 名, 路 in list(自声.items()) + list(真人.items()) + [("平静_噪5dB", 退化)]:
        r, _ = 取(str(路))
        事件率[名] = round(len(r["事件"]) / max(r["时长秒"], 1e-9), 2)
    中心 = {名: round(float(np.median([e["带心Hz"] for e in rr["链"]])), 1) for 名, rr in 记.items()}
    报["信息臂"] = dict(退化SNR5dB=dict(s中位=smed["平静_噪5dB"], A4=round(退r, 4),
                                      方向="退化应更意外或更难跟踪（信息臂，不进门）"),
                      事件率_每秒=事件率, s中位=smed, 焦点带心中位Hz=中心,
                      说明="不设「像人」门槛；不进 S2-A 门")

    # ── 判定 ──
    主判 = {"S2A-1": S1, "S2A-2": S2, "S2A-3": S3, "S2A-4": S4, "S2A-5": S5, "S2A-6": S6}
    门 = bool(all(主判.values()))
    判死 = None
    if not S2:  # 逐条细查判死条款 1
        for x in a2:
            if x["命中率"] <= x["sham"]["随机焦点"] + 0.10:
                判死 = f"自声-{x['情绪']} A-4 ≤ 随机 sham+0.10 ⇒ 判死「自声不可入耳」"
                break
    if 判死 is None and not S1:
        判死 = "自声方差≤0.01 或 熵<1.0 ⇒ 判死「自声听场退化」"
    报["判定"] = dict(主判=主判, 结论=("PASS" if 门 else "FAIL"), 判死=判死,
                      S2A门开=bool(门),
                      S2B="未测·待第二模态（A-6/McGurk/唇读；≥5 min 正面视频或参数级口形）",
                      说明="S2-A 只判「自己说的话能否被自己的耳朵按 S1 机制采成非平凡听场」；不测可懂/人耳/像人/视听绑定")
    (S2A / "结果.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报["判定"], ensure_ascii=False, indent=1))
    for k in ("S2A-1_自声非平凡", "S2A-2_自声可跟踪", "S2A-3_自声状态依赖",
              "S2A-4_真人参考", "S2A-5_决定性", "S2A-6_工程", "信息臂"):
        print(k, json.dumps(报[k], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(主())
