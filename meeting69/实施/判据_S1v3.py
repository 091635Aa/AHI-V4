# -*- coding: utf-8 -*-
"""meeting69/实施/判据_S1v3.py —— S1 v3 判据（预注册 16；先写后跑；零训练）。

与 `判据_S1v2.py`（v2）的差别（只在白名单内，16 §2）：
  ① 听场归一化 q 下限 0.05 → **1.0**（去近静音带放大）；
  ② 焦点 = **ε 单独格级 argmax + 窗口居中**（显著/IOR 不进 argmax）；
  ③ A-4 口径显式化：`f′_k`（本拍输出、下一拍使用）覆盖**本拍** max-ε 格。
其余（判据阈值、门、sham、数据、A-5/H6 信息臂）与 `判据_S1v2.py` 逐字相同。
落盘：meeting69/产物/S1v3/结果.json
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
S1 = M69 / "产物" / "S1"          # 冻结数据源（manifest/刺激）
S2 = M69 / "产物" / "S1v2"        # v2 突变刺激源
S3 = M69 / "产物" / "S1v3"        # v3 读数与派生件
SR = 22050
Q下限 = 1.0


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


前 = 载("听觉前端", HERE / "听觉前端.py")
耳 = 载("耳蜗前端_S0_2", HERE / "耳蜗前端_S0_2.py")
合 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")
F32 = 前.场.带心32(耳.中心频率())

_载 = {}


def 取(路):
    if 路 not in _载:
        r = 前.载片段(路)
        _载[路] = (r, np.log(np.asarray(r["场未适"], np.float64) + 1e-3))
    return _载[路]


def 跑(路, 模式="主臂", 拍ms=100):
    r, L = 取(路)
    return 前.听觉前端(拍ms=拍ms).拍(r, 模式=模式, L=L, 焦点选择="ε", q下限=Q下限)


def 类(清单, k):
    return [x for x in 清单 if x["类"] == k]


def 带索引(hz):
    return int(np.argmin(np.abs(F32 - float(hz))))


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    清单 = json.load(open(S1 / "manifest.json", encoding="utf-8"))
    真 = 清单["真值"]; 真实 = 清单["真实"]
    (S3 / "真实").mkdir(parents=True, exist_ok=True)
    (S3 / "刺激").mkdir(parents=True, exist_ok=True)
    报 = dict(工具="判据_S1v3.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              预注册="meeting69/16_预注册_S1v3.md",
              前端="耳蜗前端_S0_2.py（v0.3，S0.2 7/7）", 拍率主臂="100 ms（10 Hz）",
              焦点选择="ε 单独格级 argmax + 窗口居中（v3）; q下限=1.0", 真实片段=[])

    # ── v2 突变刺激（14 §4）：1 s 静音 + 爆破 + 0.2 s 静音 + 1.5 s 元音 ──
    突变清单 = []
    for k, (中心, f0) in enumerate(((2500.0, 120.0), (3500.0, 130.0), (4500.0, 140.0), (5500.0, 150.0))):
        爆 = 合.爆破(中心=中心, 带宽=2200.0, 种子=20 + k)
        x = 合.拼接(合.静默(1.0), 爆, 合.静默(0.2), 合.元音(f0, 1.5))
        路 = S3 / "刺激" / f"S1v3_突变{k}.wav"
        合.写wav(路, x)
        突变清单.append(dict(文件=f"S1v3_突变{k}.wav", 路径=str(路), 类="突变v2",
                             burst秒=1.0, burst中心Hz=中心, 元音起秒=1.212, 子=k))

    # ── 真实语音 5 s 切片（只读源；派生件写产物/S1v2/真实/） ──
    def 真实wav(实):
        路 = S3 / "真实" / (Path(实["文件"]).stem + f"@{实['起始秒']:.0f}.wav")
        if not 路.exists():
            x = 耳.读wav(实["路径"])
            i0 = int(float(实["起始秒"]) * SR); i1 = i0 + int(float(实["时长秒"]) * SR)
            合.写wav(路, x[i0:i1])
        报["真实片段"].append(dict(源=实["文件"], 派生=str(路)))
        return 路

    真实路 = [真实wav(x) for x in 真实]

    # ── 主臂：合成 + 真实 ──
    记 = {}
    for g in 真:
        记[g["文件"]] = 跑(g["路径"])
    for 路 in 真实路:
        记[str(路)] = 跑(路)

    # ── A-1a 决定性（同历史两趟） ──
    a1a = []
    for 路 in (真[0]["路径"], 真实路[0]):
        记2 = 跑(路)
        记1 = 记[真[0]["文件"]] if 路 == 真[0]["路径"] else 记[str(路)]
        同 = (记1["sha_听场"] == 记2["sha_听场"] and 记1["sha_焦点链"] == 记2["sha_焦点链"]
              and 记1["sha_遥测"] == 记2["sha_遥测"])
        a1a.append(dict(文件=Path(路).name, 同=bool(同), sha=[记1["sha_听场"], 记2["sha_听场"]]))
    A1a = all(x["同"] for x in a1a)
    报["A-1a_决定性"] = dict(细节=a1a, 判="PASS" if A1a else "FAIL")

    # ── A-1b 状态依赖（不同历史：2 s 静音 vs 2 s 不同语音） ──
    a1b = []
    for i, 路 in enumerate(真实路[:3]):
        tgt = 耳.读wav(路)                     # 5 s（已 22.05k mono）
        tgt = tgt[int(2.0 * SR):int(4.0 * SR)]   # 目标 2 s 段
        histA = np.zeros_like(tgt)               # 前史 A：2 s 静音
        srcB = 耳.读wav(真实路[(i + 1) % len(真实路)])
        histB = srcB[int(2.0 * SR):int(4.0 * SR)]  # 前史 B：另一条语音 2 s
        甲 = S3 / "刺激" / f"A1B_{i}_A.wav"; 乙 = S3 / "刺激" / f"A1B_{i}_B.wav"
        合.写wav(甲, 合.拼接(histA, tgt)); 合.写wav(乙, 合.拼接(histB, tgt))
        rA = 跑(甲); rB = 跑(乙)
        链A = [e for e in rA["链"] if e["t"] >= 2.0][:5]
        链B = [e for e in rB["链"] if e["t"] >= 2.0][:5]
        n = min(len(链A), len(链B))
        格A = set(); 格B = set()
        for e in 链A:
            格A |= {(e["b0"] + db, e["s0"] + ds) for db in range(4) for ds in range(8)}
        for e in 链B:
            格B |= {(e["b0"] + db, e["s0"] + ds) for db in range(4) for ds in range(8)}
        交 = len(格A & 格B); 并 = len(格A | 格B)
        jac = 交 / max(1, 并)
        位移 = float(np.mean([abs(math.log2(链A[j]["带心Hz"] / 链B[j]["带心Hz"])) for j in range(n)])) if n else 0.0
        过 = bool(jac <= 0.6 or 位移 >= 1.0 / 3.0)
        a1b.append(dict(对=i, n拍=n, Jaccard=round(jac, 4), 带心位移倍频程=round(位移, 4), 过=过))
    A1b = sum(1 for x in a1b if x["过"]) >= max(1, math.ceil(2 * len(a1b) / 3))
    报["A-1b_状态依赖"] = dict(细节=a1b, 判="PASS" if A1b else "FAIL")

    # ── A-2 适应 + 突变 ──
    稳 = 跑(类(真, "稳态")[0]["路径"])
    s = np.array(稳["s"]); k = np.array(稳["拍"])
    s1 = float(s[(k >= 1) & (k <= 9)].mean()); s3 = float(s[(k >= 20) & (k <= 29)].mean())
    稳比 = s3 / (s1 + 1e-12); A2a = bool(稳比 <= 0.40)
    突变 = []
    for g in 突变清单:
        rr = 跑(g["路径"]); ss = np.array(rr["s"]); kk = np.array(rr["拍"])
        w = np.where((kk >= 10) & (kk <= 12))[0]
        peak = float(ss[w].max()) if len(w) else 0.0
        bi = 带索引(g["burst中心Hz"])
        hit = False
        for j, kv in enumerate(kk):
            if 10 <= kv <= 12:          # 爆破被采到的拍（含）后 1 拍内的 f′
                b0, s0 = rr["焦点"][j]
                if not (b0 + 3 < bi - 1 or b0 > bi + 1):
                    hit = True; break
        突变.append(dict(文件=g["文件"], 峰=round(peak, 3), 峰过=bool(peak >= 0.5), 命中=bool(hit)))
    命中率 = float(np.mean([x["命中"] for x in 突变])) if 突变 else 0.0
    A2b = bool(all(x["峰过"] for x in 突变) and 命中率 >= 0.8)
    A2 = bool(A2a and A2b)
    报["A-2_适应与突变"] = dict(稳态=dict(第1秒=round(s1, 4), 第3秒=round(s3, 4), 比=round(稳比, 4), 判="PASS" if A2a else "FAIL"),
                              突变=dict(逐条=突变, 命中率=round(命中率, 3), 判="PASS" if A2b else "FAIL"),
                              判="PASS" if A2 else "FAIL")

    # ── A-3 无标签非平凡（真实 20 条） ──
    s全 = []; 带 = []
    for 路 in 真实路:
        rr = 记[str(路)]
        s全 += rr["s"]
        带 += [带索引(e["带心Hz"]) for e in rr["链"]]
    s全 = np.array(s全)
    计数 = np.bincount(带, minlength=32).astype(float)
    概率 = 计数[计数 > 0] / 计数.sum()
    熵 = float(-np.sum(概率 * np.log2(概率)))
    方差 = float(np.var(s全))
    A3 = bool(方差 > 0.01 and 熵 >= 1.0)
    报["A-3_无标签非平凡"] = dict(s方差=round(方差, 5), 带心熵bit=round(熵, 4),
                               带心分布={int(i): int(c) for i, c in enumerate(计数) if c},
                               判="PASS" if A3 else "FAIL")

    # ── A-4 误差驱动命中（真实 20 条；v3 口径 = f′ 覆盖本拍 max-ε；另报异拍参考） ──
    def a4命中(记录们, 配对="同拍"):
        命中 = 0; 总 = 0
        for rr in 记录们:
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
    主命中率, 主中, 主总 = a4命中([记[str(p)] for p in 真实路], "同拍")
    异命中率, 异中, 异总 = a4命中([记[str(p)] for p in 真实路], "异拍")
    报["A-4_误差驱动"] = dict(命中率=round(主命中率, 4), 命中=主中, 总=主总,
                            同拍命中率=round(主命中率, 4), 异拍命中率=round(异命中率, 4),
                            随机基线=round(32 / 1024, 4), 判=("PASS" if 主命中率 >= 0.80 else "FAIL"))

    # ── sham（A-4 三 sham + 静音） ──
    sham = {}
    for 模式 in ("随机焦点", "仅显著", "固定焦点"):
        记录们 = [跑(p, 模式=模式) for p in 真实路]
        r, h, t = a4命中(记录们, "同拍")
        sham[模式] = dict(命中率=round(r, 4), 命中=h, 总=t)
    # 固定焦点下 A-2 突变命中（v3 突变刺激）
    固定突变 = []
    for g in 突变清单:
        rr = 跑(g["路径"], 模式="固定焦点"); bi = 带索引(g["burst中心Hz"]); hit = False
        for j, kv in enumerate(rr["拍"]):
            if 10 <= kv <= 12:
                b0, s0 = rr["焦点"][j]
                if not (b0 + 3 < bi - 1 or b0 > bi + 1):
                    hit = True; break
        固定突变.append(hit)
    静 = 跑(类(真, "静音")[0]["路径"])
    非平凡 = int(sum(1 for e in 静["链"] if e["s"] > 0.1))
    sham["固定焦点"]["突变命中率"] = round(float(np.mean(固定突变)), 3)
    sham["静音"] = dict(非平凡链条目=非平凡, 判=("通过" if 非平凡 <= 2 else "失败"))
    S_sham = bool(sham["随机焦点"]["命中率"] <= 0.15
                  and sham["仅显著"]["命中率"] <= 主命中率 - 0.10
                  and sham["固定焦点"]["命中率"] <= 0.12
                  and sham["固定焦点"]["突变命中率"] <= 0.0 + 1e-9
                  and 非平凡 <= 2)
    报["sham"] = dict(细节=sham, 全失败=bool(S_sham))

    # ── A-5（信息臂·延后 S3） ──
    a5 = []
    for g in 类(真, "重复"):
        rr = 跑(g["路径"]); ss = np.array(rr["s"]); kk = np.array(rr["拍"])
        ons = [0.5 + i * g["周期秒"] for i in range(5)]
        值 = []
        for on in ons:
            w = np.where((kk * 0.1 >= on) & (kk * 0.1 <= on + 0.15))[0]
            值.append(float(ss[w].max()) if len(w) else 0.0)
        a5.append(dict(子=g["子"], 起始s=[round(v, 3) for v in 值],
                       比3_1=round(值[2] / (值[0] + 1e-12), 3)))
    新 = [x["起始s"][0] for x in a5]; 旧 = [v for x in a5 for v in x["起始s"][2:]]
    auc = float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in 新 for b in 旧])) if 新 and 旧 else 0.0
    报["A-5_重复收敛"] = dict(细节=a5, AUC=round(auc, 4), 效力="信息臂：零训练无预测可塑，延后 S3（09 v1.1）")

    # ── A-7 禁标签 ──
    禁 = ("torch", "transformers", "whisper", "wav2vec", "hubert", "speechbrain", "espnet",
          "funasr", "modelscope", "onnx", "tensorflow", "jax", "keras", "sentencepiece", "llama")
    命中 = []
    for fn in ("听觉场.py", "算子_流分离.py", "听觉工作区.py", "协议_听觉源.py", "听觉前端.py", "判据_S1.py"):
        for line in (HERE / fn).read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t.startswith("import ") or t.startswith("from "):
                for bad in 禁:
                    if bad in t.lower():
                        命中.append(f"{fn}: {t}")
    A7 = bool(not 命中)   # 结构上无标签通道；一致率=1.0（只有无标签条件）
    报["A-7_禁标签"] = dict(一致率=1.0, 审计命中=命中, 判="PASS" if A7 else "FAIL",
                           说明="S1 无 ASR/文本/标签通道（结构不存在）；一致率按 1.0 记")

    # ── A-8 工程 ──
    全部cost = []
    最大内存 = 0.0
    for 路 in list(真实路) + [g["路径"] for g in 真]:
        rr = 记.get(str(路)) or 记.get(Path(路).name)
        if rr is None:
            continue
        全部cost += rr["cost"]
        r, L = 取(路)
        内 = L.nbytes + r["场"].nbytes + r["场未适"].nbytes + 1024 * 4 * 2 + 192 * 4
        最大内存 = max(最大内存, 内 / 1024 / 1024)
    全部cost = np.array(全部cost)
    P95 = float(np.percentile(全部cost, 95)) * 1000
    中位 = float(np.median(全部cost)) * 1000
    # 两趟 sha（同输入两次）
    双趟同 = A1a
    A8 = bool(P95 <= 33.0 and 中位 <= 12.0 and 双趟同 and 最大内存 <= 10.0)
    报["A-8_工程"] = dict(每拍P95ms=round(P95, 3), 每拍中位ms=round(中位, 3), 拍数=int(len(全部cost)),
                         两趟sha同=bool(双趟同), 最大净资源MB=round(最大内存, 3), 判="PASS" if A8 else "FAIL")

    # ── H6 流分离（信息臂） ──
    双 = 记[类(真, "双流")[0]["文件"]]; 单 = 记[类(真, "单流")[0]["文件"]]
    双流数 = float(np.median(双["流数"])); 单流数 = float(np.median(单["流数"]))
    每流 = [v for d in 双["流均ε"] for v in d.values()]
    混合 = float(np.mean([x for x in 双["活跃ε均"] if x > 0])) if any(x > 0 for x in 双["活跃ε均"]) else 0.0
    比 = (float(np.mean(每流)) / 混合) if 每流 and 混合 > 0 else None
    H6 = bool(双流数 == 2 and 比 is not None and 比 < 0.6)
    报["H6_流分离"] = dict(双流流数中位=双流数, 单流流数中位=单流数,
                         每流均ε=round(float(np.mean(每流)), 6) if 每流 else None,
                         混合均ε=round(混合, 6), 比=round(比, 4) if 比 is not None else None,
                         判=("PASS" if H6 else "FAIL"), 效力="信息臂，不进 S1 门")

    # ── 40/50 ms 拍率消融（信息臂） ──
    消融 = {}
    for 拍ms in (40, 50):
        记们 = [跑(p, 拍ms=拍ms) for p in 真实路[:10]]
        命中率, _, _ = a4命中(记们)
        s2 = np.concatenate([np.array(r["s"]) for r in 记们])
        rr稳 = 跑(类(真, "稳态")[0]["路径"], 拍ms=拍ms)
        sS = np.array(rr稳["s"]); kS = np.array(rr稳["拍"])
        比S = float(sS[(kS * 拍ms / 1000 >= 2.0) & (kS * 拍ms / 1000 < 3.0)].mean()
                    / (sS[(kS * 拍ms / 1000 < 1.0)].mean() + 1e-12))
        消融[f"{拍ms}ms"] = dict(A4命中率=round(命中率, 4), s方差=round(float(np.var(s2)), 5),
                               稳态比=round(比S, 4), 拍数=int(len(s2)))
    报["拍率消融"] = 消融

    # ── 判定 ──
    主判 = dict(**{"A-1a": A1a, "A-1b": A1b, "A-2": A2, "A-3": A3, "A-4": bool(主命中率 >= 0.80),
                  "A-7": A7, "A-8": A8})
    过 = [k for k, v in 主判.items() if v]
    门 = bool(all(主判.values()) and S_sham)
    判死 = None
    if 主命中率 <= sham["随机焦点"]["命中率"] + 0.10:
        判死 = "A-4 未超随机 sham+0.10 ⇒ 判死采样机制（H0）"
    elif 稳比 > 0.80:
        判死 = "A-2 适应比 >0.80 ⇒ 判死适应"
    报["判定"] = dict(主判=主判, 过=过, sham全失败=bool(S_sham),
                      结论=("PASS" if 门 else ("部分" if 过 else "FAIL")),
                      进S2=bool(门), 判死=判死,
                      说明="S1 v3：q下限 1.0 + 焦点=ε 单独格级 argmax+窗居中 + A-4 同拍口径；阈值/门/sham 不变；不测可懂/不测像人")
    (S3 / "结果.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(主())


