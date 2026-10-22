# -*- coding: utf-8 -*-
"""meeting69/实施/判据_S0.py —— S0 判据（预注册 03 §3；先写后跑）。

臂：
  A0 = log-mel 80 / 10 ms（v13 口径，对照）
  A2 = 耳蜗前端_S0（64 带 + 1 ms 场 + 事件层）
  A1 = 耳蜗场（连续轨，无事件）——与前两者同源；凡依赖事件的判据用 A2。
判据：J1 VOT MAE；J2 塞擦双事件；J3 擦音分类；J4 F0 探针；J5 边界；J6 适应；J7 确定性/RTF；
      G1 对照臂同跑；G3 sham；G4 无外部模型（本脚本不 import ASR/TTS）；G5 确定性。
落盘：meeting69/产物/S0/结果.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M68 = HERE.parent
S0 = M68 / "产物" / "S0"
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


前 = 载("耳蜗前端_S0", HERE / "耳蜗前端_S0.py")


def mel(x):
    import librosa
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=220, win_length=551,
                                       n_mels=80, fmin=0, fmax=SR / 2.0, power=2.0)
    return np.log(S + 1e-6).T.astype(np.float64)


# ─────────────────────── 估计器（S0 预注册口径） ───────────────────────
def VOT_A2(场未适):
    R = 场未适.astype(np.float64) + 1e-9
    高 = R[:, 32:].mean(axis=1)
    基 = np.array([np.median(高[max(0, t - 20): max(1, t - 2)]) for t in range(len(高))])
    t0 = int(0.15 * 1000)
    b = int(np.argmax(高[t0:] / (基[t0:] + 1e-9)) + t0)
    t1 = min(len(R), b + 300)
    低中 = R[b:t1, 8:24].mean(axis=1)
    d = np.diff(低中)
    k = int(np.argmax(d[3:])) + 3 if len(d) > 3 else int(np.argmax(d))
    return (b, b + k)


def VOT_A0(m):
    R = m + 1e-9
    高 = R[:, 40:].mean(axis=1); 低 = R[:, :20].mean(axis=1)
    d = np.diff(高); t0 = int(0.15 / 0.01)
    b = int(np.argmax(d[t0:]) + t0) + 1
    t1 = min(len(低), b + int(0.3 / 0.01))
    mx = 低[b:t1].max() if t1 > b else 0.0
    v = None
    for t in range(b + 1, t1 - 1):
        if 低[t] > 0.5 * mx and 低[t + 1] > 0.5 * mx:
            v = t; break
    return (b, v)


def 塞擦_A2(ev, gt_burst, gt_fric):
    爆 = [e for e in ev if e["type"] == "爆破" and e["t"] >= gt_burst - 0.05]
    摩 = [e for e in ev if e["type"] == "摩擦" and e["t"] >= gt_burst - 0.05]
    if not 爆 or not 摩:
        return None
    gap = (min(e["t"] for e in 摩) - 爆[0]["t"]) * 1000.0
    return dict(爆=round(爆[0]["t"], 4), 摩=round(min(e["t"] for e in 摩), 4), gap=round(gap, 1))


def 塞擦_A0(m, gt_burst):
    R = m + 1e-9
    高 = R[:, 40:].mean(axis=1)
    d = np.diff(高); b = int(np.argmax(d[int(0.15 / 0.01):]) + int(0.15 / 0.01)) + 1
    if b >= len(高) - 3:
        return None
    mx = 高[b:].max()
    起 = None
    for t in range(b + 1, len(高) - 2):
        if 高[t] > 0.6 * mx and 高[t + 1] > 0.6 * mx and 高[t + 2] > 0.6 * mx:
            起 = t; break
    if 起 is None:
        return None
    return dict(爆=round(b * 0.01, 4), 摩=round(起 * 0.01, 4), gap=round((起 - b) * 10.0, 1))


def 擦特征_A2(场):
    X = 场[300:450].astype(np.float64) if len(场) >= 450 else 场[-150:].astype(np.float64)
    return np.concatenate([X.mean(axis=0), X.std(axis=0)])


def 擦特征_A0(m):
    X = m[30:45] if len(m) >= 45 else m[-15:]
    return np.concatenate([X.mean(axis=0), X.std(axis=0)])


def 最近质心(特征s, 标签s, 训练掩码):
    X = np.asarray(特征s, np.float64); y = np.asarray(标签s)
    μ = X[训练掩码].mean(axis=0); σ = X[训练掩码].std(axis=0) + 1e-9
    Xn = (X - μ) / σ
    类 = sorted(set(y))
    心 = {c: Xn[训练掩码 & (y == c)].mean(axis=0) for c in 类}
    return [min(类, key=lambda c: float(np.sum((x - 心[c]) ** 2))) for x in Xn], 类


def F0岭_probe(特征s, f0s, 折):
    """闭合式岭回归（λ=1e-2），留一 tone 交叉；返回 R² / MAE。"""
    X = np.asarray(特征s, np.float64); y = np.asarray(f0s, np.float64)
    R2, MAE = [], []
    for 型 in sorted(set(折)):
        训 = 折 != 型; 测 = ~训
        if 测.sum() < 5 or 训.sum() < 5:
            continue
        μ = X[训].mean(axis=0); σ = X[训].std(axis=0) + 1e-9
        Xn = (X - μ) / σ
        A = np.concatenate([Xn, np.ones((len(Xn), 1))], axis=1)
        w = np.linalg.solve(A[训].T @ A[训] + 1e-2 * np.eye(A.shape[1]), A[训].T @ y[训])
        pred = A[测] @ w
        ss = float(np.sum((y[测] - y[测].mean()) ** 2))
        R2.append(1.0 - float(np.sum((y[测] - pred) ** 2)) / ss if ss > 1e-9 else 0.0)
        MAE.append(float(np.mean(np.abs(pred - y[测]))))
    return (float(np.mean(R2)) if R2 else 0.0, float(np.mean(MAE)) if MAE else 0.0)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    清 = json.load(open(S0 / "manifest.json", encoding="utf-8"))["真值"]
    报 = dict(工具="判据_S0.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              预注册="meeting69/03_预注册_S0_听觉事件化.md", 臂=dict(
                  A0="log-mel 80/10ms（对照）", A1="耳蜗场 1ms（连续轨）", A2="耳蜗场+事件层"))
    # 预处理：逐文件 A2 场 / A0 mel
    场 = {}; 场未 = {}; meld = {}
    for g in 清:
        x = 前.读wav(g["路径"])
        r = 前.分析(g["路径"])
        场[g["文件"]] = r["场"].astype(np.float64)
        场未[g["文件"]] = r["场未适"].astype(np.float64)
        meld[g["文件"]] = mel(x)
    # ── J1 VOT ──
    j1 = dict(A0=[], A2=[], sham=[])
    for g in 清:
        if g["类"] != "VOT":
            continue
        gt = (g["浊起秒"] - g["burst秒"]) * 1000.0
        b, v = VOT_A2(场未[g["文件"]])
        j1["A2"].append((v - b) * 1.0 - gt)
        b0, v0 = VOT_A0(meld[g["文件"]])
        j1["A0"].append(((v0 - b0) * 10.0 if v0 is not None else 1e3) - gt)
        # sham：事件时刻随机化（±25 ms，确定性种子）
        rs = np.random.RandomState(6900 + int(g["vot_ms"]))
        j1["sham"].append((v - b) * 1.0 + rs.uniform(-25, 25) - gt)
    j1_out = {k: dict(MAE=round(float(np.mean(np.abs(v))), 3), bias=round(float(np.mean(v)), 3))
              for k, v in j1.items()}
    J1 = (j1_out["A2"]["MAE"] <= 5.0 and (j1_out["A0"]["MAE"] - j1_out["A2"]["MAE"]) >= 3.0)
    报["J1_VOT"] = dict(细节=j1_out, 判="PASS" if J1 else "FAIL")
    # ── J2 塞擦 ──
    j2 = dict(A0=[], A2=[])
    for g in 清:
        if g["类"] != "塞擦":
            continue
        gt_gap = (g["摩擦起秒"] - g["burst秒"]) * 1000.0
        r2 = 塞擦_A2(前.分析(g["路径"])["事件"], g["burst秒"], g["摩擦起秒"])
        j2["A2"].append(0.0 if r2 and abs(r2["gap"] - gt_gap) <= 10 else 1.0)
        r0 = 塞擦_A0(meld[g["文件"]], g["burst秒"])
        j2["A0"].append(0.0 if r0 and abs(r0["gap"] - gt_gap) <= 10 else 1.0)
    j2_out = {k: round(1.0 - float(np.mean(v)), 3) for k, v in j2.items()}
    J2 = j2_out["A2"] >= 0.90
    报["J2_塞擦"] = dict(检出率=j2_out, 判="PASS" if J2 else "FAIL")
    # ── J3 擦音（5 类 × 5 重复；3 训 2 测） ──
    特征A0 = []; 特征A2 = []; 标签 = []; 掩码 = []
    for g in 清:
        if g["类"] != "擦音":
            continue
        标签.append(g["标签"])
        特征A0.append(擦特征_A0(meld[g["文件"]]))
        特征A2.append(擦特征_A2(场未[g["文件"]]))
        掩码.append(g["重复"] < 3)
    pred0, 类 = 最近质心(特征A0, 标签, np.array(掩码))
    pred2, _ = 最近质心(特征A2, 标签, np.array(掩码))
    acc = lambda p: float(np.mean([a == b for a, b in zip(np.array(p)[~np.array(掩码)], np.array(标签)[~np.array(掩码)])]))
    j3 = dict(A0=round(acc(pred0), 3), A2=round(acc(pred2), 3))
    J3 = j3["A2"] >= 0.75 and j3["A2"] >= j3["A0"] + 0.05
    报["J3_擦音"] = dict(准确率=j3, 判="PASS" if J3 else "FAIL")
    # ── J4 F0 探针 ──
    特0 = []; 特2 = []; 靶0 = []; 靶2 = []; 折0 = []; 折2 = []; f0mae = []
    for g in 清:
        if g["类"] != "声调":
            continue
        m = meld[g["文件"]]
        n = len(m)
        gt = np.interp((np.arange(n) * 0.01) - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
        sel0 = np.arange(n) >= 31
        特0.append(m[sel0]); 靶0.append(gt[sel0]); 折0.append([g["标签"]] * int(sel0.sum()))
        r = 前.分析(g["路径"])
        f = r["场未适"].astype(np.float64)
        nf = len(f)
        gtf = np.interp((np.arange(nf) * 0.001) - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
        sel2 = np.arange(nf) >= 310
        特2.append(f[sel2]); 靶2.append(gtf[sel2]); 折2.append([g["标签"]] * int(sel2.sum()))
        f0v = np.array(r["F0"]["f0"]); t = np.array(r["F0"]["t"])
        gt_at = np.interp(t - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
        ok = f0v > 0
        if ok.sum() > 5:
            f0mae.append(float(np.mean(np.abs(f0v[ok] - gt_at[ok]) / gt_at[ok])))
    X0 = np.concatenate(特0); X2 = np.concatenate(特2)
    y0 = np.concatenate(靶0); y2 = np.concatenate(靶2)
    折0v = np.concatenate([np.array(f, dtype=object) for f in 折0])
    折2v = np.concatenate([np.array(f, dtype=object) for f in 折2])
    r20, _ = F0岭_probe(X0, y0, 折0v); r22, _ = F0岭_probe(X2, y2, 折2v)
    j4 = dict(A0_R2=round(r20, 3), A2_R2=round(r22, 3),
              F0_MAE_rel=round(float(np.mean(f0mae)), 4) if f0mae else None)
    J4 = j4["A2_R2"] >= 0.5 and (j4["F0_MAE_rel"] is not None and j4["F0_MAE_rel"] <= 0.05)
    报["J4_F0"] = dict(细节=j4, 判="PASS" if J4 else "FAIL")
    # ── J5 边界 ──
    g = next(x for x in 清 if x["类"] == "边界")
    ev = 前.分析(g["路径"])["事件"]
    爆s = [e["t"] for e in ev if e["type"] == "爆破"]
    err2 = []
    for gt in (g["click秒"], g["burst秒"]):
        最 = min(爆s, key=lambda t: abs(t - gt)) if 爆s else None
        err2.append(abs(最 - gt) * 1000 if 最 is not None else 1e3)
    m = meld[g["文件"]]
    高 = m[:, 40:].mean(axis=1)
    d = np.diff(高)
    峰 = [int(i) + 1 for i in range(1, len(d) - 1) if d[i] > d[i - 1] and d[i] >= d[i + 1] and d[i] > np.percentile(d, 90)]
    err0 = []
    for gt in (g["click秒"], g["burst秒"]):
        最 = min(峰, key=lambda i: abs(i * 0.01 - gt)) if 峰 else None
        err0.append(abs(最 * 0.01 - gt) * 1000 if 最 is not None else 1e3)
    j5 = dict(A0_中位=round(float(np.median(err0)), 2), A2_中位=round(float(np.median(err2)), 2),
              A2_p90=round(float(np.percentile(err2, 90)), 2))
    J5 = j5["A2_中位"] <= 5.0 and j5["A2_p90"] <= 15.0
    报["J5_边界"] = dict(细节=j5, 判="PASS" if J5 else "FAIL")
    # ── J6 适应 ──
    g = next(x for x in 清 if x["类"] == "适应")
    f = 场[g["文件"]]
    早 = float(f[200:300].mean()); 晚 = float(f[1500:1800].mean())
    m = meld[g["文件"]]
    m早 = float(m[20:30].mean()); m晚 = float(m[150:180].mean())
    j6 = dict(A2_比=round(晚 / (早 + 1e-12), 4), A0_比=round(m晚 / (m早 + 1e-12), 4))
    J6 = j6["A2_比"] <= 0.5
    报["J6_适应"] = dict(细节=j6, 判="PASS" if J6 else "FAIL")
    # ── J7 确定性 / 实时率（真实 60 s 片段；warm 口径） ──
    实 = Path(r"<path>\SpeakerA7代模型训练数据\(1).wav")
    if 实.exists():
        r1 = 前.分析(实); r2 = 前.分析(实)
        同 = r1["sha256"] == r2["sha256"]
        率 = min(r1["实时率"], r2["实时率"])
    else:
        同 = None; 率 = None
    J7 = (同 is True) and (率 is not None and 率 <= 0.5)
    报["J7_确定性"] = dict(两趟同=同, 实时率=率, 判="PASS" if J7 else "FAIL")
    # ── 结论 ──
    主判 = dict(J1=J1, J2=J2, J3=J3, J4=J4, J5=J5, J6=J6, J7=J7)
    过 = [k for k, v in 主判.items() if v]
    报["判定"] = dict(主判=主判, 过=过, 结论=("PASS" if len(过) == 7 else ("部分" if 过 else "FAIL")),
                     说明="S0 不测可懂/不测像人；A0 全面不差 ⇒ 换表示证伪（见 G1）")
    (S0 / "结果.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
