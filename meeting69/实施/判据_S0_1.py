# -*- coding: utf-8 -*-
"""meeting69/实施/判据_S0_1.py —— S0.1 判据（预注册 06；先写后跑）。

臂：
  A0 = log-mel 80 / 10 ms（v13 口径，对照；估计器与 S0 判据件逐行相同）
  A2 = 耳蜗前端 v0.2（F0 估计器 v1 + 三适应变体按 06 §4 选择规则选一）
判据（与 03 §3.1 逐字相同，未改）：
  J1 VOT MAE；J2 塞擦双事件；J3 擦音分类；J4 F0（MAE≤5% 且场→F0 探针 R²≥0.5）；
  J5 边界；J6 适应（后段/前段 ≤0.5）；J7 确定性/RTF。
附加：
  J6 三变体对比（V1 除式5/100 / V2 除式5/200 / V3 减式）→ 按 06 §4 选择；
  J4-G 泛化臂（8 条新声调；信息臂，不改 S1 门）；
  G3 sham；G4 静态 import 审计；G5 两趟 sha256。
落盘：meeting69/产物/S0.1/结果.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
S0 = M69 / "产物" / "S0"
S01 = M69 / "产物" / "S0.1"
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


前 = 载("耳蜗前端_S0_1", HERE / "耳蜗前端_S0_1.py")

_缓存 = {}


def 取(路, 变体):
    k = (str(路), 变体)
    if k not in _缓存:
        _缓存[k] = 前.分析(路, 变体=变体)
    return _缓存[k]


def mel(x):
    import librosa
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=220, win_length=551,
                                       n_mels=80, fmin=0, fmax=SR / 2.0, power=2.0)
    return np.log(S + 1e-6).T.astype(np.float64)


# ─────────────────────── 估计器（与 S0 判据件逐行相同） ───────────────────────
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


def J4_评估(清, 声调集, 变体, meld=None):
    """返回 (A2_R2, F0_MAE_rel, 覆盖明细)。与 S0 J4 同口径；F0 用 v0.2 估计器。"""
    特2 = []; 靶2 = []; 折2 = []; f0mae = []; 覆盖 = {}
    for g in 声调集:
        r = 取(g["路径"], 变体)
        f = r["场未适"].astype(np.float64)
        nf = len(f)
        gtf = np.interp((np.arange(nf) * 0.001) - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
        sel2 = np.arange(nf) >= 310
        特2.append(f[sel2]); 靶2.append(gtf[sel2]); 折2.append([g["标签"]] * int(sel2.sum()))
        f0v = np.array(r["F0"]["f0"]); t = np.array(r["F0"]["t"])
        gt_at = np.interp(t - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
        ok = f0v > 0
        覆盖[g["标签"]] = round(float(ok.mean()), 3)
        if ok.sum() > 5:
            f0mae.append(float(np.mean(np.abs(f0v[ok] - gt_at[ok]) / gt_at[ok])))
    X2 = np.concatenate(特2); y2 = np.concatenate(靶2)
    折2v = np.concatenate([np.array(f, dtype=object) for f in 折2])
    r22, _ = F0岭_probe(X2, y2, 折2v)
    return r22, (float(np.mean(f0mae)) if f0mae else None), 覆盖, (float(np.mean(list(覆盖.values()))) if 覆盖 else None)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    清 = json.load(open(S0 / "manifest.json", encoding="utf-8"))["真值"]
    报 = dict(工具="判据_S0_1.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              预注册="meeting69/06_预注册_S0.1_F0与适应修正.md",
              刺激="47 个冻结 S0 合成刺激（产物/S0/）+ 8 条泛化声调（产物/S0.1/刺激_泛化/）",
              臂=dict(A0="log-mel 80/10ms（对照）", A2="耳蜗前端 v0.2（F0 v1 + 适应变体）"))

    # ── J6 先行：三变体对比 + 按 06 §4 选择（阈值不动） ──
    g适 = next(x for x in 清 if x["类"] == "适应")
    变体结果 = {}
    for v in 前.适应变体集:
        r = 取(g适["路径"], v)
        f = r["场"].astype(np.float64)
        早 = float(f[200:300].mean()); 晚 = float(f[1500:1800].mean())
        压 = r["场未适"].astype(np.float64)
        压稳 = float(压[200:2200].mean())
        变体结果[v] = dict(比=round(晚 / (早 + 1e-12), 4), 早=round(早, 6), 晚=round(晚, 6),
                           稳态残余比=round(晚 / (压稳 + 1e-12), 4), 阈值=0.5, 判=("PASS" if 晚 / (早 + 1e-12) <= 0.5 else "FAIL"))
    候选 = [v for v in ("V1", "V2", "V3") if 变体结果[v]["比"] <= 0.5]
    选中 = 候选[0] if 候选 else min(("V1", "V2", "V3"), key=lambda v: 变体结果[v]["比"])
    J6 = bool(变体结果[选中]["比"] <= 0.5)
    报["J6_适应"] = dict(细节=变体结果, 选中=选中,
                         选择规则="06 §4：仅比≤0.5 者入候选；多候选取 D2 亲和度 V1>V2>V3；无候选取最小比并判 FAIL",
                         判="PASS" if J6 else "FAIL")

    # ── 预处理：A0 mel（47 个冻结刺激） ──
    meld = {g["文件"]: mel(前.读wav(g["路径"])) for g in 清}

    # ── J1 VOT ──
    j1 = dict(A0=[], A2=[], sham=[])
    for g in 清:
        if g["类"] != "VOT":
            continue
        gt = (g["浊起秒"] - g["burst秒"]) * 1000.0
        b, v = VOT_A2(取(g["路径"], 选中)["场未适"])
        j1["A2"].append((v - b) * 1.0 - gt)
        b0, v0 = VOT_A0(meld[g["文件"]])
        j1["A0"].append(((v0 - b0) * 10.0 if v0 is not None else 1e3) - gt)
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
        r2 = 塞擦_A2(取(g["路径"], 选中)["事件"], g["burst秒"], g["摩擦起秒"])
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
        特征A2.append(擦特征_A2(取(g["路径"], 选中)["场未适"]))
        掩码.append(g["重复"] < 3)
    pred0, 类 = 最近质心(特征A0, 标签, np.array(掩码))
    pred2, _ = 最近质心(特征A2, 标签, np.array(掩码))
    acc = lambda p: float(np.mean([a == b for a, b in zip(np.array(p)[~np.array(掩码)], np.array(标签)[~np.array(掩码)])]))
    j3 = dict(A0=round(acc(pred0), 3), A2=round(acc(pred2), 3))
    J3 = j3["A2"] >= 0.75 and j3["A2"] >= j3["A0"] + 0.05
    报["J3_擦音"] = dict(准确率=j3, 判="PASS" if J3 else "FAIL")

    # ── J4 F0（冻结 S0 声调集；阈值 ≤5% 与 R²≥0.5 不动） ──
    声调 = [g for g in 清 if g["类"] == "声调"]
    r22, f0mae, 覆盖, 平均覆盖 = J4_评估(清, 声调, 选中)
    # A0 探针 R²（与 S0 同口径）
    特0 = []; 靶0 = []; 折0 = []
    for g in 声调:
        m = meld[g["文件"]]; n = len(m)
        gt = np.interp((np.arange(n) * 0.01) - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
        sel0 = np.arange(n) >= 31
        特0.append(m[sel0]); 靶0.append(gt[sel0]); 折0.append([g["标签"]] * int(sel0.sum()))
    X0 = np.concatenate(特0); y0 = np.concatenate(靶0)
    折0v = np.concatenate([np.array(f, dtype=object) for f in 折0])
    r20, _ = F0岭_probe(X0, y0, 折0v)
    j4 = dict(A0_R2=round(r20, 3), A2_R2=round(r22, 3),
              F0_MAE_rel=round(float(f0mae), 4) if f0mae is not None else None,
              浊帧覆盖=覆盖, 平均浊帧覆盖=round(平均覆盖, 3) if 平均覆盖 is not None else None,
              估计器="YIN 40ms/2.5ms 窗中心 θ=0.15 能量门5%+非周期门0.35（06 §3）")
    J4 = j4["A2_R2"] >= 0.5 and (j4["F0_MAE_rel"] is not None and j4["F0_MAE_rel"] <= 0.05)
    报["J4_F0"] = dict(细节=j4, 判="PASS" if J4 else "FAIL")

    # ── J4-G 泛化臂（8 条新声调；信息臂，不改 S1 门） ──
    泛 = json.load(open(S01 / "manifest_泛化.json", encoding="utf-8"))["真值"]
    每 = {}
    for g in 泛:
        r = 取(g["路径"], 选中)
        f0v = np.array(r["F0"]["f0"]); t = np.array(r["F0"]["t"])
        gt = np.interp(t - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
        ok = f0v > 0
        每[g["标签"]] = dict(相对MAE=round(float(np.mean(np.abs(f0v[ok] - gt[ok]) / gt[ok])), 4) if ok.sum() > 5 else None,
                            覆盖=round(float(ok.mean()), 3))
    mx = [v["相对MAE"] for v in 每.values() if v["相对MAE"] is not None]
    cov = [v["覆盖"] for v in 每.values()]
    j4g = dict(每条=每, 平均相对MAE=round(float(np.mean(mx)), 4) if mx else None,
               平均覆盖=round(float(np.mean(cov)), 3),
               阈值=0.05, 判=("PASS" if mx and float(np.mean(mx)) <= 0.05 else "FAIL"),
               效力="信息臂：不改 S1 门；J4 PASS 而本臂 FAIL ⇒ 估计器/合成器特异性标记")
    报["J4G_泛化"] = j4g

    # ── J5 边界 ──
    g = next(x for x in 清 if x["类"] == "边界")
    ev = 取(g["路径"], 选中)["事件"]
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

    # ── J7 确定性 / 实时率（真实 60 s 片段；warm 口径；选中变体） ──
    实 = Path(r"<path>\SpeakerA7代模型训练数据\(1).wav")
    if 实.exists():
        r1 = 前.分析(实, 变体=选中); r2 = 前.分析(实, 变体=选中)
        同 = r1["sha256"] == r2["sha256"]
        率 = min(r1["实时率"], r2["实时率"])
    else:
        同 = None; 率 = None
    J7 = (同 is True) and (率 is not None and 率 <= 0.5)
    报["J7_确定性"] = dict(两趟同=同, 实时率=率, 判="PASS" if J7 else "FAIL")

    # ── G1 对照汇总（A0 是否全面不差） ──
    g1 = dict(J1="A2 优" if J1 else "未过", J3="A2 优" if J3 else "未过",
              J5=("A0 中位更小（差 %.1f ms）" % (j5["A2_中位"] - j5["A0_中位"])),
              J6="A0 无适应（%s）" % 报["J6_适应"]["细节"]["V1"]["比"],
              J4=("A2 R² %.3f > A0 %.3f" % (j4["A2_R2"], j4["A0_R2"])))
    报["G1_对照汇总"] = g1

    # ── G4 静态审计：import 闭包无外部模型 ──
    禁 = ("torch", "transformers", "whisper", "wav2vec", "hubert", "speechbrain", "espnet",
          "funasr", "modelscope", "onnx", "tensorflow", "jax", "keras", "sentencepiece", "llama")
    命中 = []
    imports = []
    for fn in ("耳蜗前端_S0_1.py", "判据_S0_1.py", "合成_泛化声调_S0.1.py"):
        for line in (HERE / fn).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                imports.append(f"{fn}: {s}")
                low = s.lower()
                for b in 禁:
                    if b in low:
                        命中.append(f"{fn}: {s}")
    报["G4_审计"] = dict(命中=命中, import数=len(imports), imports=imports,
                        判="PASS" if not 命中 else "FAIL",
                        说明="只允许 numpy/scipy/librosa(A0 mel 对照)/标准库；C1 无 Transformer 应用；C10 无外挂模型")

    # ── 结论 ──
    主判 = dict(J1=bool(J1), J2=bool(J2), J3=bool(J3), J4=bool(J4), J5=bool(J5), J6=bool(J6), J7=bool(J7))
    过 = [k for k, v in 主判.items() if v]
    进S1 = all(主判[k] for k in ("J1", "J2", "J3", "J4", "J5", "J6", "J7"))
    报["判定"] = dict(主判=主判, 过=过, 结论=("PASS" if len(过) == 7 else ("部分" if 过 else "FAIL")),
                      进S1=bool(进S1), 选中适应变体=选中,
                      说明="S0.1 只修 F0 估计器与适应实现；判据阈值与判死条件未改（06）；不测可懂/不测像人")
    (S01 / "结果.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
