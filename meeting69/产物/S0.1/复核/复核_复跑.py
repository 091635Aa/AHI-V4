# -*- coding: utf-8 -*-
"""独立复核复跑（不覆盖正式读数）：S0.1 判据 + S0 判据 + 附加探针。
只写 产物/S0.1/复核/ 下的文件。"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

M69 = Path(r"<workspace>/03_AHI\AHI4\meeting69")
实施 = M69 / "实施"
复核 = M69 / "产物" / "S0.1" / "复核"
复核.mkdir(parents=True, exist_ok=True)


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def 递归比(a, b, path="", 忽略=("时间",), 出=None):
    if 出 is None:
        出 = []
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a), set(b)
        for k in sorted(ka - kb):
            出.append(dict(路径=f"{path}/{k}", 类型="仅正式有", 正式=a[k], 复跑=None))
        for k in sorted(kb - ka):
            出.append(dict(路径=f"{path}/{k}", 类型="仅复跑有", 正式=None, 复跑=b[k]))
        for k in sorted(ka & kb):
            if k in 忽略:
                continue
            递归比(a[k], b[k], f"{path}/{k}", 忽略, 出)
        return 出
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            出.append(dict(路径=path + "/len", 类型="长度不同", 正式=len(a), 复跑=len(b)))
            return 出
        for i, (x, y) in enumerate(zip(a, b)):
            递归比(x, y, f"{path}[{i}]", 忽略, 出)
        return 出
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if float(a) != float(b):
            出.append(dict(路径=path, 类型="数值不同", 正式=a, 复跑=b, 差=round(float(b) - float(a), 9)))
        return 出
    if a != b:
        出.append(dict(路径=path, 类型="值不同", 正式=a, 复跑=b))
    return 出


结果 = {}

# ── 1) S0.1 判据复跑（S01 → 复核目录；不动正式 结果.json） ──
t1 = 复核 / "S0.1_复跑"
t1.mkdir(exist_ok=True)
shutil.copy2(M69 / "产物" / "S0.1" / "manifest_泛化.json", t1 / "manifest_泛化.json")
m1 = 载("判据_S0_1", 实施 / "判据_S0_1.py")
m1.S01 = t1
print("[复核] 跑 判据_S0_1 →", t1, flush=True)
m1.主()
正式1 = json.loads((M69 / "产物" / "S0.1" / "结果.json").read_text(encoding="utf-8"))
复跑1 = json.loads((t1 / "结果.json").read_text(encoding="utf-8"))
结果["S0.1复跑差异"] = 递归比(正式1, 复跑1)
结果["S0.1复跑差异数"] = len(结果["S0.1复跑差异"])
print("[复核] S0.1 差异数 =", len(结果["S0.1复跑差异"]), flush=True)

# ── 2) S0 判据复跑（S0 → 复核目录；不覆盖 产物/S0/结果.json） ──
t0 = 复核 / "S0_复跑"
t0.mkdir(exist_ok=True)
shutil.copy2(M69 / "产物" / "S0" / "manifest.json", t0 / "manifest.json")
m0 = 载("判据_S0", 实施 / "判据_S0.py")
m0.S0 = t0
print("[复核] 跑 判据_S0 →", t0, flush=True)
m0.主()
正式0 = json.loads((M69 / "产物" / "S0" / "结果.json").read_text(encoding="utf-8"))
复跑0 = json.loads((t0 / "结果.json").read_text(encoding="utf-8"))
结果["S0复跑差异"] = 递归比(正式0, 复跑0)
结果["S0复跑差异数"] = len(结果["S0复跑差异"])
print("[复核] S0 差异数 =", len(结果["S0复跑差异"]), flush=True)

# ── 3) 前端内部一致性：S0 与 S0.1 的 场未适 是否逐位相同（决定 J1/J2/J3/J5 不变性） ──
前0 = 载("耳蜗前端_S0", 实施 / "耳蜗前端_S0.py")
前1 = 载("耳蜗前端_S0_1x", 实施 / "耳蜗前端_S0_1.py")
清 = json.loads((M69 / "产物" / "S0" / "manifest.json").read_text(encoding="utf-8"))["真值"]
同未 = True
同场V1 = True
最大差未 = 0.0
for g in 清:
    a = 前0.分析(g["路径"])
    b = 前1.分析(g["路径"], 变体="V1")
    if not np.array_equal(a["场未适"], b["场未适"]):
        同未 = False
        最大差未 = max(最大差未, float(np.abs(a["场未适"] - b["场未适"]).max()))
    if not np.array_equal(a["场"], b["场"]):
        同场V1 = False
结果["前端一致性"] = dict(场未适逐位同=bool(同未), 场未适最大差=最大差未,
                        S0场与S0_1_V1场逐位同=bool(同场V1),
                        说明="场未适=压缩.T（不经适应），故与变体无关；V1 组合规则=S0 现状")

# ── 4) 确定性：同一刺激两次 分析（sha256 覆盖 场；另比 F0/事件） ──
声 = next(x for x in 清 if x["类"] == "声调")
r1 = 前1.分析(声["路径"], 变体="V3")
r2 = 前1.分析(声["路径"], 变体="V3")
结果["确定性"] = dict(sha256_1=r1["sha256"], sha256_2=r2["sha256"], sha256同=r1["sha256"] == r2["sha256"],
                   场逐位同=bool(np.array_equal(r1["场"], r2["场"])),
                   F0逐位同=bool(r1["F0"]["f0"] == r2["F0"]["f0"] and r1["F0"]["t"] == r2["F0"]["t"]),
                   事件同=bool(r1["事件"] == r2["事件"]),
                   注="sha256 只覆盖『场』；F0/事件另比")

# ── 5) V3 稳态场是否趋零（J6 是否「虚假通过」的实质检验） ──
g适 = next(x for x in 清 if x["类"] == "适应")
rv = {v: 前1.分析(g适["路径"], 变体=v) for v in ("V1", "V2", "V3")}
统 = {}
for v, r in rv.items():
    f = r["场"].astype(np.float64); c = r["场未适"].astype(np.float64)
    晚 = f[1500:1800]; 早 = f[200:300]
    统[v] = dict(全局均值=round(float(f.mean()), 6), 未适全局均值=round(float(c.mean()), 6),
                整体残余比=round(float(f.mean() / (c.mean() + 1e-12)), 6),
                晚段零值占比=round(float((晚 <= 1e-6).mean()), 4),
                晚段非零中位=round(float(np.median(晚[晚 > 1e-6])) if (晚 > 1e-6).any() else 0.0, 6),
                早段均值=round(float(早.mean()), 6), 晚段均值=round(float(晚.mean()), 6),
                比=round(float(晚.mean() / (早.mean() + 1e-12)), 4),
                晚段与未适逐带相关=round(float(np.corrcoef(晚.mean(axis=0), c[1500:1800].mean(axis=0))[0, 1]), 4))
结果["V3饱和检验"] = 统

# ── 6) 包络实现审计：1 ms 低通是沿时间还是沿频带？ ──
α = 1.0 / (前1.SR * 0.001)
测 = np.arange(2 * 6, dtype=np.float64).reshape(2, 6) + 1.0     # (2波段, 6样本)
包测 = 前1.包络(测)
沿带 = np.empty_like(测); acc = 0.0
for i in range(测.shape[0]):
    acc = acc + (np.maximum(测, 0)[i] - acc) * α
    沿带[i] = acc
沿时 = np.empty_like(测); acc = 0.0
for k in range(测.shape[1]):
    acc = acc + (np.maximum(测, 0)[:, k] - acc) * α
    沿时[:, k] = acc
结果["包络实现"] = dict(
    实测样例=包测.tolist(),
    沿频带递推=沿带.tolist(), 沿时间递推=沿时.tolist(),
    与沿频带逐位同=bool(np.array_equal(包测, 沿带)),
    与沿时间逐位同=bool(np.array_equal(包测, 沿时)),
    α=α, 说明="包络(样本) 的 len(x) 是频带数 64，故 acc 沿轴0（频带）递推，时间轴无低通")

# ── 7) 该缺陷的频率后果：1 kHz 纯音在 64 带上的扩散（场未适，按带均值） ──
n = int(0.2 * 前1.SR); t = np.arange(n) / 前1.SR
音 = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
样本, fs = 前1.滤波组(音)
包实现 = 前1.包络(样本)
from scipy.signal import lfilter
包正确 = lfilter([α], [1.0, -(1.0 - α)], np.maximum(样本, 0.0), axis=1)


def 带廓(包):
    n1 = len(音) // 前1.包络步
    b1 = 包[:, : n1 * 前1.包络步].reshape(包.shape[0], n1, 前1.包络步).mean(axis=2)
    c = np.power(np.maximum(b1, 0), 0.4)
    return c[:, 250:420].mean(axis=1)   # 稳态段


廓实 = 带廓(包实现); 廓正 = 带廓(包正确)
结果["纯音带扩散"] = dict(
    实现_带廓=[round(float(v), 5) for v in 廓实],
    正确_带廓=[round(float(v), 5) for v in 廓正],
    实现_超10pct带数=int((廓实 > 0.1 * 廓实.max()).sum()),
    正确_超10pct带数=int((廓正 > 0.1 * 廓正.max()).sum()),
    实现_最大带=int(np.argmax(廓实)), 正确_最大带=int(np.argmax(廓正)),
    实现_峰值带中心Hz=round(float(fs[int(np.argmax(廓实))]), 1),
    正确_峰值带中心Hz=round(float(fs[int(np.argmax(廓正))]), 1),
    说明="64 带 ERB；正确实现下 1 kHz 纯音应只点亮少数带；实现下若多带同时高响应即为跨带泄漏")

# ── 8) J4 的 GT 时间轴敏感性（manifest 轮廓 220 点实际跨 0.4966 s，却按 linspace(0,0.5) 映射） ──
敏 = {}
for g in [x for x in 清 if x["类"] == "声调"]:
    r = 前1.分析(g["路径"], 变体="V3")
    f0v = np.array(r["F0"]["f0"]); tt = np.array(r["F0"]["t"])
    ok = f0v > 0
    gt旧 = np.interp(tt - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
    # 轮廓 220 点由 f0[::50] 取得，实际末点时刻 = 50*(220-1)/SR
    实际末 = 50 * (len(g["f0轮廓"]) - 1) / 前1.SR
    gt新 = np.interp(tt - 0.3, np.linspace(0, 实际末, len(g["f0轮廓"])), g["f0轮廓"])
    敏[g["标签"]] = dict(旧口径MAE=round(float(np.mean(np.abs(f0v[ok] - gt旧[ok]) / gt旧[ok])), 5),
                        校正时间轴MAE=round(float(np.mean(np.abs(f0v[ok] - gt新[ok]) / gt新[ok])), 5),
                        浊帧=int(ok.sum()), 帧数=int(len(f0v)))
结果["J4_GT轴敏感性"] = dict(逐条=敏,
                        旧口径均值=round(float(np.mean([v["旧口径MAE"] for v in 敏.values()])), 5),
                        校正轴均值=round(float(np.mean([v["校正时间轴MAE"] for v in 敏.values()])), 5),
                        说明="若校正后明显更小，则 0.17% 中含系统性的 GT 插值偏置")

# ── 9) J4 逐条明细复算（与 明细_J4逐条.json 对照；并核「是否丢难帧」） ──
明 = json.loads((M69 / "产物" / "S0.1" / "明细_J4逐条.json").read_text(encoding="utf-8"))
逐 = {}
浊实 = 0; 浊旧 = 0
for g in [x for x in 清 if x["类"] == "声调"]:
    r = 前1.分析(g["路径"], 变体="V3")
    f0v = np.array(r["F0"]["f0"]); tt = np.array(r["F0"]["t"])
    gt = np.interp(tt - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
    ok = f0v > 0
    e = np.abs(f0v[ok] - gt[ok]) / gt[ok]
    逐[g["标签"]] = dict(MAE_rel=round(float(e.mean()), 5), p90=round(float(np.percentile(e, 90)), 5),
                        max=round(float(e.max()), 5), 浊帧=int(ok.sum()), 帧数=int(len(f0v)),
                        覆盖率=round(float(ok.mean()), 3))
    浊实 += int(ok.sum())
    # S0 口径浊帧（25 ms 窗）
    r0 = 前0.分析(g["路径"])
    浊旧 += int((np.array(r0["F0"]["f0"]) > 0).sum())
结果["J4逐条复算"] = dict(复算=逐, 明细=明,
                      浊帧合计_S01=int(浊实), 浊帧合计_S0=int(浊旧),
                      说明="覆盖率为比值；分母帧数随窗长变化（305 vs 311），须同时看浊帧绝对数")

# ── 10) 泛化刺激确定性：重跑到临时目录后与现刺激逐字节比 ──
合 = 载("合成_泛化声调_S0.1", 实施 / "合成_泛化声调_S0.1.py")
tmp = 复核 / "泛化重跑"
(tmp / "产物" / "S0.1").mkdir(parents=True, exist_ok=True)
合.M69 = tmp
合.出 = tmp / "产物" / "S0.1" / "刺激_泛化"
合.主()
同w = {}
for p in sorted((M69 / "产物" / "S0.1" / "刺激_泛化").glob("*.wav")):
    q = 合.出 / p.name
    同w[p.name] = dict(存在=q.exists(), 逐字节同=bool(q.exists() and q.read_bytes() == p.read_bytes()))
结果["泛化刺激确定性"] = 同w

(复核 / "复核_复跑.json").write_text(json.dumps(结果, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: (v if not isinstance(v, list) else f"{len(v)} 项") for k, v in 结果.items()},
                 ensure_ascii=False, indent=1))
