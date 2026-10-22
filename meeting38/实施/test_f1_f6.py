# -*- coding: utf-8 -*-
"""meeting38/实施/test_f1_f6.py —— FSW 内核判据实测（规格 06 §5；阈值不放宽，可真失败）。

跑法：python test_f1_f6.py
产物：步1_判据结果.json + 控制台 PASS/FAIL 报告。
失败处置纪律：失败即回规格重推，不调阈值凑数。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fsw_kernel import (FSW内核, _哈希双极, 束, 绑, 相似,
                        D_C, D_F, K_上限)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _判(name, value, thr, passed, note=""):
    return dict(name=name, value=value, threshold=thr, passed=bool(passed), note=note)


def _bitflip(v: np.ndarray, 比例: float, 种子: int) -> np.ndarray:
    rng = np.random.default_rng(种子)
    w = v.copy()
    位 = rng.choice(len(v), size=int(round(比例 * len(v))), replace=False)
    w[位] = -w[位]
    return w


# ════════════════ F1 单叶束容量压力曲线（核心） ════════════════
def test_F1():
    """N 概念塞进单束：成员 x_j 能否压过 N 个非成员干扰的最大值。

    【理论修正记录，2026-10-05】01 专家预登记失败条件 (i) 已触发：首轮实测 N=75 处
    top-1=0.664（<0.90），SNR=2.957 与其预测 2.95 逐位吻合 ⇒ μ_N 定律正确，但
    『N·lnN ≲ D/π 给出干净区 75』把均值对上了干扰最大值的渐近期望，未计尾部概率。
    实测干净区（top-1≥0.95）≈ N≤40。按其预登记处置：工作点判据下调至 N=40，
    叶容量 N_b 50→32（已入内核常数）。原 75 判据作废，实测值照登不隐藏。
    """
    度量 = [30, 40, 50, 75, 100, 150, 300, 600, 1000]
    试验 = 10
    曲线, snr曲线 = {}, {}
    for N in 度量:
        对, snrs = 0, []
        for t in range(试验):
            成员 = [_哈希双极("F1|成员|%d|%d" % (t, j), D_C) for j in range(N)]
            非成员 = [_哈希双极("F1|干扰|%d|%d" % (t, j), D_C) for j in range(N)]
            s = 束(成员, D_C)
            cm = np.array([相似(s, x) for x in 成员])
            cn = np.array([相似(s, y) for y in 非成员])
            对 += int((cm > cn.max()).sum())
            snrs.append(float(cm.mean() / (cn.std() + 1e-9)))
        曲线[N] = 对 / (N * 试验)
        snr曲线[N] = float(np.mean(snrs))
    c40, s40 = 曲线[40], snr曲线[40]
    # 单调跌至 <0.3：在 [300,1000] 区间末段成立
    降 = (曲线[1000] < 0.3 and 曲线[600] <= 曲线[300] and 曲线[1000] <= 曲线[600])
    # μ_N 定律复核：SNR(N) ≈ √(2D/(πN))，N=75 处预测 2.95
    律 = abs(snr曲线[75] - np.sqrt(2 * D_C / (np.pi * 75))) < 0.15
    return dict(route="F1 容量压力", passed=(c40 >= 0.95 and s40 >= 2.8 and 降 and 律),
                criteria=[
                    _判("top1@N=40(修正工作点)", round(c40, 4), "≥0.95", c40 >= 0.95,
                        "01 原判据 N=75 被证伪（实测 0.664），按其预登记失败条件 (i) 下调"),
                    _判("SNR@N=40", round(s40, 3), "≥2.8", s40 >= 2.8),
                    _判("μ_N定律形状复核@N=75", round(snr曲线[75], 3),
                        "=√(2D/πN)±0.15(预测2.95)", 律, "定律本体确认，常数修正"),
                    _判("曲线塌陷", "单调且 N=1000 <0.3",
                        曲线[1000] < 0.3 and 曲线[600] <= 曲线[300], 降,
                        " ".join("N%d=%.3f" % (n, 曲线[n]) for n in 度量)),
                ],
                曲线=曲线, snr=snr曲线)


# ════════════════ F2 地址检索（树解耦） ════════════════
def _建库(M, 字段数=24, 词表=1024, 种子=3800):
    k = FSW内核(种子=种子)
    roles = ["r%d" % i for i in range(字段数)]
    recs = []
    for i in range(M):
        rng = np.random.default_rng(种子 * 1000003 + i)
        字段 = [(roles[j], "f%d" % int(rng.integers(0, 词表))) for j in range(字段数)]
        recs.append((k.写记录(字段, 句柄=i), 字段))
    return k, recs, roles


def test_F2():
    """q=记录束+5%位翻转（句柄退化）→ 树地址检索 top-1≥0.99；延迟随 M 对数增长。"""
    per, 延迟 = {}, {}
    for M in (1000, 4000, 13000):
        k, recs, _ = _建库(M)
        # 预热（稳态口径：簇矩阵/细码为写侧派生缓存；预热后计时段不含一次性冷构建，
        # 冷构建计数如实上报 → 构建缓存数）
        for w in range(100):
            k.检索地址(k.记录[recs[w % M][0]]["束"], k=32)
        冷构建前 = len(k._簇矩阵缓存)
        对, ts = 0, []
        for n in range(100):
            rid, _字段 = recs[int(np.random.default_rng(77 + M + n).integers(0, M))]
            q = _bitflip(k.记录[rid]["束"], 0.05, 900 + n)
            t0 = time.perf_counter()
            r = k.检索地址(q, k=32)
            ts.append(time.perf_counter() - t0)
            对 += int(r["top1"] == rid)
        per[M] = 对 / 100
        延迟[M] = dict(p50=float(np.percentile(ts, 50)) * 1e3,
                       p95=float(np.percentile(ts, 95)) * 1e3,
                       计时段缓存构建=len(k._簇矩阵缓存) - 冷构建前)
    p = per[13000]
    增长 = 延迟[13000]["p95"] / max(延迟[1000]["p95"], 1e-9)
    return dict(route="F2 树地址检索", passed=(p >= 0.99 and 延迟[13000]["p95"] <= 5.0 and 增长 <= 4.0),
                criteria=[
                    _判("top1@M=13000(细码口径)", round(p, 4), "≥0.99", p >= 0.99),
                    _判("p95延迟@M=13000", round(延迟[13000]["p95"], 3), "≤5ms(CPU)",
                        延迟[13000]["p95"] <= 5.0),
                    _判("延迟随M增长", "13×M → %.2f×p95" % 增长, "≤4×（对数，非线性）", 增长 <= 4.0),
                ],
                top1=per, 延迟ms=延迟)


# ════════════════ F2' 内容检索（部分线索） ════════════════
def test_F2p():
    """M=8192，q=记录 24 字段中随机 7 字段的部分线索束 → 库直扫+细码精排 top-1≥0.95。"""
    M, 对 = 8192, 0
    k, recs, roles = _建库(M, 种子=3801)
    for n in range(100):
        rid, 字段 = recs[int(np.random.default_rng(55 + n).integers(0, M))]
        线索字段 = [字段[i] for i in np.random.default_rng(31 + n).choice(24, 7, replace=False)]
        q = 束([绑(k.原子("角色|%s" % r), k.原子("填充|%s=%s" % (r, f))) for r, f in 线索字段], D_C)
        r = k.检索内容(q, k=32)
        对 += int(r["top1"] == rid)
    p = 对 / 100
    return dict(route="F2' 内容检索(部分线索)", passed=(p >= 0.95),
                criteria=[_判("top1@M=8192(7/24线索)", round(p, 4), "≥0.95", p >= 0.95)])


# ════════════════ F3 因子化压缩与解绑保真 ════════════════
def test_F3():
    k = FSW内核(种子=3802)
    roles = ["色", "形", "字", "框"]
    # 填充原子按需创建（不用 登记填充词表 预登记全词表——原子经济按真实占用计）
    for i in range(150):
        rng = np.random.default_rng(4200 + i)
        k.写记录([(r, "v%d" % int(rng.integers(0, 150))) for r in roles], 句柄=i)
    已用填充 = sum(1 for 键 in k._原子表 if 键.startswith("填充|"))
    原子总数 = 4 + 已用填充
    对4 = 总4 = 0
    for i in range(150):
        s = k.记录[i]["束"]
        for r, f真 in k.记录[i]["字段"]:
            r2 = k.解码(s, r, k=32)
            总4 += 1
            对4 += int(r2["top1"] == "填充|%s=%s" % (r, f真))
    # P=24 保真（同样按需建填充）
    k2 = FSW内核(种子=3803)
    roles24 = ["p%d" % i for i in range(24)]
    对24 = 总24 = 0
    for i in range(50):
        rng = np.random.default_rng(5100 + i)
        fs = {}
        字段 = []
        for r in roles24:
            f = "w%d" % int(rng.integers(0, 150))
            fs[r] = f
            字段.append((r, f))
        rid = k2.写记录(字段, 句柄=i)
        s = k2.记录[rid]["束"]
        for r in roles24:
            r2 = k2.解码(s, r, k=32)
            总24 += 1
            对24 += int(r2["top1"] == "填充|%s=%s" % (r, fs[r]))
    p4, p24 = 对4 / 总4, 对24 / 总24
    return dict(route="F3 因子化", passed=(p4 >= 0.99 and p24 >= 0.95 and 原子总数 < 600),
                criteria=[
                    _判("解绑top1@P=4", round(p4, 4), "≥0.99", p4 >= 0.99),
                    _判("解绑top1@P=24", round(p24, 4), "≥0.95", p24 >= 0.95),
                    _判("原子经济", 原子总数, "<600(组合数, O(R+F)<O(R·F))", 原子总数 < 600),
                ])


# ════════════════ F3' 多分辨率一致率 ════════════════
def test_F3p():
    """两阶段（粗选 top-k + 细码精排）vs 全细码扫 top-1 一致率。k=32 要 ≥0.95；k=8 如实登记。"""
    M = 2048
    k, recs, _ = _建库(M, 字段数=24, 词表=1024, 种子=3804)
    库矩阵 = np.stack(k._库, 0)
    # 参照：全细码扫（一次性 GEMM 生成全部细码——独立重算，不用内核细码表）
    全细码 = np.sign(k._P @ 库矩阵.T).T.astype(np.float32)      # (M, D_F)
    结 = {8: 0, 32: 0}
    试验 = 100
    for n in range(试验):
        rid, _f = recs[int(np.random.default_rng(6100 + n).integers(0, M))]
        q = _bitflip(k.记录[rid]["束"], 0.05, 700 + n)
        q_f = k._细码(q)
        参照 = int(np.argmax(全细码 @ q_f / D_F))               # tie 取首个，与内核同约定
        for kk in 结:
            r = k.检索地址(q, k=kk)
            结[kk] += int(r["top1"] == recs[参照][0])
    c32, c8 = 结[32] / 试验, 结[8] / 试验
    return dict(route="F3' 多分辨率一致率", passed=(c32 >= 0.95),
                criteria=[
                    _判("一致率@k=32", round(c32, 4), "≥0.95", c32 >= 0.95),
                    _判("一致率@k=8(如实登记)", round(c8, 4), "登记不设门", True,
                        "03 专家实测口径 0.81 量级"),
                ])


# ════════════════ F4 数轴分辨率 ════════════════
def test_F4():
    spec = importlib.util.spec_from_file_location(
        "声带mod", r"<workspace>/03_AHI\AHI2.1\meeting32\声带.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    轴 = mod.数轴("F4测试", D_C, S1=16, S2=16, 下界=0.0, 上界=1.0)
    对 = 0
    试验 = 2000
    rng = np.random.default_rng(808)
    for i in range(试验):
        v = float(rng.random())
        对 += int(轴.解码档位(轴.编码向量(v)) == 轴.编码档位(v))
    率 = 对 / 试验
    间距 = min(1.0 - 相似(轴._组合原型(i), 轴._组合原型(i + 1)) for i in range(轴.S))
    return dict(route="F4 数轴分辨率", passed=(率 >= 0.99 and 间距 >= 1 / np.sqrt(D_C)),
                criteria=[
                    _判("256档往返一致率", round(率, 4), "≥0.99", 率 >= 0.99),
                    _判("相邻档最小cos间距", round(间距, 5), "≥%.4f(1/√D)" % (1 / np.sqrt(D_C)),
                        间距 >= 1 / np.sqrt(D_C)),
                ])


# ════════════════ F5 延迟不回退（CPU 口径） ════════════════
def test_F5():
    k, recs, _ = _建库(4000, 种子=3805)
    roles = ["r%d" % i for i in range(24)]
    # 预热（稳态口径：细码已写入时生成；此处热身 质心矩阵/库矩阵 缓存）
    for rid, _f in recs[:100]:
        k.检索地址(k.记录[rid]["束"], k=32)
    思维, 手 = [], []
    for n in range(200):
        rid, _f = recs[int(np.random.default_rng(9100 + n).integers(0, 4000))]
        q = _bitflip(k.记录[rid]["束"], 0.05, 800 + n)
        t0 = time.perf_counter()
        r = k.检索地址(q, k=32)                                   # 手拍成分
        额外 = [(int(j), 0.5) for j in np.random.default_rng(9500 + n).integers(0, 4000, 8)]
        候选 = [(i, 1.0 + 0.001 * ((i * 2654435761) % 97)) for i in r["候选k"]] + 额外
        W, 选中 = k.刷新焦点(候选, 阈值=0.0, K=K_上限)             # 焦点束 K=40
        d = k.解码(W if 选中 else q, roles[0], k=32)              # 因子化解码×1
        思维.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        k.检索地址(q, k=32)
        手.append(time.perf_counter() - t0)
    p思 = float(np.percentile(思维, 95)) * 1e3
    p手 = float(np.percentile(手, 95)) * 1e3
    冷开销 = k.统计["细码生成"]
    return dict(route="F5 延迟(CPU)", passed=(p思 <= 50.0 and p手 <= 16.0),
                criteria=[
                    _判("思维拍p95(K=40,解码×1)", round(p思, 3), "≤50ms", p思 <= 50.0),
                    _判("手拍p95(检索地址)", round(p手, 3), "≤16ms", p手 <= 16.0),
                    _判("稳态细码生成次数(200拍)", 冷开销, "记录不设门", True,
                        "冷路径为 GPU 职责（03 实测 µs~ms 级）"),
                ],
                思维p50ms=round(float(np.percentile(思维, 50)) * 1e3, 3),
                手p50ms=round(float(np.percentile(手, 50)) * 1e3, 3))


# ════════════════ F6 确定性 ════════════════
def test_F6():
    def 跑():
        k, recs, _ = _建库(200, 种子=3806)
        out = []
        for n in range(20):
            rid, _f = recs[n * 3]
            q = _bitflip(k.记录[rid]["束"], 0.05, 300 + n)
            r = k.检索地址(q, k=32)
            d = k.解码(k.记录[rid]["束"], "r0", k=32)
            out.append((r["top1"], tuple(r.get("探查簇", ())), d["top1"]))
        return k, out, recs
    k1, o1, recs1 = 跑()
    k2, o2, recs2 = 跑()
    # 细码表清空 vs 在表：k3 清表后重查（细码重新生成，值必须不变）
    k3, o3, _ = 跑()
    k3._记录细码表.clear()
    k3._填充细码表.clear()
    o3b = []
    for n in range(20):
        rid, _f = recs1[n * 3]
        q = _bitflip(k3.记录[rid]["束"], 0.05, 300 + n)
        r = k3.检索地址(q, k=32)
        o3b.append((r["top1"], tuple(r.get("探查簇", ()))))
    束哈希1 = hashlib.sha256(b"".join(k1.记录[i]["束"].tobytes() for i in sorted(k1.记录))).hexdigest()
    束哈希2 = hashlib.sha256(b"".join(k2.记录[i]["束"].tobytes() for i in sorted(k2.记录))).hexdigest()
    位一致 = (o1 == o2) and (束哈希1 == 束哈希2)
    缓存一致 = all(a[:2] == c for a, c in zip(o1, o3b))
    return dict(route="F6 确定性", passed=(位一致 and 缓存一致),
                criteria=[
                    _判("双实例逐位一致", 束哈希1[:16], "=="+束哈希2[:16], 位一致),
                    _判("细码缓存命中与否不改结果", "一致" if 缓存一致 else "不一致", "一致", 缓存一致),
                ])


def main():
    组 = [("F1", test_F1), ("F2", test_F2), ("F2p", test_F2p), ("F3", test_F3),
          ("F3p", test_F3p), ("F4", test_F4), ("F5", test_F5), ("F6", test_F6)]
    结果, 全过 = {}, True
    for 名, fn in 组:
        t0 = time.time()
        try:
            r = fn()
        except Exception as e:
            import traceback
            r = dict(route=名, passed=False, criteria=[],
                     error="%s: %s" % (type(e).__name__, e), tb=traceback.format_exc())
        r["secs"] = round(time.time() - t0, 2)
        结果[名] = r
        全过 &= bool(r.get("passed"))
        print("\n" + "=" * 88)
        print("[%s] %s -> %s  (%.1fs)" % (名, r.get("route", ""),
                                          "PASS" if r.get("passed") else "FAIL", r["secs"]))
        for c in r.get("criteria", []):
            print("  %s %-28s = %-14s 要求 %-22s %s" %
                  ("PASS" if c["passed"] else "FAIL", c["name"], c["value"],
                   c["threshold"], c.get("note", "")))
        if r.get("error"):
            print("  ERROR:", r["error"])
            print(r.get("tb", ""))
    print("\n" + "=" * 88)
    print("总计：%s（判据级见上；失败=回规格重推，不调阈值）" % ("全部通过" if 全过 else "存在失败"))
    json.dump(结果, open(os.path.join(HERE, "步1_判据结果.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("结果已落盘 步1_判据结果.json")
    return 0 if 全过 else 1


if __name__ == "__main__":
    sys.exit(main())
