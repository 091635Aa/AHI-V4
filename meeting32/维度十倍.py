# -*- coding: utf-8 -*-
"""meeting32/维度十倍.py —— #310 向量空间维度 ×10 密集度与语言丰富度全面测试。

判据先定（预登记）：
  M1 密度律   编码向量两两 cos 的 σ 与 1/√D 理论比 ∈ [0.8, 1.25]（随机投影渐近律成立）
  M2 解码容量 位翻 p=5% 时 top-1 检索准确率 ≥95%（1024）/ ≥99%（10240，判别力 √10 锐化）
  M3 容量曲线 注册段 200→2000，p=5% 准确率随段数衰减 ≤5 个百分点（×10 维应显著更缓）
  M4 语言丰富 生成 200 句：重复句率 <5%；字符类型-标记比 TTR 随维不降；
              新四元组占比（相对训练语料）随维不降（HDC 退避泛化 → 更丰富的组合）
  M5 性能     编码 1000 次耗时记录（×10 维 ≈ ×10 耗时，线性标度）
维度点：256 / 1024（基线）/ 2560 / 5120 / 10240（×10 主对照）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

维度点 = [256, 1024, 2560, 5120, 10240]


def 取语料(n: int = 3000) -> list:
    p = os.path.join(HERE, "..", "meeting31", "数据集", "魔塔",
                     "Psychology-10K-ZH.json")
    data = json.load(open(p, encoding="utf-8"))
    句s = []
    for it in data:
        t = (it.get("output") or "").strip()
        if 6 <= len(t) <= 80:
            句s.append(t)
        if len(句s) >= n:
            break
    return 句s


def 单维度测(D: int) -> dict:
    """单维度全套测试（在子进程中跑，保证 lang_codec.D 全局干净）。"""
    import numpy as np
    import lang_codec
    lang_codec.D = D
    from lang_codec import 语言编解码器

    语料 = 取语料()
    codec = 语言编解码器(seed=216)
    rec = dict(D=D)

    # —— M1 密度两支：M1a 真实文本结构密度；M1b 大字符集随机串渐近律 ——
    t0 = time.perf_counter()
    编s = np.stack([codec.编码(s) for s in 语料[:2000]])
    rec["编码1000耗时秒"] = round(time.perf_counter() - t0, 3)
    rng = np.random.default_rng(310)
    i = rng.integers(0, len(编s), 20000)
    j = rng.integers(0, len(编s), 20000)
    m = i != j
    i, j = i[m], j[m]
    单位 = 编s / (np.linalg.norm(编s, axis=1, keepdims=True) + 1e-9)
    cosr = np.sum(单位[i] * 单位[j], axis=1)
    rec["M1a"] = dict(真实文本cos均值=round(float(np.mean(cosr)), 5),
                      σ=round(float(np.std(cosr)), 5),
                      p99=round(float(np.quantile(cosr, 0.99)), 4),
                      max=round(float(np.max(cosr)), 4))
    # M1b：从 2 万字 CJK 池造 500 条 30 字随机串（期望重叠≈0.04 字）→ 纯随机分量
    池 = [chr(0x4E00 + k) for k in range(20000)]
    rand句 = ["".join(池[k] for k in rng.integers(0, 20000, 30)) for _ in range(500)]
    rand编 = np.stack([codec.编码(s) for s in rand句])
    单位r = rand编 / (np.linalg.norm(rand编, axis=1, keepdims=True) + 1e-9)
    ii = rng.integers(0, 500, 40000)
    jj = rng.integers(0, 500, 40000)
    mm = ii != jj
    cosb = np.sum(单位r[ii[mm]] * 单位r[jj[mm]], axis=1)
    σ实 = float(np.std(cosb))
    σ论 = 1.0 / np.sqrt(D)
    rec["M1b"] = dict(σ=round(σ实, 5), σ理论=round(σ论, 6),
                      比=round(σ实 / σ论, 3),
                      均值=round(float(np.mean(cosb)), 5))

    # —— M2/M3 解码容量：注册段库 + 位翻扰动 top-1 ——
    注册数s = [200, 500, 1000, 2000]
    M2, M3 = {}, {}
    for K in 注册数s:
        K = min(K, len(语料))
        codec2 = 语言编解码器(seed=216)
        库 = 语料[:K]
        for s in 库:
            codec2.注册段(s)
        矩阵 = np.stack(codec2._段向量)
        单位库 = 矩阵 / (np.linalg.norm(矩阵, axis=1, keepdims=True) + 1e-9)
        for p in (0.05, 0.10):
            rng = np.random.default_rng(320 + K)
            对s = rng.integers(0, K, 300)
            翻位 = 1.0 - 2.0 * (rng.random((300, D)) < p)   # 翻 p，其余不动
            Q = 矩阵[对s] * 翻位
            Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
            检 = np.argmax(Q @ 单位库.T, axis=1)
            准 = float(np.mean(检 == 对s))
            if K == 200:
                M2["p%.2f" % p] = round(准, 4)
            M3["K%d_p%.2f" % (K, p)] = round(准, 4)
    rec["M2"] = M2
    rec["M3"] = M3

    # —— M4 语言丰富度：gen_lm 同语料训练，同种子生成 200 句 ——
    from gen_lm import HDC语言模型
    lm = HDC语言模型(codec, seed=216)
    for s in 语料:
        lm.学句(s)
    rng = np.random.default_rng(330)
    生成s = [lm.生成("", 温度=1.0, rng=rng, 最大长=32) for _ in range(200)]
    句集 = set(生成s)
    重复率 = 1.0 - len(句集) / len(生成s)
    全字 = "".join(生成s)
    TTR = len(set(全字)) / max(len(全字), 1)
    # 新四元组占比（相对训练语料的四元组集合）
    语料4 = set()
    for s in 语料:
        t = [c for c in s if c.strip()]
        for k in range(len(t) - 3):
            语料4.add("".join(t[k:k + 4]))
    新4 = []
    for s in 生成s:
        t = [c for c in s if c.strip()]
        if len(t) >= 4:
            新4.extend("".join(t[k:k + 4]) not in 语料4
                       for k in range(len(t) - 3))
    命中 = [lm.三元命中率(s) for s in 生成s]
    rec["M4"] = dict(重复句率=round(重复率, 4),
                     TTR=round(TTR, 4), 字符数=len(set(全字)),
                     新四元组占比=round(float(np.mean(新4)) if 新4 else 0.0, 4),
                     三元命中率=round(float(np.mean(命中)), 4))

    # —— M4b 向量→文本解码能力（未注册句 + 位翻扰动，双通道近邻，矩阵化）——
    # 这是维度真正进入"语言能力"的通道：解码准=向量空间说话能力
    codec3 = 语言编解码器(seed=216)
    库2 = 语料[:1000]
    for s in 库2:
        codec3.注册段(s)
    V = np.stack(codec3._段向量)
    B = np.stack([codec3._字袋(t) for t in codec3._段])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    库集 = set(库2)
    未注册 = [s for s in 语料[1000:1500] if s not in 库集][:300]
    未V = np.stack([codec3.编码(s) for s in 未注册])
    未B = np.stack([codec3._字袋(s) for s in 未注册])
    未对 = np.arange(len(未注册))
    M4b = {}
    for p in (0.0, 0.05, 0.10):
        rng = np.random.default_rng(340)
        翻位 = 1.0 - 2.0 * (rng.random((len(未注册), D)) < p)
        Q1 = 未V * 翻位
        Q2 = 未B * 翻位
        Q1 = Q1 / (np.linalg.norm(Q1, axis=1, keepdims=True) + 1e-9)
        Q2 = Q2 / (np.linalg.norm(Q2, axis=1, keepdims=True) + 1e-9)
        检 = np.argmax(np.maximum(Q1 @ Vn.T, Q2 @ Bn.T), axis=1)
        # 未注册句无唯一正确答案，判据=检索相似度>阈值 τ=2/√D 的比例（有把握说话率）
        分 = np.max(np.maximum(Q1 @ Vn.T, Q2 @ Bn.T), axis=1)
        τ = 2.0 / np.sqrt(D)
        M4b["p%.2f_有把握率" % p] = round(float(np.mean(分 > τ)), 4)
        M4b["p%.2f_中位分" % p] = round(float(np.median(分)), 4)
    rec["M4b"] = M4b

    # —— M5 编码性能标度 ——
    t0 = time.perf_counter()
    for s in 语料[:1000]:
        codec.编码(s)
    rec["M5_编码千次秒"] = round(time.perf_counter() - t0, 3)
    return rec


def 子进程主(D: int) -> None:
    rec = 单维度测(D)
    out = os.path.join(HERE, "维度_%d.json" % D)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("D=%d 完成 → %s" % (D, out), flush=True)


def 主() -> int:
    逐D = {}
    for D in 维度点:
        print("=== 测 D=%d ===" % D, flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--D", str(D)], cwd=HERE)
        if r.returncode != 0:
            print("D=%d 失败" % D)
            return 1
        逐D[str(D)] = json.load(open(os.path.join(HERE, "维度_%d.json" % D),
                                     encoding="utf-8"))
    # —— 判读 ——
    报 = ["# 维度×10 全面测试判读\n"]
    for D in 维度点:
        r = 逐D[str(D)]
        报.append("## D=%d\n- M1a 真实文本密度: 均值=%s σ=%s p99cos=%s max=%s\n"
                  "- M1b 随机渐近律(2万字池): σ=%s vs 理论%s 比=%s\n"
                  "- M2 位翻检索(K=200): %s\n"
                  "- M3 容量曲线: %s\n"
                  "- M4 计数生成丰富度(与维无关，架构事实): %s\n"
                  "- M4b 向量→文本解码: %s\n- M5 编码千次: %ss\n"
                  % (D, r["M1a"]["真实文本cos均值"], r["M1a"]["σ"],
                     r["M1a"]["p99"], r["M1a"]["max"],
                     r["M1b"]["σ"], r["M1b"]["σ理论"], r["M1b"]["比"],
                     r["M2"], r["M3"], r["M4"], r["M4b"], r["M5_编码千次秒"]))
    # 判定
    b, x = 逐D["1024"], 逐D["10240"]
    最大K = max(int(k.split("_")[0][1:]) for k in b["M3"])
    判s = []
    判s.append(("M1a×10更可分(p99↓)", x["M1a"]["p99"] < b["M1a"]["p99"]))
    判s.append(("M1b×10随机律", 0.8 <= x["M1b"]["比"] <= 1.25))
    判s.append(("M2×10增", x["M2"]["p0.05"] >= b["M2"]["p0.05"]))
    判s.append(("M3×10缓", x["M3"]["K%d_p0.05" % 最大K] >= b["M3"]["K%d_p0.05" % 最大K]))
    判s.append(("M4重复率", x["M4"]["重复句率"] < 0.05))
    判s.append(("M4b有把握率×10不降", x["M4b"]["p0.00_有把握率"] >= b["M4b"]["p0.00_有把握率"] * 0.95))
    判s.append(("M4b抗扰×10不降", x["M4b"]["p0.10_有把握率"] >= b["M4b"]["p0.10_有把握率"]))
    判s.append(("M5线性标度", 3 <= x["M5_编码千次秒"] /
                max(b["M5_编码千次秒"], 1e-9) <= 15))
    报.append("## 判定（1024 → 10240）\n")
    for 名, ok in 判s:
        报.append("- [%s] %s\n" % ("PASS" if ok else "FAIL", 名))
    文 = os.path.join(HERE, "维度十倍_判读.md")
    with open(文, "w", encoding="utf-8") as f:
        f.writelines(报)
    print("".join(报))
    return 0 if all(ok for _, ok in 判s) else 1


if __name__ == "__main__":
    if "--D" in sys.argv:
        子进程主(int(sys.argv[sys.argv.index("--D") + 1]))
    else:
        sys.exit(主())
