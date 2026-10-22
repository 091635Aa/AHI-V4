# -*- coding: utf-8 -*-
"""AHI4/meeting51/实施/环境_千层任务.py —— 千层判据用的确定性任务环境（C9/C10/C13 合规）。

本件只造**世界序列**，不含任何学习器；随机只来自 `default_rng(显式种子)`，不用外部模型。
四个环境（对应 05 §3 判据总表中各组的探针）：

  ① `生成序列_乘积`（T1）
       观测量：12 通道 f∈{±1}；每拍把 (f_a, f_{a+1}) 送给分区 a。
       世界：  f_a(t+1) = (∏_{k=0..阶-1} f_{a+k}(t)) · n_a(t)，n_a=−1 的概率 = 噪率。
       目标：  y_a(t) = f_a(t+1) 的符号（类 0/1）。
       阶=3 ⇒ T1主：3 阶乘积；1 档也可（靠慢/快时间档平方出交叉项）补偿。
       阶=4 ⇒ T1深：4 阶乘积；信息需跨 4 个通道，1 档（只看 2 通道）不可达 ⇒ 深度必要性探针。
       理想预测上限 = 1 − 噪率（噪声不可预测）。

  ② `生成序列_全局奇偶`（T2）
       世界：  f_a(t+1) = f_a(t) · g(t) · n_a(t)，g(t)=∏_b f_b(t)（全局奇偶位）。
       目标：  f_a(t+1) 的符号 ⇒ 必须由**全部 12 通道**算出 g(t)（局部拍不到）。
       用途：  乙3 环路必要性（单拍只够半环）、乙4 反馈必要性（缺一反方向则缺半环）。

  ③ `生成模式` + `损坏`（T3 记忆）
       K 个 ±1 模式（维度=分区×输入维）；按损坏率把部分分量翻成随机符号。
       用途：  丙1 残缺线索召回相变 / 丙2 冻结-解冻 / 丙5 冻结不参与广播。

  ④ `生成序列_竞争`（T4 竞争）
       两路同分布输入 A/B（各占 D 维的一半区域），加 ±偏置。
       用途：  乙2 竞争胜出可复现（Δ=0.01 可复现切换胜者，且两趟逐位同）。
"""
from __future__ import annotations

import numpy as np

# ── 冻结常量（判据不据结果改）──
通道数默认 = 12
输入维 = 2                 # 每分区输入 (f_a, f_{a+1})
种子环境默认 = 52001


def _rng(种子):
    return np.random.default_rng(int(种子))


def _符号列(x):
    """±1 序列 → 类标签（+1→0, −1→1）。"""
    return np.where(np.asarray(x) > 0, 0, 1).astype(np.int64)


def 生成序列_乘积(种子, 长度, 阶=3, 噪率=0.02, 通道数=通道数默认, 方向="前向"):
    """T1：f_a(t+1) = (∏_{k<阶} f_{a+k}(t))·n_a(t)；返回 输入/目标/状态 三个数组。"""
    rng = _rng(种子)
    n = int(长度)
    N = int(通道数)
    K = int(阶)
    f = np.empty((n + 1, N), dtype=np.int64)
    f[0] = np.where(rng.random(N) < 0.5, 1, -1)
    for t in range(n):
        积 = np.ones(N, dtype=np.int64)
        for k in range(K):
            积 = 积 * f[t, (np.arange(N) + k) % N]
        噪 = np.where(rng.random(N) < float(噪率), -1, 1).astype(np.int64)
        f[t + 1] = 积 * 噪
    f = f[:n]                                              # 只保留观测量
    输入 = np.zeros((n, N, 输入维), dtype=np.float64)
    for a in range(N):
        输入[:, a, 0] = f[:, a]
        输入[:, a, 1] = f[:, (a + 1) % N]
    # 目标：下一拍符号（末拍用自身，避免越界；判据只用前 n−1 拍）
    目标 = np.zeros((n, N), dtype=np.int64)
    目标[:-1] = _符号列(f[1:])
    目标[-1] = _符号列(f[-1])
    return dict(输入=输入, 目标=目标, 状态=f, 长度=n, 通道数=N, 阶=K, 噪率=float(噪率),
                名称="T1乘积", 上限=float(1.0 - float(噪率)))


def 生成序列_静态(种子, 长度, 阶=3, 噪率=0.05, 通道数=通道数默认, 方向="前向"):
    """T1'（有效样本充足版，判据主用）：f 每拍独立 ±1（无长相关），目标是**当前**通道的
    乘积布尔函数（自监督形态：“预测下一拍观测”退化为可观测目标）。样本独立 ⇒ 线性读出可学。

    方向="前向"：y_a = ∏_{k<阶} f_{a+k}（需要前馈方向信息）
    方向="后向"：y_a = ∏_{k<阶} f_{a−k}（需要反馈方向信息；乙4 探针）
    方向="双向"：y_a = (∏_{k<阶} f_{a+k})·(∏_{k<阶} f_{a−k})
    """
    rng = _rng(种子)
    n = int(长度)
    N = int(通道数)
    K = int(阶)
    f = np.where(rng.random((n, N)) < 0.5, 1, -1).astype(np.int64)
    目标积 = np.ones((n, N), dtype=np.int64)
    if 方向 == "前向":
        取 = [(np.arange(N) + k) % N for k in range(K)]
    elif 方向 == "后向":
        取 = [(np.arange(N) - k) % N for k in range(K)]
    elif 方向 == "双向":
        取 = [(np.arange(N) + k) % N for k in range(K)] + [(np.arange(N) - k) % N for k in range(1, K)]
    else:
        raise ValueError("方向须为 前向/后向/双向，收到 %r" % (方向,))
    for 位 in 取:
        目标积 = 目标积 * f[:, 位]
    噪 = np.where(rng.random((n, N)) < float(噪率), -1, 1).astype(np.int64)
    目标积 = 目标积 * 噪
    输入 = np.zeros((n, N, 输入维), dtype=np.float64)
    for a in range(N):
        输入[:, a, 0] = f[:, a]
        输入[:, a, 1] = f[:, (a + 1) % N]
    return dict(输入=输入, 目标=_符号列(目标积), 状态=f, 长度=n, 通道数=N, 阶=K,
                噪率=float(噪率), 方向=str(方向), 名称="T1静态", 上限=float(1.0 - float(噪率)))


def 生成序列_全局奇偶(种子, 长度, 噪率=0.02, 通道数=通道数默认):
    """T2：f_a(t+1)=f_a(t)·g(t)·n_a(t)，g(t)=∏_b f_b(t)（全局奇偶位）。"""
    rng = _rng(种子)
    n = int(长度)
    N = int(通道数)
    f = np.empty((n + 1, N), dtype=np.int64)
    f[0] = np.where(rng.random(N) < 0.5, 1, -1)
    for t in range(n):
        g = np.prod(f[t])
        噪 = np.where(rng.random(N) < float(噪率), -1, 1).astype(np.int64)
        f[t + 1] = f[t] * g * 噪
    f = f[:n]
    输入 = np.zeros((n, N, 输入维), dtype=np.float64)
    for a in range(N):
        输入[:, a, 0] = f[:, a]
        输入[:, a, 1] = f[:, (a + 1) % N]
    目标 = np.zeros((n, N), dtype=np.int64)
    目标[:-1] = _符号列(f[1:])
    目标[-1] = _符号列(f[-1])
    return dict(输入=输入, 目标=目标, 状态=f, 长度=n, 通道数=N, 阶=0, 噪率=float(噪率),
                名称="T2全局奇偶", 上限=float(1.0 - float(噪率)))


def 生成模式(种子, 个数=8, 维=通道数默认 * 输入维):
    """T3：K 个 ±1 模式（行向量，已展平）。"""
    rng = _rng(种子)
    return np.where(rng.random((int(个数), int(维))) < 0.5, 1.0, -1.0)


def 损坏(模式, 损坏率, rng):
    """把损坏率比例的分量替换为随机 ±1（确定性：由传入 rng 决定）。"""
    p = np.asarray(模式, dtype=np.float64).copy()
    n = int(p.size)
    k = int(round(float(损坏率) * n))
    位置 = rng.permutation(n)[:k]
    p[位置] = np.where(rng.random(k) < 0.5, 1.0, -1.0)
    return p


def 模式为输入(模式行, 通道数=通道数默认, 输入维=输入维):
    """展平模式 → (通道数, 输入维) 的外部输入矩阵。"""
    return np.asarray(模式行, dtype=np.float64).reshape(int(通道数), int(输入维))


def 生成序列_竞争(种子, 长度, 偏置=0.0, 维=64):
    """T4：两路同分布输入；A 占前 维/2 维、B 占后 维/2 维；B 路加 偏置。"""
    rng = _rng(种子)
    n = int(长度)
    d = int(维)
    甲 = np.zeros((n, d), dtype=np.float64)
    乙 = np.zeros((n, d), dtype=np.float64)
    for t in range(n):
        甲[t, : d // 2] = np.where(rng.random(d // 2) < 0.5, 1.0, -1.0)
        乙[t, d // 2:] = np.where(rng.random(d - d // 2) < 0.5, 1.0, -1.0)
    return dict(甲=甲, 乙=乙, 偏置=float(偏置), 长度=n, 维=d)


def 准确率(预测, 目标):
    """逐分区平均准确率（预测/目标均为 (拍, 通道) 整数类）。"""
    预 = np.asarray(预测, dtype=np.int64)
    真 = np.asarray(目标, dtype=np.int64)
    if 预.shape != 真.shape:
        raise ValueError("预测形状 %r 与目标形状 %r 不一致" % (预.shape, 真.shape))
    return float((预 == 真).mean())


def 逐通道准确率(预测, 目标):
    预 = np.asarray(预测, dtype=np.int64)
    真 = np.asarray(目标, dtype=np.int64)
    return [float((预[:, a] == 真[:, a]).mean()) for a in range(预.shape[1])]
