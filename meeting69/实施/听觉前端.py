# -*- coding: utf-8 -*-
"""meeting69/实施/听觉前端.py —— S1 听觉前端（听场 1024 + 预测/误差/惊讶 + 焦点链 + 流 + 遥测）。

零训练、确定性、单线程（09 §2；C13/C14）。`拍()` 对一段已过前端 `场未适` 的分析结果
逐拍构建（100 ms 主臂；40/50 ms 仅消融），每拍增量 = 听场构建 + 预测 + 尴尬 + 焦点 + 流 + 遥测。
"""
from __future__ import annotations

import hashlib
from collections import deque
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


前 = 载("耳蜗前端_S0_2", HERE / "耳蜗前端_S0_2.py")
场 = 载("听觉场", HERE / "听觉场.py")
流 = 载("算子_流分离", HERE / "算子_流分离.py")
工 = 载("听觉工作区", HERE / "听觉工作区.py")
协 = 载("协议_听觉源", HERE / "协议_听觉源.py")

拍毫秒 = 100
意外尺度 = 0.05
预测β = 0.5
权意外 = 0.7
权显著 = 0.3
权IOR = 0.3
IORτ拍 = 4.0
IOR半带宽 = 2
IOR半宽槽 = 4
盲 = ("主臂", "随机焦点", "仅显著", "固定焦点")
意外尺度_默认 = 0.05


def 载片段(路, 变体="V3"):
    return 前.分析(路, 变体=变体)


class 听觉前端:
    def __init__(self, 拍ms=拍毫秒):
        self.拍ms = int(拍ms)
        self.区 = 工.听觉工作区()

    def 拍(self, r, 模式="主臂", 种子=6900, L=None, 焦点选择="窗口", q下限=0.05,
          意外通道="截断", 意外尺度=None, 听感权=False):
        """焦点选择：'窗口'（09 v1：725 窗 mean-ε argmax）｜'格级'（09 v2：混项格级）｜'ε'（16 v3：ε 单独格级 + 窗居中）。
        q下限：09 v1/v2 = 0.05；16 v3 = 1.0。
        意外通道：'截断'（09 原口径：ε=|Ψ−Ψ̂|）｜'原值'（18 v4：ε=|Z−Ẑ|/3）｜'原值适应'（20 v5：带内基线适应）。
        听感权：21 v6 用固定 A 计权乘到 ε（去低带底噪偏置）。"""
        if 模式 not in 盲:
            raise ValueError(f"未知模式: {模式}")
        if 焦点选择 not in ("窗口", "格级", "ε"):
            raise ValueError(f"未知焦点选择: {焦点选择}")
        if 意外通道 not in ("截断", "原值", "原值适应"):
            raise ValueError(f"未知意外通道: {意外通道}")
        if 意外尺度 is None:
            意外尺度 = 0.10 if 意外通道 == "原值适应" else (0.30 if 意外通道 == "原值" else 意外尺度_默认)
        if L is None:
            L = np.log(np.asarray(r["场未适"], np.float64) + 1e-3)
        T = len(L)
        f32 = 场.带心32(r["中心频率"])
        听感权向量 = np.maximum(场.A计权(f32), 0.05) if 听感权 else np.ones(场.频带数)
        K = T // self.拍ms
        beats = list(range(1, K))
        记 = dict(s=[], 链=[], maxε=[], 焦点=[], 流数=[], 流均ε=[], ε均=[], 活跃ε均=[], cost=[], 拍=[])
        Ψ2 = None; Ψ1 = None
        Z2 = None; Z1 = None
        ε史 = deque(maxlen=20)
        ior_foci = []
        流表 = {}
        链 = []
        h场 = hashlib.sha256(); h链 = hashlib.sha256()
        遥 = None
        for k in beats:
            t = k * self.拍ms
            t0 = time.perf_counter()
            if 意外通道 in ("原值", "原值适应"):
                Ψ, Z = 场.构建(L, t, q下限=q下限, 返回未截断=True)
                if Z1 is None:
                    predZ = np.zeros_like(Z)
                elif Z2 is None:
                    predZ = Z1.copy()
                else:
                    predZ = Z1 + 预测β * (Z1 - Z2)
                ε = (np.abs(Z - predZ) / 3.0).astype(np.float32)
                if 听感权:
                    ε = (ε * np.repeat(听感权向量, 场.时槽).astype(np.float32)).astype(np.float32)
                if 意外通道 == "原值适应":
                    带均 = ε.reshape(场.频带数, 场.时槽).mean(axis=1)      # (32,) 原始误差的带基线
                    if len(ε史) >= 4:
                        m = np.median(np.stack(list(ε史), axis=0), axis=0)
                        ε = np.maximum(ε - np.repeat(m, 场.时槽).astype(np.float32), 0.0).astype(np.float32)
                    ε史.append(带均)
            else:
                Ψ = 场.构建(L, t, q下限=q下限)
                if Ψ1 is None:
                    pred = np.zeros_like(Ψ)
                elif Ψ2 is None:
                    pred = Ψ1.copy()
                else:
                    pred = np.clip(Ψ1 + 预测β * (Ψ1 - Ψ2), -1.0, 1.0).astype(np.float32)
                ε = np.abs(Ψ - pred).astype(np.float32)
            s = float(min(2.0, max(0.0, float(ε.mean()) / 意外尺度)))
            逐格 = 场.逐格意外(ε)
            eg = ε.reshape(场.频带数, 场.时槽).astype(np.float64)
            Ψg = Ψ.reshape(场.频带数, 场.时槽).astype(np.float64)
            ior = np.zeros((场.频带数, 场.时槽), np.float64)
            for (fb, b0, s0) in ior_foci:
                val = 0.5 * math.exp(-(k - fb) / IORτ拍)
                b1, b2 = max(0, b0 - IOR半带宽), min(场.频带数, b0 + IOR半带宽 + 1)
                s1, s2 = max(0, s0 - IOR半宽槽), min(场.时槽, s0 + IOR半宽槽 + 1)
                ior[b1:b2, s1:s2] = np.maximum(ior[b1:b2, s1:s2], val)
            We = 场.窗和矩阵(eg); Wψ = 场.窗和矩阵(Ψg * Ψg); Wi = 场.窗和矩阵(ior)
            en = We / (We.max() + 1e-12)
            sn = Wψ / (Wψ.max() + 1e-12)
            iorn = Wi / float(场.频带数 * 场.时槽 / 4 / 8)      # 窗内均值（32 格）
            if 模式 == "随机焦点":
                idx = int(np.random.RandomState(种子 + k).randint(0, 29 * 25))
                b0, s0 = divmod(idx, 25)
            elif 焦点选择 in ("格级", "ε"):
                en_c = eg / (eg.max() + 1e-12)
                sn_c = (Ψg * Ψg) / ((Ψg * Ψg).max() + 1e-12)
                if 模式 == "仅显著":
                    sc = sn_c
                elif 模式 == "固定焦点":
                    b0, s0 = 0, 0
                    sc = None
                elif 焦点选择 == "ε":
                    sc = en_c
                else:
                    sc = 权意外 * en_c + 权显著 * sn_c - 权IOR * ior
                if sc is not None:
                    cb, cs = divmod(int(np.argmax(sc)), 场.时槽)
                    b0 = int(np.clip(round(cb - 1.5), 0, 28))
                    s0 = int(np.clip(round(cs - 3.5), 0, 24))
            elif 模式 == "仅显著":
                idx = int(np.argmax(权显著 * sn - 权IOR * iorn))
                b0, s0 = divmod(idx, 25)
            elif 模式 == "固定焦点":
                b0, s0 = 0, 0
            else:
                idx = int(np.argmax(权意外 * en + 权显著 * sn - 权IOR * iorn))
                b0, s0 = divmod(idx, 25)
            带心 = float(math.sqrt(f32[b0] * f32[b0 + 3]))
            带宽 = float(math.log2(f32[b0 + 3] / f32[b0]))
            停留 = 0
            if 链 and (链[-1]["b0"], 链[-1]["s0"]) == (b0, s0):
                停留 = 链[-1]["停留拍"] + 1
            链.append(dict(拍号=k, t=round(t / 1000.0, 3), b0=int(b0), s0=int(s0),
                          带心Hz=round(带心, 1), 带宽oct=round(带宽, 3), s=round(s, 4), 停留拍=停留,
                          sha=hashlib.sha256(np.ascontiguousarray(Ψ).tobytes()).hexdigest()[:16]))
            ior_foci.append((k, b0, s0))
            if len(ior_foci) > 4:
                ior_foci.pop(0)
            流表, _指派 = 流.分离(ε, 流表, k)
            流信息 = 流.流统计(流表, ε)
            self.区.写听觉(拍号=k, 听场=Ψ, 焦点=dict(b0=b0, s0=s0, 带心Hz=带心, 带宽oct=带宽, 停留拍=停留),
                          误差场=ε, 意外度=s, 逐格意外=逐格, 焦点链=链[-64:], 流表=流表)
            遥 = 协.快照(self.区, 事件率=0.0)
            cost = time.perf_counter() - t0
            h场.update(np.ascontiguousarray(Ψ).tobytes())
            h链.update(repr((k, b0, s0, round(s, 4))).encode())
            记["s"].append(s); 记["链"].append(链[-1]); 记["maxε"].append(int(np.argmax(ε)))
            记["焦点"].append((int(b0), int(s0))); 记["流数"].append(int(len(流表)))
            记["流均ε"].append({int(sid): round(float(v["平均ε"]), 6) for sid, v in 流信息.items()})
            记["ε均"].append(round(float(ε.mean()), 6))
            记["活跃ε均"].append(round(float(eg[eg > (eg.mean() + eg.std())].mean()) if (eg > (eg.mean() + eg.std())).any() else 0.0, 6))
            记["cost"].append(float(cost)); 记["拍"].append(int(k))
            Ψ2, Ψ1 = Ψ1, Ψ
            if 意外通道 == "原值":
                Z2, Z1 = Z1, Z
        记["sha_听场"] = h场.hexdigest()[:16]
        记["sha_焦点链"] = h链.hexdigest()[:16]
        记["sha_遥测"] = (遥 or {}).get("指纹")
        记["拍ms"] = self.拍ms
        记["片段秒"] = round(T / 1000.0, 3)
        return 记


