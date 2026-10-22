# -*- coding: utf-8 -*-
"""meeting55/实施/决策_网络.py —— 听觉状态 → 决策网络（torch 小 MLP）→ 执行参数。

输入 12 维（相对 F0/能量/语速/停顿/谱形 + 情绪 one-hot + 声纹占位）；
输出 3 头：回应情绪(3) / 语速档(3) / 音量档(3)。teacher=规则（合成数据）。
必须失败：固定输入不变；打乱标签学不会（≈机会）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import 听_多源 as 听  # noqa: E402

产物 = os.path.join(HERE, "产物")
os.makedirs(产物, exist_ok=True)
情绪列 = ("平静", "高兴", "低落")
语速档 = ("慢", "中", "快")
音量档 = ("低", "中", "高")


def _输入(f, emo, 音="女声-明亮"):
    b = 听.标定()["规则"]
    f0 = f["F0均值"] if f["F0均值"] > 0 else b.F0基线.get(音, 190)
    st = 12.0 * np.log2(max(f0, 60) / max(b.F0基线.get(音, 190), 60))
    db = 10.0 * np.log10(max(f["能量"], 1e-8) / max(b.能基线.get(音, 1e-5), 1e-8))
    v = [st, db, f["峰率"], f["停顿比"], np.log(max(f["F0标准差"], 1) / 30.0),
         np.log(max(f["谱重心"], 100) / 2000.0), np.log(max(f["谱平坦"], 1e-5) / 1e-3),
         f["F0范围"] / 200.0]
    v += [1.0 if emo[0] == k else 0.0 for k in 情绪列]
    v += [1.0]
    return np.array(v, dtype=np.float64)


def teacher(情绪, 风格):
    if 情绪 == "低落":
        return "平静", "慢", "低"
    if 情绪 == "高兴":
        return "高兴", "快", "中"
    if 风格.get("语速") == "快":
        return "平静", "中", "低"
    return "平静", "中", "中"


def _典型特征(情绪, 语速, rng):
    标 = 听.标定()
    样本 = 标.get("_样本") or []
    候 = [f for 音, f, e in 样本 if e == 情绪] or [f for _, f, _ in 样本]
    if 候:
        f = dict(候[int(rng.integers(0, len(候)))])
        if 语速 == "快":
            f["峰率"] *= 1.25
        elif 语速 == "慢":
            f["峰率"] *= 0.8
        return f
    return dict(F0均值=200.0, 能量=0.01, 峰率=3.2, 停顿比=0.3, F0标准差=30.0,
                谱重心=1800.0, 谱平坦=1e-3, F0范围=60.0)


class 网(nn.Module):
    def __init__(self, 隐=32):
        super().__init__()
        self.fc1 = nn.Linear(12, 隐); self.act = nn.Tanh()
        self.he = nn.Linear(隐, 3); self.hr = nn.Linear(隐, 3); self.hv = nn.Linear(隐, 3)

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.he(h), self.hr(h), self.hv(h)


class 决策网络:
    def __init__(self, 种子=5501):
        torch.manual_seed(int(种子))
        self.m = 网()
        self.均值 = None; self.标准差 = None

    def 数据(self, n=800, 种子=5502):
        rng = np.random.default_rng(种子)
        X, Ye, Yr, Yv = [], [], [], []
        for _ in range(n):
            情 = str(rng.choice(情绪列))
            速 = str(rng.choice(["快", "中", "慢"]))
            e, r, v = teacher(情, dict(语速=速))
            f = _典型特征(情, 速, rng)
            X.append(_输入(f, (情,)))
            Ye.append(情绪列.index(e)); Yr.append(语速档.index(r)); Yv.append(音量档.index(v))
        return np.array(X), np.array(Ye), np.array(Yr), np.array(Yv)

    def 训练(self, 轮=220, 学习率=0.02):
        X, Ye, Yr, Yv = self.数据()
        self.均值 = X.mean(axis=0); self.标准差 = X.std(axis=0) + 1e-8
        Z = torch.tensor((X - self.均值) / self.标准差, dtype=torch.float32)
        优 = torch.optim.Adam(self.m.parameters(), lr=学习率)
        损 = nn.CrossEntropyLoss()
        te = torch.tensor(Ye); tr = torch.tensor(Yr); tv = torch.tensor(Yv)
        for _ in range(int(轮)):
            优.zero_grad()
            oe, orr, ov = self.m(Z)
            L = 损(oe, te) + 损(orr, tr) + 损(ov, tv)
            L.backward(); 优.step()
        with torch.no_grad():
            oe, orr, ov = self.m(Z)
            甲 = float((oe.argmax(1) == te).float().mean())
            乙 = float((orr.argmax(1) == tr).float().mean())
            丙 = float((ov.argmax(1) == tv).float().mean())
        self.训练准确率 = dict(回应情绪=round(甲, 3), 语速=round(乙, 3), 音量=round(丙, 3))
        return self

    def 判(self, x):
        with torch.no_grad():
            z = torch.tensor(((np.asarray(x) - self.均值) / self.标准差)[None, :], dtype=torch.float32)
            oe, orr, ov = self.m(z)
        e = 情绪列[int(oe.argmax(1))]
        r = 语速档[int(orr.argmax(1))]
        v = 音量档[int(ov.argmax(1))]
        return dict(回应情绪=e, 语速=r, 音量=v,
                    置信=round(float(torch.softmax(oe, 1).max()), 3))

    def 存(self, 路径):
        torch.save(dict(m=self.m.state_dict(), 均值=self.均值.tolist(),
                        标准差=self.标准差.tolist(),
                        训练准确率=getattr(self, "训练准确率", {})), 路径)

    @classmethod
    def 读(cls, 路径):
        d = torch.load(路径, map_location="cpu")
        o = cls(); o.m.load_state_dict(d["m"])
        o.均值 = np.array(d["均值"]); o.标准差 = np.array(d["标准差"])
        o.训练准确率 = d.get("训练准确率", {})
        return o


_网 = None


def 取网():
    global _网
    if _网 is None:
        p = os.path.join(产物, "决策网络.pt")
        if os.path.exists(p):
            _网 = 决策网络.读(p)
        else:
            _网 = 决策网络().训练()
            _网.存(p)
    return _网


def 决策(听事件, 声纹名=None):
    r = 取网().判(_输入(听事件["特征"], (听事件["情绪"]["情绪"],), 声纹名 or "女声-明亮"))
    r["输入情绪"] = 听事件["情绪"]["情绪"]
    r["输入风格"] = 听事件["风格"]
    return r


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    网 = 决策网络().训练()
    print("训练准确率:", 网.训练准确率)
    网.存(os.path.join(产物, "决策网络.pt"))
    for 情 in 情绪列:
        f = _典型特征(情, "中", np.random.default_rng(1))
        r = 网.判(_输入(f, (情,)))
        print("用户情绪", 情, "→ 决策", r)
