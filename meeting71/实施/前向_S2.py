# -*- coding: utf-8 -*-
"""meeting71/实施/前向_S2.py —— S2 自监督前向模型：控制帧 → 内耳场（自己的耳）统计。

数据：8 条语料（够探针）；对每条
  控制侧：器官v2.分析控制 → 10 ms 帧 [logF0/300(浊音), logE, NR, LPC 倒谱 c1..c12]（16 维）
  内耳侧：耳蜗前端_S0_2.分析 → 场未适 (1 ms,64 带) → 10 ms 均值 log（64 维）
模型：MLP 16→256→256→64（无 attention），AdamW；按样本留出（2 条）评估 R²。
确定性：同种子两次训练 → 留出预测 sha256 逐位同。
输出：<path>\AHI_m71\S2\结果.json
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M70 = HERE.parent.parent / "meeting70"
M69 = HERE.parent.parent / "meeting69"
E = Path(r"<path>\AHI_m71")
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", M70 / "实施" / "器官v2.py")
耳 = 载("耳蜗前端_S0_2", M69 / "实施" / "耳蜗前端_S0_2.py")

import torch
import torch.nn as nn


class 前向(nn.Module):
    def __init__(self, 入=16, 出=64, 宽=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(入, 宽), nn.GELU(),
            nn.Linear(宽, 宽), nn.GELU(),
            nn.Linear(宽, 出))

    def forward(self, x):
        return self.net(x)


def 倒谱由A(A, 阶=12):
    out = np.zeros((len(A), 阶))
    for i, a in enumerate(A):
        p = len(a) - 1
        c = np.zeros(阶 + 1)
        c[0] = np.log(max(float(np.dot(a, a)), 1e-12)) / 2
        for n in range(1, 阶 + 1):
            s = 0.0
            for k in range(1, min(n, p)):
                s += k * c[k] * a[n - k]
            c[n] = -a[n] - s / n if n <= p else -s / n
        out[i] = c[1:]
    return out


def 准备():
    件 = []
    meta = json.loads((HERE.parent.parent / "meeting66" / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    ids = sorted(样本)[:8]
    t0 = time.time()
    for sid in ids:
        wav = Path(r"<path>\SpeakerA7代模型训练数据") / 样本[sid]["wav"]
        c = 器.分析控制(器.耳.读wav(wav))
        a = 耳.分析(str(wav))
        场 = np.asarray(a["场未适"], np.float32)                 # (T1,64) 1 ms
        f0 = np.asarray(c["f0"][::2], np.float64)                 # 10 ms
        Ee = np.asarray(c["E"][::2], np.float64)
        NR = np.asarray(c["NR"][::2], np.float64)
        ce = 倒谱由A(c["A"][::2])
        T = min(len(f0), len(Ee), len(NR), len(ce), len(场) // 10)
        X = np.concatenate([
            np.where(f0[:T] > 0, np.log(np.maximum(f0[:T], 1.0) / 300.0), 0.0)[:, None],
            np.log(np.maximum(Ee[:T], 1e-6))[:, None],
            NR[:T, None], ce[:T]], 1).astype(np.float32)
        Y = np.log(np.maximum(场[:T * 10].reshape(T, 10, 64).mean(1), 1e-6)).astype(np.float32)
        件.append((X, Y))
        print("|", sid, X.shape, round(time.time() - t0, 1), "s", flush=True)
    return 件, ids


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    E.mkdir(parents=True, exist_ok=True)
    (E / "S2").mkdir(parents=True, exist_ok=True)
    件, ids = 准备()
    Xall = np.concatenate([x for x, _ in 件], 0)
    Yall = np.concatenate([y for _, y in 件], 0)
    mu = Yall.mean(0); sd = Yall.std(0) + 1e-6
    Yn = (Yall - mu) / sd
    留出 = {len(件) - 2, len(件) - 1}
    训 = [i for i in range(len(件)) if i not in 留出]
    Xtr = np.concatenate([件[i][0] for i in 训], 0); Ytr = np.concatenate([Yn[(件[i][0]).shape[0] * 0 if False else 0:0] for i in []], 0) if False else None
    # 逐样本索引拼接
    idx_tr, idx_va = [], []
    off = 0
    for i, (x, _) in enumerate(件):
        (idx_va if i in 留出 else idx_tr).extend(range(off, off + len(x)))
        off += len(x)
    Xtr, Ytr = Xall[idx_tr], Yn[idx_tr]
    Xva, Yva = Xall[idx_va], Yn[idx_va]
    print("train", Xtr.shape, "val", Xva.shape, flush=True)

    def 训练一次(种子):
        torch.manual_seed(种子)
        模型 = 前向()
        opt = torch.optim.AdamW(模型.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.as_tensor(Xtr); yt = torch.as_tensor(Ytr)
        for step in range(3000):
            i = torch.randint(0, len(xt), (256,))
            p = 模型(xt[i])
            loss = ((p - yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pv = 模型(torch.as_tensor(Xva)).numpy()
        return 模型, pv, float(loss.item())

    模型, pv, 末损 = 训练一次(0)
    模型2, pv2, _ = 训练一次(0)
    sha = lambda a: hashlib.sha256(np.asarray(a, np.float32).tobytes()).hexdigest()[:16]
    # R²（逐带，再平均）
    yv = Yva
    ss_res = ((yv - pv) ** 2).sum(0)
    ss_tot = ((yv - yv.mean(0)) ** 2).sum(0) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    报 = dict(工具="前向_S2.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 样本=ids,
              训练帧=int(len(Xtr)), 留出帧=int(len(Xva)), 训练末batch损失=round(末损, 5),
              R2均值=round(float(r2.mean()), 4), R2中位=round(float(np.median(r2)), 4),
              R2最小=round(float(r2.min()), 4), R2最大=round(float(r2.max()), 4),
              留出预测sha=sha(pv), 两次训练sha同=sha(pv) == sha(pv2))
    (E / "S2" / "结果.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
