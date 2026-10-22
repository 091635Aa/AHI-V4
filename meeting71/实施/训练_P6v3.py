# -*- coding: utf-8 -*-
"""meeting71/实施/训练_P6v3.py —— P6 v3：mel 目标 + 自条件（prev-mel）+ 推理迭代。

v2 判死原因：非自条件 mel 回归 → 预测塌到均值（ASR「啦啦啦」）；但同合成链喂**真实 mel** CER 0.093
（自检 `自检117_mel.wav`）⇒ 合成无罪，缺的是 mel 质量。
v3（v13 的教训：自条件+计划采样是拟合关键）：输入加 prev-mel，训练教师强制、推理 3 轮阻尼迭代。
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M71 = HERE.parent
产 = M71 / "产物" / "P6v3"
v2 = None


def 载v2():
    global v2
    if v2 is None:
        s = importlib.util.spec_from_file_location("v2mod", HERE / "训练_P6v2.py")
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        v2 = m
    return v2


import torch
import torch.nn as nn

设备 = "cuda" if torch.cuda.is_available() else "cpu"
P = 载v2()


class 生成器AR(nn.Module):
    def __init__(self, 通道=384, 层=5):
        super().__init__()
        self.e_sm = nn.Embedding(22, 32)
        self.e_ym = nn.Embedding(39, 32)
        self.e_tone = nn.Embedding(5, 32)
        self.ina = nn.Conv1d(99 + 80, 通道, 5, padding=2)
        self.blocks = nn.ModuleList()
        for k in range(层):
            d = 2 ** k
            self.blocks.append(nn.Sequential(
                nn.Conv1d(通道, 通道, 5, padding=2 * d, dilation=d), nn.GELU(),
                nn.Conv1d(通道, 通道, 1), nn.GELU()))
        self.head = nn.Conv1d(通道, 81, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tok, rel, pos, logT, prev):
        e = torch.cat([self.e_sm(tok[..., 0]), self.e_ym(tok[..., 1]), self.e_tone(tok[..., 2])], -1)
        x = torch.cat([e, rel[..., None], pos[..., None], logT[..., None], prev], -1).transpose(1, 2)
        h = self.ina(x)
        for b in self.blocks:
            h = h + b(h)
        y = self.head(h).transpose(1, 2)
        return y[..., :80], y[..., 80]


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "推理":
        return 推理()
    return 训练(int(sys.argv[2]) if len(sys.argv) > 2 else 200,
               int(sys.argv[3]) if len(sys.argv) > 3 else 60)


def 训练(轮数=200, 每轮批=60):
    产.mkdir(parents=True, exist_ok=True)
    with open(M71 / "产物" / "P6v2" / "数据.pkl", "rb") as f:
        d = pickle.load(f)
    件 = d["件"]
    所有mel = np.concatenate([x["mel"] for x in 件], 0)
    mu = 所有mel.mean(0); sd = 所有mel.std(0) + 1e-6
    np.savez(产 / "统计.npz", mu=mu, sd=sd)
    sids = sorted(x["sid"] for x in 件)
    val_sids = set(sids[-6:])
    train = [x for x in 件 if x["sid"] not in val_sids]
    val = [x for x in 件 if x["sid"] in val_sids]
    print("train", len(train), "val", len(val), "device", 设备, flush=True)

    def 预(表):
        out = []
        for x in 表:
            tok, rel, pos, logT = P.构造帧输入(x["toks"], x["durs"])
            T = min(len(tok), len(x["mel"]), len(x["f0"]))
            v = (x["f0"][:T] > 0).astype(np.float32)
            f0s = np.where(v > 0, 12 * np.log2(np.maximum(x["f0"][:T], 1e-3) / 300.0), 0.0)
            mn = (x["mel"][:T] - mu) / sd
            out.append(dict(tok=tok[:T], rel=rel[:T], pos=pos[:T], logT=logT[:T],
                            mn=mn.astype(np.float32), f0s=f0s.astype(np.float32), v=v))
        return out
    tr = 预(train); va = 预(val)
    模型 = 生成器AR().to(设备)
    opt = torch.optim.AdamW(模型.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=轮数)
    最佳 = 1e9
    for ep in range(1, 轮数 + 1):
        模型.train(); 批 = []
        for _ in range(每轮批):
            x = tr[np.random.randint(0, len(tr))]
            T = len(x["f0s"]); W = 800
            s = 0 if T <= W else np.random.randint(0, T - W)
            e = min(T, s + W)
            if e - s < 200:
                continue
            sl = slice(s, e)
            prev = np.concatenate([np.zeros((1, 80), np.float32), x["mn"][s:e - 1]], 0)
            批.append((x["tok"][sl], x["rel"][sl], x["pos"][sl], x["logT"][sl],
                      x["mn"][sl], prev, x["f0s"][sl], x["v"][sl]))
        W = min(len(b[0]) for b in 批)
        批 = [tuple(np.asarray(a)[:W] for a in b) for b in 批]

        def T_(arr):
            return torch.as_tensor(arr, dtype=torch.float32, device=设备)
        tok = torch.as_tensor(np.stack([b[0] for b in 批]).astype(np.int64), device=设备)
        rel = T_(np.stack([b[1] for b in 批])); pos = T_(np.stack([b[2] for b in 批]))
        logT = T_(np.stack([b[3] for b in 批])); mn = T_(np.stack([b[4] for b in 批]))
        prev = T_(np.stack([b[5] for b in 批]))
        f0 = T_(np.stack([b[6] for b in 批])); v = T_(np.stack([b[7] for b in 批]))
        pm, pf = 模型(tok, rel, pos, logT, prev)
        l_mel = (pm - mn).abs().mean()
        l_f = (((pf - f0) ** 2) * v).sum() / (v.sum() + 1e-6)
        loss = l_mel + 1.0 * l_f
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(模型.parameters(), 3.0); opt.step(); sched.step()
        if ep % 10 == 0:
            模型.eval(); vl = 0.0; cnt = 0
            with torch.no_grad():
                for x in va:
                    T = len(x["f0s"])
                    if T < 200:
                        continue
                    Wv = min(800, T); s = max(0, (T - Wv) // 2); e = s + Wv
                    prevv = np.concatenate([np.zeros((1, 80), np.float32), x["mn"][s:e - 1]], 0)
                    pm2, pf2 = 模型(torch.as_tensor(x["tok"][s:e][None].astype(np.int64), device=设备),
                                   torch.as_tensor(x["rel"][s:e][None], dtype=torch.float32, device=设备),
                                   torch.as_tensor(x["pos"][s:e][None], dtype=torch.float32, device=设备),
                                   torch.as_tensor(x["logT"][s:e][None], dtype=torch.float32, device=设备),
                                   torch.as_tensor(prevv[None], dtype=torch.float32, device=设备))
                    mv = torch.as_tensor(x["mn"][s:e][None], dtype=torch.float32, device=设备)
                    fv = torch.as_tensor(x["f0s"][s:e][None], dtype=torch.float32, device=设备)
                    vv = torch.as_tensor(x["v"][s:e][None], dtype=torch.float32, device=设备)
                    vl += float((pm2 - mv).abs().mean() + (((pf2 - fv) ** 2) * vv).sum() / (vv.sum() + 1e-6))
                    cnt += 1
                vl /= max(1, cnt)
            print("| ep", ep, "loss %.4f" % float(loss.item()), "val %.4f" % vl, flush=True)
            if vl < 最佳:
                最佳 = vl
                torch.save({"模型": 模型.state_dict(), "val": vl}, 产 / "模型.pt")
    torch.save({"模型": 模型.state_dict(), "val": 最佳}, 产 / "模型.pt")
    print("训练完成 最佳val", 最佳, flush=True)
    return 0


def 推理(每音节=24, 轮=3, 阻尼=0.5):
    with open(M71 / "产物" / "P6v2" / "数据.pkl", "rb") as f:
        d = pickle.load(f)
    字表 = d["字表"]
    st = np.load(产 / "统计.npz"); mu = st["mu"]; sd = st["sd"]
    ck = torch.load(产 / "模型.pt", map_location=设备)
    模型 = 生成器AR().to(设备); 模型.load_state_dict(ck["模型"]); 模型.eval()
    报 = dict(工具="训练_P6v3.py 推理", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    with torch.no_grad():
        for 名, 文 in P.句表.items():
            keys = P.键_of(文, 字表)
            toks = np.array(keys, np.int64)
            durs = np.full(len(toks), 每音节, np.int64)
            tok, rel, pos, logT = P.构造帧输入(toks, durs)
            tokv = torch.as_tensor(tok[None].astype(np.int64), device=设备)
            relv = torch.as_tensor(rel[None], dtype=torch.float32, device=设备)
            posv = torch.as_tensor(pos[None], dtype=torch.float32, device=设备)
            logTv = torch.as_tensor(logT[None], dtype=torch.float32, device=设备)
            T = len(tok)
            prev = np.zeros((1, T, 80), np.float32)          # 归一化域
            f0p = None
            for r in range(轮):
                pm, pf = 模型(tokv, relv, posv, logTv, torch.as_tensor(prev, dtype=torch.float32, device=设备))
                cur = pm[0].cpu().numpy()
                if f0p is None:
                    f0p = pf[0].cpu().numpy()
                prev = (阻尼 * prev[0] + (1 - 阻尼) * cur)[None] if r > 0 else cur[None]
            mel = prev[0] * sd + mu
            f0 = 300.0 * (2.0 ** (np.clip(f0p, -12, 12) / 12.0))
            f0s = np.copy(f0)
            for i in range(len(f0s)):
                lo = max(0, i - 2); hi = min(len(f0s), i + 3)
                f0s[i] = float(np.median(f0[lo:hi]))
            mp = np.exp(np.clip(mel, -40, 40))
            flat = np.exp(np.mean(np.log(mp + 1e-10), 1)) / (np.mean(mp, 1) + 1e-10)
            NR = np.clip((flat - 0.10) / 0.60, 0.0, 1.0)
            y = P.合成mel(mel, f0s, NR)
            P.写(产 / f"{名}.wav", y)
            报["项"].append(dict(样本=名, 文本=文, 路径=str(产 / f"{名}.wav"), 秒=round(len(y) / P.SR, 3)))
            print("|", 名, round(len(y) / P.SR, 2), "s", flush=True)
    引 = P.量.载引擎()
    from faster_whisper import WhisperModel
    import glob
    snap = glob.glob(str(Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-medium" / "snapshots" / "*"))[0]
    wm = WhisperModel(snap, device="cpu", compute_type="int8")
    for x in 报["项"]:
        q = 引["转写"](x["路径"])
        segs, _ = wm.transcribe(x["路径"], language="zh", beam_size=5)
        wh = "".join(s.text for s in segs).strip()
        x.update(转写=q, Whisper=wh, CER=round(P.量.CER(x["文本"], q), 4))
        print("|", x["样本"], "CER=%.3f" % x["CER"], "|", q[:30], flush=True)
    r0 = [x for x in 报["项"] if "CER" in x]
    报["均值CER"] = round(float(np.mean([x["CER"] for x in r0])), 4) if r0 else None
    (产 / "推理_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("均值", 报["均值CER"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
