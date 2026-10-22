# -*- coding: utf-8 -*-
"""meeting71/实施/训练_P6v2.py —— P6 v2：mel 目标控制生成器（D4 全量，非 Transformer 从零）。

用法：
  & $py 训练_P6v2.py 训练 [轮数] [每轮批]
  & $py 训练_P6v2.py 推理
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
import time
import wave
from pathlib import Path

import numpy as np
import librosa

HERE = Path(__file__).resolve().parent
M71 = HERE.parent
M70 = M71.parent / "meeting70"
M64 = M71.parent / "meeting64" / "实施"
产 = M71 / "产物" / "P6v2"
SR = 22050
声母表 = ["∅", "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]
韵母表 = ["a", "o", "e", "i", "u", "v", "er", "ii", "iii", "ai", "ei", "ao", "ou", "ia", "ie", "iao", "iou", "ua", "uo",
        "uai", "uei", "ve", "an", "en", "in", "uen", "vn", "ang", "eng", "ing", "ong", "iong", "ian", "uan", "uang",
        "van", "", "iang", "io"]
句表 = {
    "P6v2主句": "今天天气很好我听见了自己的声音我想再说一遍",
    "P6v2新句1": "今天的风很轻我慢慢说给你听",
    "P6v2新句2": "窗外的雨停了我该休息一会",
    "P6v2新句3": "你先别走我还想再说两句",
}
参数 = dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", M70 / "实施" / "器官v2.py")
量 = 载("量具_v9", M64 / "量具_v9.py")

import torch
import torch.nn as nn

设备 = "cuda" if torch.cuda.is_available() else "cpu"


class 生成器(nn.Module):
    def __init__(self, 通道=256, 层=4, 出=81):
        super().__init__()
        self.e_sm = nn.Embedding(22, 32)
        self.e_ym = nn.Embedding(39, 32)
        self.e_tone = nn.Embedding(5, 32)
        self.ina = nn.Conv1d(99, 通道, 5, padding=2)
        self.blocks = nn.ModuleList()
        for k in range(层):
            d = 2 ** k
            self.blocks.append(nn.Sequential(
                nn.Conv1d(通道, 通道, 5, padding=2 * d, dilation=d), nn.GELU(),
                nn.Conv1d(通道, 通道, 1), nn.GELU()))
        self.head = nn.Conv1d(通道, 出, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tok, rel, pos, logT):
        e = torch.cat([self.e_sm(tok[..., 0]), self.e_ym(tok[..., 1]), self.e_tone(tok[..., 2])], -1)
        x = torch.cat([e, rel[..., None], pos[..., None], logT[..., None]], -1).transpose(1, 2)
        h = self.ina(x)
        for b in self.blocks:
            h = h + b(h)
        y = self.head(h).transpose(1, 2)
        return y[..., :80], y[..., 80]


def 构造帧输入(toks, durs):
    T = int(np.sum(durs))
    tokf = np.repeat(toks, durs, axis=0)[:T]
    rel = np.concatenate([np.linspace(0, 1, d, endpoint=True) for d in durs if d > 0])[:T]
    pos = np.linspace(0, 1, T, endpoint=True)
    logT = np.full(T, np.log(max(T, 1)))
    return tokf, rel, pos, logT


def 训练(轮数=300, 每轮批=60):
    产.mkdir(parents=True, exist_ok=True)
    with open(产 / "数据.pkl", "rb") as f:
        d = pickle.load(f)
    件 = d["件"]
    sids = sorted(x["sid"] for x in 件)
    val_sids = set(sids[-6:])
    train = [x for x in 件 if x["sid"] not in val_sids]
    val = [x for x in 件 if x["sid"] in val_sids]
    print("train", len(train), "val", len(val), "device", 设备, flush=True)

    def 预(表):
        out = []
        for x in 表:
            tok, rel, pos, logT = 构造帧输入(x["toks"], x["durs"])
            T = min(len(tok), len(x["mel"]), len(x["f0"]))
            v = (x["f0"][:T] > 0).astype(np.float32)
            f0s = np.where(v > 0, 12 * np.log2(np.maximum(x["f0"][:T], 1e-3) / 300.0), 0.0)
            out.append(dict(tok=tok[:T], rel=rel[:T], pos=pos[:T], logT=logT[:T],
                            mel=x["mel"][:T], f0s=f0s.astype(np.float32), v=v))
        return out
    tr = 预(train); va = 预(val)
    模型 = 生成器().to(设备)
    opt = torch.optim.AdamW(模型.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=轮数)
    最佳 = 1e9
    for ep in range(1, 轮数 + 1):
        模型.train()
        批 = []
        for _ in range(每轮批):
            x = tr[np.random.randint(0, len(tr))]
            T = len(x["f0s"]); W = 800
            s = 0 if T <= W else np.random.randint(0, T - W)
            e = min(T, s + W)
            if e - s < 120:
                continue
            批.append(tuple(a[s:e] for a in (x["tok"], x["rel"], x["pos"], x["logT"], x["mel"], x["f0s"], x["v"])))
        W = min(len(b[0]) for b in 批)
        批 = [tuple(np.asarray(a)[:W] for a in b) for b in 批]

        def T_(arr):
            return torch.as_tensor(arr, dtype=torch.float32, device=设备)
        tok = torch.as_tensor(np.stack([b[0] for b in 批]).astype(np.int64), device=设备)
        rel = T_(np.stack([b[1] for b in 批])); pos = T_(np.stack([b[2] for b in 批]))
        logT = T_(np.stack([b[3] for b in 批]))
        mel = T_(np.stack([b[4] for b in 批]))
        f0 = T_(np.stack([b[5] for b in 批])); v = T_(np.stack([b[6] for b in 批]))
        pmel, pf0 = 模型(tok, rel, pos, logT)
        l_mel = ((pmel - mel).abs()).mean()
        l_f = (((pf0 - f0) ** 2) * v).sum() / (v.sum() + 1e-6)
        loss = l_mel + 1.0 * l_f
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(模型.parameters(), 3.0); opt.step()
        sched.step()
        if ep % 10 == 0:
            模型.eval()
            with torch.no_grad():
                vl = 0.0; cnt = 0
                for x in va:
                    T = len(x["f0s"])
                    if T < 120:
                        continue
                    Wv = min(800, T); s = max(0, (T - Wv) // 2); e = s + Wv
                    tokv = torch.as_tensor(x["tok"][s:e][None].astype(np.int64), device=设备)
                    relv = torch.as_tensor(x["rel"][s:e][None], dtype=torch.float32, device=设备)
                    posv = torch.as_tensor(x["pos"][s:e][None], dtype=torch.float32, device=设备)
                    logTv = torch.as_tensor(x["logT"][s:e][None], dtype=torch.float32, device=设备)
                    pm, pf = 模型(tokv, relv, posv, logTv)
                    mv = torch.as_tensor(x["mel"][s:e][None], dtype=torch.float32, device=设备)
                    fv = torch.as_tensor(x["f0s"][s:e][None], dtype=torch.float32, device=设备)
                    vv = torch.as_tensor(x["v"][s:e][None], dtype=torch.float32, device=设备)
                    vl += float(((pm - mv).abs().mean() + (((pf - fv) ** 2) * vv).sum() / (vv.sum() + 1e-6)).item())
                    cnt += 1
                vl /= max(1, cnt)
            print("| ep", ep, "loss %.4f" % float(loss.item()), "val %.4f" % vl, flush=True)
            if vl < 最佳:
                最佳 = vl
                torch.save({"模型": 模型.state_dict(), "val": vl}, 产 / "模型.pt")
    模型.eval()
    torch.save({"模型": 模型.state_dict(), "val": 最佳}, 产 / "模型.pt")
    print("训练完成 最佳val", 最佳, flush=True)
    return 0


def 包络(mel):
    功率 = np.maximum(np.exp(np.clip(np.asarray(mel, np.float64), -40, 40)) - 1e-6, 0.0)
    M = librosa.feature.inverse.mel_to_stft(功率.T, sr=SR, n_fft=1024, power=2.0, fmin=0.0, fmax=SR / 2.0)
    return np.maximum(np.asarray(M, np.float64), 0.0)


def 激励(f0帧, NR帧, 种子=20261022):
    T = len(f0帧); N = T * 220
    f0s = np.interp(np.arange(N) / 220, np.arange(T) + 0.5, f0帧)
    ph = np.mod(np.cumsum(f0s) / SR, 1.0)
    流, 导 = 器.声门表(0.62, 0.42)
    源 = np.interp(ph * 器._表长, np.arange(器._表长), 导)
    rng = np.random.RandomState(种子)
    噪 = rng.randn(N)
    噪hp = np.append(噪[0], 噪[1:] - 噪[:-1])
    NRs = np.interp(np.arange(N) / 220, np.arange(T) + 0.5, NR帧)
    return 源 * 0.6 + 噪hp * NRs * 0.8


def 合成mel(mel, f0g, NR, 迭代=32):
    T = len(mel)
    源 = 激励(f0g, NR)
    S = librosa.stft(源, n_fft=1024, hop_length=220, win_length=551, window="hann", center=True)
    M = 包络(mel)
    T2 = S.shape[1]
    if M.shape[1] != T2:
        idx = np.clip(np.round(np.linspace(0, M.shape[1] - 1, T2)).astype(int), 0, M.shape[1] - 1)
        M = M[:, idx]
    M = M * (float(np.mean(np.abs(源))) + 1e-9) * 8.0
    相位 = np.angle(S)
    y = None
    for _ in range(迭代):
        Y = M * np.exp(1j * 相位)
        y = librosa.istft(Y, hop_length=220, win_length=551, n_fft=1024, window="hann", center=True)
        y = np.nan_to_num(np.asarray(y, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        S2 = librosa.stft(y, n_fft=1024, hop_length=220, win_length=551, window="hann", center=True)
        if S2.shape[1] != T2:
            break
        相位 = np.angle(S2)
    p = float(np.abs(y).max())
    if p > 1e-9:
        y = y / p * 0.9
    return y


def 键_of(文本, 字表):
    from pypinyin import lazy_pinyin, Style
    字s = [ch for ch in 文本 if "\u4e00" <= ch <= "\u9fff"]
    out = []
    for ch in 字s:
        if ch in 字表:
            out.append(字表[ch]); continue
        ys = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)
        if not ys:
            continue
        s = ys[0]; tone = 5
        if s and s[-1].isdigit():
            tone = int(s[-1]); s = s[:-1]
        sm = "∅"; rest = s
        for ini in ("zh", "ch", "sh"):
            if s.startswith(ini):
                sm, rest = ini, s[len(ini):]; break
        if sm == "∅":
            for ini in "bpmfdtnlgkhjqxrzcs":
                if s.startswith(ini):
                    sm, rest = ini, s[1:]; break
        if sm in 声母表 and rest in 韵母表:
            out.append((声母表.index(sm), 韵母表.index(rest), max(0, min(4, tone - 1))))
    return out


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def 推理(每音节=24):
    with open(产 / "数据.pkl", "rb") as f:
        d = pickle.load(f)
    字表 = d["字表"]
    ck = torch.load(产 / "模型.pt", map_location=设备)
    模型 = 生成器().to(设备)
    模型.load_state_dict(ck["模型"]); 模型.eval()
    报 = dict(工具="训练_P6v2.py 推理", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    with torch.no_grad():
        for 名, 文 in 句表.items():
            keys = 键_of(文, 字表)
            toks = np.array(keys, np.int64)
            durs = np.full(len(toks), 每音节, np.int64)
            tok, rel, pos, logT = 构造帧输入(toks, durs)
            pm, pf = 模型(torch.as_tensor(tok[None].astype(np.int64), device=设备),
                          torch.as_tensor(rel[None], dtype=torch.float32, device=设备),
                          torch.as_tensor(pos[None], dtype=torch.float32, device=设备),
                          torch.as_tensor(logT[None], dtype=torch.float32, device=设备))
            mel = pm[0].cpu().numpy(); f0p = pf[0].cpu().numpy()
            f0 = 300.0 * (2.0 ** (np.clip(f0p, -12, 12) / 12.0))
            f0s = np.copy(f0)
            for i in range(len(f0s)):
                lo = max(0, i - 2); hi = min(len(f0s), i + 3)
                f0s[i] = float(np.median(f0[lo:hi]))
            mp = np.exp(np.clip(mel, -40, 40))
            flat = np.exp(np.mean(np.log(mp + 1e-10), 1)) / (np.mean(mp, 1) + 1e-10)
            NR = np.clip((flat - 0.10) / 0.60, 0.0, 1.0)
            y = 合成mel(mel, f0s, NR)
            写(产 / f"{名}.wav", y)
            报["项"].append(dict(样本=名, 文本=文, 路径=str(产 / f"{名}.wav"), 秒=round(len(y) / SR, 3)))
            print("|", 名, round(len(y) / SR, 2), "s", flush=True)
    引 = 量.载引擎()
    from faster_whisper import WhisperModel
    import glob
    snap = glob.glob(str(Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-medium" / "snapshots" / "*"))[0]
    wm = WhisperModel(snap, device="cpu", compute_type="int8")
    for x in 报["项"]:
        q = 引["转写"](x["路径"])
        segs, _ = wm.transcribe(x["路径"], language="zh", beam_size=5)
        wh = "".join(s.text for s in segs).strip()
        x.update(转写=q, Whisper=wh, CER=round(量.CER(x["文本"], q), 4))
        print("|", x["样本"], "CER=%.3f" % x["CER"], "|", q[:30], flush=True)
    r0 = [x for x in 报["项"] if "CER" in x]
    报["均值CER"] = round(float(np.mean([x["CER"] for x in r0])), 4) if r0 else None
    (产 / "推理_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("均值", 报["均值CER"], flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "推理":
        sys.exit(推理())
    sys.exit(训练(int(sys.argv[2]) if len(sys.argv) > 2 else 300,
               int(sys.argv[3]) if len(sys.argv) > 3 else 60))
