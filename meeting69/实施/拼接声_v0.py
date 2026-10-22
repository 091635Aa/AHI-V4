# -*- coding: utf-8 -*-
"""meeting69/实施/拼接声_v0.py —— 拼接式「他的真人声」：从 167 条真人语料里按音节检索、拼接。

背景（`诊断_声码器保真.json` + `听审/Omni听审*.json`）：
  · 真人 mel→GL 被 CAM++ 判 0.9285、被 Qwen3-Omni 判「真人、正常童声」（声码器无罪）；
  · v13 生成的 wav 被 Omni 听成「类似猪叫的怪叫声」、CAM++ 相似度 0.13–0.23（模型 mel 是病灶）。
本件**绕开声学模型**：直接取真人录音的音节片段拼接——音色/清晰度来自他本人。
数据：`meeting66/产物/数据_对齐asr`（npz 逐音节 声母/韵母/调/帧数）+ `meeting67/产物/P2/对齐证据.json`
（逐条 mel↔整条 wav 的整数帧偏移，148/167 非零，残余 ≤5 ms）+ `<path>\\SpeakerA7代模型训练数据\\*.wav`。
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
M66 = M69.parent / "meeting66"
M67 = M69.parent / "meeting67"
出 = M69 / "产物" / "拼接"
SR = 22050
帧 = 220


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


耳 = 载("耳蜗前端_S0_2", HERE / "耳蜗前端_S0_2.py")
协 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")

# ── pypinyin → 语料词表 ──
from pypinyin import lazy_pinyin, Style

声母表 = ["∅", "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]
韵母表 = ["a", "o", "e", "i", "u", "v", "er", "ii", "iii", "ai", "ei", "ao", "ou", "ia", "ie", "iao", "iou", "ua", "uo",
        "uai", "uei", "ve", "an", "en", "in", "uen", "vn", "ang", "eng", "ing", "ong", "iong", "ian", "uan", "uang",
        "van", "", "iang", "io"]
声母id = {s: i for i, s in enumerate(声母表)}
韵母id = {s: i for i, s in enumerate(韵母表)}


def 音节到三元组(音节):
    """'jin1' → ('j','in',1)；处理 y/w 与 ii/iii/iu/ui/un 等。"""
    s = 音节.strip().lower()
    tone = 5
    if s and s[-1].isdigit():
        tone = int(s[-1]); s = s[:-1]
    # y/w 归一
    if s.startswith("y"):
        r = s[1:]
        if r.startswith("u"):
            s = "v" + r[1:]
        elif r in ("i",):
            s = "i"
        elif r.startswith("i"):
            s = r
        elif r == "ue":
            s = "ve"
        elif r == "uan":
            s = "van"
        elif r == "un":
            s = "vn"
        elif r == "ing":
            s = "ing"
        elif r == "ong":
            s = "iong"
        elif r == "":
            s = "i"
    elif s.startswith("w"):
        r = s[1:]
        if r == "u":
            s = "u"
        elif r in ("a", "o", "ai", "ei", "an", "en", "ang", "eng"):
            mp = {"a": "ua", "o": "uo", "ai": "uai", "ei": "uei", "an": "uan", "en": "uen", "ang": "uang", "eng": "ong"}
            s = mp[r]
        else:
            s = "u" + r
    # 声母
    sm = "∅"; rest = s
    for ini in ("zh", "ch", "sh"):
        if s.startswith(ini):
            sm, rest = ini, s[len(ini):]; break
    if sm == "∅":
        for ini in "bpmfdtnlgkhjqxrzcs":
            if s.startswith(ini):
                sm, rest = ini, s[1:]; break
    ym = rest
    # 韵母归一
    mp = {"iu": "iou", "ui": "uei", "un": "uen", "ü": "v", "üe": "ve", "üan": "van", "ün": "vn",
          "ve": "ve", "van": "van", "vn": "vn", "ue": "ve"}
    ym = mp.get(ym, ym)
    if ym == "i":
        if sm in ("zh", "ch", "sh", "r"):
            ym = "ii"
        elif sm in ("z", "c", "s"):
            ym = "iii"
    if ym == "i̇":
        ym = "i"
    return sm, ym, tone


def 建索引():
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    对齐 = json.loads((M67 / "产物" / "P2" / "对齐证据.json").read_text(encoding="utf-8"))
    偏移 = {r["id"]: int(r["偏移帧"]) for r in 对齐["per_sample"]}
    样本 = {r["id"]: r for r in meta["样本"]}
    索引 = {}
    for sid in sorted(样本):
        z = np.load(M66 / "产物" / "数据_对齐asr" / "样本" / 样本[sid]["文件"], allow_pickle=True)
        声 = z["声母"]; 韵 = z["韵母"]; 调 = z["调"]; 帧数 = z["音节帧数"]
        起 = 0
        for i in range(len(声)):
            dur = int(帧数[i])
            key = (int(声[i]), int(韵[i]), int(调[i]))
            索引.setdefault(key, []).append(dict(sid=sid, i=i, 起帧=起, 止帧=起 + dur, 帧数=dur))
            起 += dur
    return 索引, 偏移, 样本


_wav缓存 = {}


def 取wav(sid, 样本):
    if sid not in _wav缓存:
        p = Path(r"<path>\SpeakerA7代模型训练数据") / 样本[sid]["wav"]
        _wav缓存[sid] = 耳.读wav(p)
    return _wav缓存[sid]


def 切段(sid, 起帧, 止帧, 偏移, 样本, 余量帧=1):
    x = 取wav(sid, 样本)
    a = max(0, (偏移 + 起帧 - 余量帧)) * 帧
    b = min(len(x), (偏移 + 止帧 + 余量帧) * 帧)
    seg = np.asarray(x[a:b], np.float32)
    if seg.size < 帧 * 4:
        return None
    # 能量裁边
    w = int(0.005 * SR)
    e = np.array([np.sqrt(np.mean(seg[i:i + w] ** 2)) for i in range(0, max(1, len(seg) - w), w)])
    if e.size == 0:
        return None
    thr = max(0.06 * float(e.max()), 1e-4)
    idx = np.where(e > thr)[0]
    if idx.size == 0:
        return None
    a2 = max(0, idx[0] * w - w)
    b2 = min(len(seg), (idx[-1] + 2) * w)
    seg = seg[a2:b2]
    # 轻淡入淡出 3 ms
    n = int(0.003 * SR)
    if len(seg) > 2 * n > 0:
        seg[:n] *= np.linspace(0, 1, n)
        seg[-n:] *= np.linspace(1, 0, n)
    return seg


def 选段(索引, sm, ym, tone, 期望帧=24):
    # 语料 调 id = 0..4（实测），pypinyin 声调 1..4、轻声 5 ⇒ 转 0..4
    调id = (int(tone) - 1) if 1 <= int(tone) <= 5 else 4
    if 调id > 4:
        调id = 4
    键表 = [(声母id[sm], 韵母id[ym], 调id), (声母id[sm], 韵母id[ym], None)]
    for k_sm, k_ym, k_tone in 键表:
        cands = []
        for (s0, y0, t0), lst in 索引.items():
            if s0 != k_sm or y0 != k_ym:
                continue
            if k_tone is not None and t0 != k_tone:
                continue
            cands += lst
        if not cands:
            continue
        # 偏好：时长接近期望、句中（i 不首不尾）
        def 分(c):
            return abs(c["帧数"] - 期望帧) + (6 if c["i"] == 0 else 0)
        cands.sort(key=分)
        return cands[0], ("同调" if k_tone is not None else "异调兜底")
    return None, "缺"


def 合成句(文本, 期望帧=24, 交叉毫秒=8):
    音节s = lazy_pinyin(文本, style=Style.TONE3, neutral_tone_with_five=True)
    索引, 偏移, 样本 = 建索引()
    segs = []; 明细 = []
    for 音 in 音节s:
        sm, ym, tone = 音节到三元组(音)
        c, 别 = 选段(索引, sm, ym, tone, 期望帧)
        if c is None:
            明细.append(dict(音节=音, 声母=sm, 韵母=ym, 调=tone, 状态="缺"))
            continue
        seg = 切段(c["sid"], c["起帧"], c["止帧"], 偏移[c["sid"]], 样本)
        if seg is None:
            明细.append(dict(音节=音, 状态="切失败")); continue
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        seg = seg / rms * 0.06
        segs.append(seg)
        明细.append(dict(音节=音, 声母=sm, 韵母=ym, 调=tone, 状态=别, sid=c["sid"], 段=c["i"],
                       帧数=c["帧数"], 秒=round(len(seg) / SR, 3)))
    if not segs:
        return None, 明细
    n = int(SR * 交叉毫秒 / 1000)
    out = segs[0].copy()
    for s in segs[1:]:
        if len(out) > n and len(s) > n:
            w = np.linspace(0, 1, n, dtype=np.float32)
            out[-n:] = out[-n:] * (1 - w) + s[:n] * w
            out = np.concatenate([out, s[n:]])
        else:
            out = np.concatenate([out, s])
    p = float(np.abs(out).max())
    if p > 1e-9:
        out = out / p * 0.9
    return out.astype(np.float32), 明细


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    出.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    文 = "今天天气很好我听见了自己的声音我想再说一遍"
    x, 明细 = 合成句(文)
    报 = dict(工具="拼接声_v0.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 文本=文, 明细=明细)
    if x is not None:
        路 = 出 / "拼接_一个完整的话.wav"
        协.写wav(路, x)
        报["路径"] = str(路)
        报["时长秒"] = round(len(x) / SR, 3)
        报["缺音节数"] = sum(1 for d in 明细 if d["状态"] in ("缺", "切失败"))
    (出 / "拼接_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报, ensure_ascii=False, indent=1)[:3000])
    print("耗时", round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
