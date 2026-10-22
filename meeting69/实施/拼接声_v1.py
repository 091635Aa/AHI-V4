# -*- coding: utf-8 -*-
"""meeting69/实施/拼接声_v1.py —— T1 S1：D0(≤5min) 单元库 + 检索 + 调型 PSOLA + 韵律模板。

零训练：不更新任何权重、不拟合网络；只用 D0 的 5 条真人录音切片。
流程：
  1) 单元库：D0 npz（声母/韵母/调/帧数）+ 对齐偏移 → 从 L: 真人 wav 切音节，记 F0/能量轮廓；
  2) 检索：同(声母,韵母,调) → 同音节任意调 → 同韵母；代价=|时长差|+λ1·调型距离+λ2·边界能量；
  3) 韵律：句内统计模板（句首略高、句末下降）给每个音节目标 F0 中位；调型模板给 F0 轮廓；
  4) PSOLA：按目标 F0/目标时长重合成每个音节（保共振峰），再交叉淡化拼接。
产物：产物/拼接v1/拼接_<句号>.wav + 明细.json
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
M66 = M69.parent / "meeting66"
M67 = M69.parent / "meeting67"
出 = M69 / "产物" / "拼接v1"
SR = 22050
帧 = 220
D0目录 = M67 / "数据" / "子集_D0_le5min" / "样本"
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


耳 = 载("耳蜗前端_S0_2", HERE / "耳蜗前端_S0_2.py")
协 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")

声母表 = ["∅", "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]
韵母表 = ["a", "o", "e", "i", "u", "v", "er", "ii", "iii", "ai", "ei", "ao", "ou", "ia", "ie", "iao", "iou", "ua", "uo",
        "uai", "uei", "ve", "an", "en", "in", "uen", "vn", "ang", "eng", "ing", "ong", "iong", "ian", "uan", "uang",
        "van", "", "iang", "io"]
声母id = {s: i for i, s in enumerate(声母表)}
韵母id = {s: i for i, s in enumerate(韵母表)}

# 调型模板（标准普通话单字调，F0 半音相对中位；先写死，非训练）
调型 = {
    0: np.array([0.0, 0.4, 0.7, 0.9]),                    # T1 平/高平
    1: np.array([-0.6, -0.2, 0.5, 1.0]),                  # T2 升
    2: np.array([0.2, -0.6, -0.8, 0.6]),                  # T3 降升
    3: np.array([1.0, 0.4, -0.4, -0.9]),                  # T4 降
    4: np.array([0.0, 0.0, 0.0, 0.0]),                    # 轻声/中性（近中位）
}


def 音节到三元组(音节):
    s = 音节.strip().lower()
    tone = 5
    if s and s[-1].isdigit():
        tone = int(s[-1]); s = s[:-1]
    if s.startswith("y"):
        r = s[1:]
        if r.startswith("u"):
            s = "v" + r[1:]
        elif r == "i":
            s = "i"
        elif r.startswith("i"):
            s = r
        elif r == "ue":
            s = "ve"
        elif r == "uan":
            s = "van"
        elif r == "un":
            s = "vn"
        elif r == "ong":
            s = "iong"
        elif r == "":
            s = "i"
    elif s.startswith("w"):
        r = s[1:]
        mp = {"u": "u", "a": "ua", "o": "uo", "ai": "uai", "ei": "uei",
              "an": "uan", "en": "uen", "ang": "uang", "eng": "ong"}
        s = mp.get(r, "u" + r)
    sm = "∅"; rest = s
    for ini in ("zh", "ch", "sh"):
        if s.startswith(ini):
            sm, rest = ini, s[len(ini):]; break
    if sm == "∅":
        for ini in "bpmfdtnlgkhjqxrzcs":
            if s.startswith(ini):
                sm, rest = ini, s[1:]; break
    ym = rest
    mp = {"iu": "iou", "ui": "uei", "un": "uen", "ü": "v", "üe": "ve", "üan": "van", "ün": "vn",
          "ve": "ve", "van": "van", "vn": "vn", "ue": "ve"}
    ym = mp.get(ym, ym)
    if ym == "i":
        if sm in ("zh", "ch", "sh", "r"):
            ym = "ii"
        elif sm in ("z", "c", "s"):
            ym = "iii"
    调id = (tone - 1) if 1 <= tone <= 5 else 4
    if 调id > 4:
        调id = 4
    return sm, ym, tone, 调id


def 轮廓(seg):
    """→ (F0 半音相对中位轮廓, 浊音比例, 中位F0)。"""
    seg = np.asarray(seg, np.float32)
    if len(seg) < int(0.045 * SR):        # 短于 F0 分析窗
        return None, 0.0, 0.0
    t, f0 = 耳.F0轨迹(seg)
    v = np.asarray(f0, np.float64)
    if np.sum(v > 0) < 3:
        return None, 0.0, 0.0
    基 = np.median(v[v > 0])
    s = np.zeros_like(v)
    s[v > 0] = 12.0 * np.log2(v[v > 0] / 基)
    return s, float(np.mean(v > 0)), float(基)


def 调型距离(s, tid):
    """轮廓 vs 调型模板的距离（都归一化到 [-1,1] 的 4 点形状）。"""
    if s is None or len(s[s > 0]) < 3:
        return 9.9
    ok = np.where(s > 0)[0] if False else np.arange(len(s))
    ss = s.copy()
    # 只取有声段重采样成 4 点
    idx = np.where(ss != 0)[0]
    if len(idx) < 4:
        return 9.9
    v = ss[idx]
    xp = np.linspace(0, 1, len(v))
    q = np.interp(np.linspace(0, 1, 4), xp, v)
    q = q - q.mean()
    尺 = np.max(np.abs(q)) + 1e-9
    q = q / 尺
    tpl = 调型[tid]
    tpl = tpl - tpl.mean()
    tpl = tpl / (np.max(np.abs(tpl)) + 1e-9)
    return float(np.mean((q - tpl) ** 2))


def 边界谷(x, pos, 宽秒=0.045, 步=0.01):
    """在 pos±宽秒 内找能量最低点（10 ms 窗），返回样本下标——用来把 ASR 字符边界修到声学谷。"""
    w = int(步 * SR)
    lo = max(0, int(pos - 宽秒 * SR)); hi = min(len(x) - w, int(pos + 宽秒 * SR))
    if hi <= lo + w:
        return int(pos)
    e = np.array([np.sqrt(np.mean(np.asarray(x[i:i + w], np.float64) ** 2)) for i in range(lo, hi, w)])
    return int(lo + int(np.argmin(e)) * w)


def 吸附谷(e, pos, 宽帧=12):
    """把帧下标 pos 吸附到 ±宽帧(10 ms) 内最近的**局部能量谷**（严格局部极小），否则回原位。"""
    lo = max(1, int(pos) - int(宽帧)); hi = min(len(e) - 2, int(pos) + int(宽帧))
    if hi <= lo:
        return int(pos)
    候 = [j for j in range(lo, hi) if e[j] <= e[j - 1] and e[j] <= e[j + 1]]
    if not 候:
        return int(pos)
    return int(min(候, key=lambda j: abs(j - int(pos))))


def 吸附峰(flux, pos, 宽帧=12):
    """把边界吸附到 ±宽帧 内**谱通量最大**处（真正的音节边界更可能在谱变化峰上）。"""
    lo = max(1, int(pos) - int(宽帧)); hi = min(len(flux) - 2, int(pos) + int(宽帧))
    if hi <= lo:
        return int(pos)
    return int(lo + int(np.argmax(flux[lo:hi])))


def 建库():
    对齐 = json.loads((M67 / "产物" / "P2" / "对齐证据.json").read_text(encoding="utf-8"))
    偏移 = {r["id"]: int(r["偏移帧"]) for r in 对齐["per_sample"]}
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    库 = []
    wav缓存 = {}
    for f in sorted(D0目录.glob("*.npz")):
        sid = int(f.stem)
        z = np.load(f, allow_pickle=True)
        if sid not in wav缓存:
            wav缓存[sid] = 耳.读wav(音频根 / 样本[sid]["wav"])
        x = wav缓存[sid]
        声, 韵, 调id数组, 帧数 = z["声母"], z["韵母"], z["调"], z["音节帧数"]
        mel = np.asarray(z["mel"], np.float64)
        flux = np.sqrt(np.mean(np.diff(mel, axis=0) ** 2, axis=1)) if len(mel) > 2 else np.zeros(1)
        # 全条 10 ms 能量包络（兜底）
        n帧 = len(x) // 帧
        e全 = np.array([np.sqrt(np.mean(np.asarray(x[j * 帧:(j + 1) * 帧], np.float64) ** 2)) for j in range(n帧)])
        cum = 0
        for i in range(len(声)):
            dur = int(帧数[i])
            起 = 偏移[sid] + 吸附峰(flux, cum)
            止 = 偏移[sid] + 吸附峰(flux, cum + dur)
            if 止 - 起 < 4:
                起, 止 = 偏移[sid] + cum, 偏移[sid] + cum + dur
            a = max(0, (起 - 1)) * 帧
            b = min(len(x), (止 + 1) * 帧)
            seg = np.asarray(x[a:b], np.float32)
            cum += dur
            if seg.size < 帧 * 6:
                continue
            # 能量裁边
            w = int(0.005 * SR)
            e = np.array([np.sqrt(np.mean(seg[j:j + w] ** 2)) for j in range(0, max(1, len(seg) - w), w)])
            if e.size == 0:
                continue
            thr = max(0.05 * float(e.max()), 1e-4)
            ok = np.where(e > thr)[0]
            if ok.size == 0:
                continue
            seg = seg[max(0, ok[0] * w - w):min(len(seg), (ok[-1] + 2) * w)]
            轮, 浊比, 基 = 轮廓(seg)
            库.append(dict(sid=sid, i=i, sm=int(声[i]), ym=int(韵[i]), tone=int(调id数组[i]),
                          秒=round(len(seg) / SR, 3), F0=基, 轮廓=轮, 浊比=浊比, 波形=seg))
    return 库, 偏移, 样本


def 选段(库, smid, ymid, tid, 目标秒):
    键 = [(smid, ymid, tid), (smid, ymid, None)]
    for k_sm, k_ym, k_t in 键:
        候 = [u for u in 库 if u["sm"] == k_sm and u["ym"] == k_ym and (k_t is None or u["tone"] == k_t)]
        if not 候:
            continue
        # 同音节内：时长接近优先、浊音多的优先（不用 F0 轮廓项选段，避免选到怪上下文）
        候.sort(key=lambda u: abs(u["秒"] - 目标秒) + (0.15 if u["浊比"] < 0.5 else 0.0))
        return 候[0], ("同调" if k_t is not None else "异调兜底")
    return None, "缺"


def psola(seg, 目标F0_轮廓, 目标秒):
    """TD-PSOLA：按目标 F0 轮廓与时长重合成（保共振峰）。失败则原样 + 重采样。"""
    seg = np.asarray(seg, np.float64)
    原秒 = len(seg) / SR
    if 原秒 <= 0:
        return np.zeros(1, np.float32)
    if len(seg) < int(0.045 * SR):
        return _ola(seg, 目标秒)
    t, f0 = 耳.F0轨迹(seg.astype(np.float32))
    v = np.asarray(f0, np.float64)
    if np.sum(v > 0) < 4:
        # 清音段：只做时长伸缩（OLA）
        return _ola(seg, 目标秒)
    # 原 F0 逐样本（浊音）；清音用邻近值
    f0s = np.interp(np.arange(len(seg)) / SR, t, v)
    f0s = np.where(f0s > 0, f0s, np.nan)
    # 填充清音
    idx = np.where(~np.isnan(f0s))[0]
    f0s = np.interp(np.arange(len(seg)), idx, f0s[idx])
    # 分析基音标记：从最大峰起，按周期步进，局部取峰
    标 = []
    pos = int(np.argmax(np.abs(seg[: int(0.05 * SR)])))
    while pos < len(seg) - 1:
        标.append(pos)
        p = int(round(SR / max(f0s[pos], 60.0)))
        下 = pos + p
        if 下 >= len(seg):
            break
        搜0, 搜1 = max(pos + int(0.6 * p), 0), min(pos + int(1.5 * p), len(seg) - 1)
        if 搜1 <= 搜0:
            pos = 下; continue
        pos = 搜0 + int(np.argmax(np.abs(seg[搜0:搜1])))
    if len(标) < 4:
        return _ola(seg, 目标秒)
    标 = np.asarray(标, np.float64)
    # 目标合成标记：按目标 F0 包络、目标时长铺（2 周期 Hann 颗粒；浮点位置避免累积漂移）
    目标长 = int(max(1, round(目标秒 * SR)))
    xs = np.linspace(0, 1, len(目标F0_轮廓))
    目标半音 = np.interp(np.linspace(0, 1, 目标长), xs, 目标F0_轮廓)
    中位 = float(np.median(f0s[标.astype(int)]))
    目标f0样 = 中位 * (2.0 ** (目标半音 / 12.0))
    y = np.zeros(目标长 + 4 * int(0.02 * SR), np.float64)
    权重 = np.zeros_like(y)
    ratioT = (len(seg) - 1) / max(目标长 - 1, 1)
    出pos = float(标[0])
    i = 0
    while 出pos < 目标长 - 1:
        分析时刻 = 出pos * ratioT
        近 = int(np.searchsorted(标, 分析时刻))
        近 = min(max(近, 1), len(标) - 1)
        if abs(标[近 - 1] - 分析时刻) < abs(标[近] - 分析时刻):
            近 -= 1
        p原 = SR / max(f0s[int(标[近])], 60.0)
        半 = max(2, int(round(p原)))
        a = int(round(标[近])) - 半; b = int(round(标[近])) + 半 + 1
        a = max(0, a); b = min(len(seg), b)
        if b - a < 8:
            出pos += SR / max(目标f0样[int(min(出pos, 目标长 - 1))], 60.0)
            continue
        w = np.hanning(b - a)
        片 = seg[a:b] * w
        c = int(round(出pos)) - (b - a) // 2
        s0 = max(0, c); e0 = min(len(y), s0 + len(片))
        y[s0:e0] += 片[: e0 - s0]
        权重[s0:e0] += w[: e0 - s0]
        出pos += SR / max(目标f0样[int(min(出pos, 目标长 - 1))], 60.0)
        i += 1
        if i > 40000:
            break
    m = 权重 > 1e-6
    y[m] /= 权重[m]
    y = y[:目标长]
    p = float(np.abs(y).max())
    if p > 1e-9:
        y = y / p * 0.9
    return y.astype(np.float32)


def _ola(seg, 目标秒):
    """清音/兜底：简单 OLA 时长伸缩。"""
    seg = np.asarray(seg, np.float64)
    n = max(1, int(round(目标秒 * SR)))
    if n == len(seg):
        return seg.astype(np.float32)
    xp = np.linspace(0, 1, len(seg))
    return np.interp(np.linspace(0, 1, n), xp, seg).astype(np.float32)


def 建字表(库):
    """从 D0 语料的「字 ↔ npz 三元组」直接建字表（词音外采，非训练）。

    ASR/对齐把半个音节的韵母归一化得与标准 pinyin 不一致（例：直→zh+iii、泼→p+uo、
    自→z+ii），用 pypinyin 反查会大面积错配——直接读语料自带的字表最稳。"""
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    文 = {r["id"]: r["文"] for r in meta["样本"]}
    字表 = {}
    for u in 库:
        sid = u["sid"]; i = u["i"]
        s = 文.get(sid, "")
        if i >= len(s):
            continue
        ch = s[i]
        if not ("\u4e00" <= ch <= "\u9fff"):
            continue
        字表.setdefault(ch, []).append((u["sm"], u["ym"], u["tone"]))
    # 每个字取众数
    out = {}
    for ch, lst in 字表.items():
        from collections import Counter
        out[ch] = Counter(lst).most_common(1)[0][0]
    return out


def 字到键(字表, ch):
    if ch in 字表:
        return 字表[ch], "字表"
    from pypinyin import lazy_pinyin, Style
    ys = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)
    if not ys:
        return None, "非汉"
    sm, ym, tone, tid = 音节到三元组(ys[0])
    return (声母id[sm], 韵母id[ym], tid), "pinyin"


def 合成句(文本, 基F0=1.0, 语速=1.0, 停顿毫秒=80, 用PSOLA=True):
    库, 偏移, 样本 = 建库()
    字表 = 建字表(库)
    字s = [ch for ch in 文本 if "\u4e00" <= ch <= "\u9fff"]
    n = len(字s)
    segs = []; 明细 = []
    for j, ch in enumerate(字s):
        键, 来源 = 字到键(字表, ch)
        if 键 is None:
            明细.append(dict(字=ch, 状态="缺")); continue
        smid, ymid, tid = 键
        目标秒 = 0.24 * 语速
        目标秒 = max(0.14, 目标秒)
        u, 别 = 选段(库, smid, ymid, tid, 目标秒)
        if u is None:
            明细.append(dict(字=ch, 状态="缺")); continue
        # 句内韵律：句首 +0.6 半音、句末 −1.2 半音、线性下降
        pos = j / max(n - 1, 1)
        基 = 0.6 * (1 - pos) - 1.2 * pos
        轮廓目标 = 调型[tid] + 基
        if 用PSOLA:
            y = psola(u["波形"], 轮廓目标, 目标秒)
        else:
            y = np.asarray(u["波形"], np.float32)     # 关闭 PSOLA：原段直拼（时长随单元）
        # 能量归一（每单元到同一 RMS，保留句内相对起伏）
        r = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-9
        y = (y / r * 0.05).astype(np.float32)
        # 能量：句内略降
        y = y * (1.0 - 0.15 * pos)
        segs.append(y)
        明细.append(dict(字=ch, 状态=别, 来源=来源, sid=u["sid"], 段=u["i"], 原秒=u["秒"],
                         目标秒=round(目标秒, 3), 调=u["tone"], 音节调=tid))
    if not segs:
        return None, 明细
    nf = int(SR * 0.008)
    out = segs[0].copy()
    for s in segs[1:]:
        if len(out) > nf and len(s) > nf:
            w = np.linspace(0, 1, nf, dtype=np.float32)
            out[-nf:] = out[-nf:] * (1 - w) + s[:nf] * w
            out = np.concatenate([out, s[nf:]])
        else:
            out = np.concatenate([out, s])
    # 句末收尾
    p = float(np.abs(out).max())
    if p > 1e-9:
        out = out / p * 0.9
    return out.astype(np.float32), 明细


def 主(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    出.mkdir(parents=True, exist_ok=True)
    文 = sys.argv[1] if len(sys.argv) > 1 else "今天天气很好我听见了自己的声音我想再说一遍"
    名 = sys.argv[2] if len(sys.argv) > 2 else "主句"
    用PSOLA = not (len(sys.argv) > 3 and sys.argv[3] == "无PSOLA")
    t0 = time.time()
    x, 明细 = 合成句(文, 用PSOLA=用PSOLA)
    报 = dict(工具="拼接声_v1.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 文本=文, 样本=名,
              PSOLA=用PSOLA, 明细=明细)
    if x is not None:
        路 = 出 / f"拼接_{名}.wav"
        协.写wav(路, x)
        报["路径"] = str(路); 报["时长秒"] = round(len(x) / SR, 3)
        报["缺"] = sum(1 for d in 明细 if d["状态"] == "缺")
    (出 / f"明细_{名}.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in 报.items() if k != "明细"}, ensure_ascii=False))
    print("状态统计", {s: sum(1 for d in 明细 if d["状态"] == s) for s in set(d["状态"] for d in 明细)}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
