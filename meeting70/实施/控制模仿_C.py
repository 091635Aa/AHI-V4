# -*- coding: utf-8 -*-
"""meeting70/实施/控制模仿_C.py —— C 步：控制域模仿（拼控制帧，不拼波形）。

思路：D0（≤5 min）里按音节切出**控制轨迹**（F0/浊音/共振峰/噪声/能量/24 阶 LPC 包络），
按字表检索拼接控制帧，平滑后驱动**器官 v2.1** 说任意句。零训练；不拼任何采样波形。
注：A 门未过（0.3212>0.25），本件按「信息臂：定位瓶颈」跑，**不声称任何门通过**。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M70 = HERE.parent
M66 = M70.parent / "meeting66"
M67 = M70.parent / "meeting67"
出 = M70 / "产物"
SR = 22050
D0目录 = M67 / "数据" / "子集_D0_le5min" / "样本"
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", HERE / "器官v2.py")
协 = 载("合成_S0刺激", M70.parent / "meeting69" / "实施" / "合成_S0刺激.py")


def 建库_控制():
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    文 = {r["id"]: r["文"] for r in meta["样本"]}
    库 = []
    wav缓存 = {}
    for f in sorted(D0目录.glob("*.npz")):
        sid = int(f.stem)
        z = np.load(f, allow_pickle=True)
        if sid not in wav缓存:
            wav缓存[sid] = 器.耳.读wav(音频根 / 样本[sid]["wav"])
        c = 器.分析控制(wav缓存[sid])
        帧数 = z["音节帧数"]; 声 = z["声母"]; 韵 = z["韵母"]; 调 = z["调"]
        cum = 0
        for i in range(len(声)):
            dur = int(帧数[i])
            a = cum * 2; b = (cum + dur) * 2          # 10 ms 帧 → 5 ms 控制帧
            cum += dur
            if b > c["T"] or b - a < 8:
                continue
            ch = 文[sid][i] if i < len(文[sid]) else ""
            库.append(dict(sid=sid, i=i, ch=ch, sm=int(声[i]), ym=int(韵[i]), tone=int(调[i]),
                           f0=c["f0"][a:b].copy(), 浊=c["浊"][a:b].copy(),
                           F=c["F"][a:b].copy(), B=c["B"][a:b].copy(),
                           E=c["E"][a:b].copy(), NR=c["NR"][a:b].copy(), A=c["A"][a:b].copy(),
                           秒=(b - a) * 0.005))
    return 库, 样本, 文


def 建字表(库, 文):
    from collections import Counter
    表 = {}
    for u in 库:
        ch = u["ch"]
        if not ("\u4e00" <= ch <= "\u9fff"):
            continue
        表.setdefault(ch, []).append((u["sm"], u["ym"], u["tone"]))
    return {ch: Counter(v).most_common(1)[0][0] for ch, v in 表.items()}


def 选单元(库, 键, ch):
    sm, ym, tone = 键
    候 = [u for u in 库 if u["sm"] == sm and u["ym"] == ym and u["tone"] == tone and u["ch"] == ch]
    if not 候:
        候 = [u for u in 库 if u["sm"] == sm and u["ym"] == ym and u["tone"] == tone]
    if not 候:
        候 = [u for u in 库 if u["sm"] == sm and u["ym"] == ym]
    if not 候:
        return None
    候.sort(key=lambda u: abs(u["秒"] - 0.24))
    return 候[0]


def 合成句_控制(文本, 参数, 库=None, 样本=None, 文=None, 字表=None, 逐音节=False):
    if 库 is None:
        库, 样本, 文 = 建库_控制()
        字表 = 建字表(库, 文)
    from pypinyin import lazy_pinyin, Style
    # 字表优先；生字用 pypinyin 三元组（语料 调 id 0..4）
    声母表 = ["∅", "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]
    韵母表 = ["a", "o", "e", "i", "u", "v", "er", "ii", "iii", "ai", "ei", "ao", "ou", "ia", "ie", "iao", "iou", "ua", "uo",
             "uai", "uei", "ve", "an", "en", "in", "uen", "vn", "ang", "eng", "ing", "ong", "iong", "ian", "uan", "uang",
             "van", "", "iang", "io"]
    字s = [ch for ch in 文本 if "\u4e00" <= ch <= "\u9fff"]
    n = len(字s)
    segs = []; 状态 = []
    for j, ch in enumerate(字s):
        if ch in 字表:
            键 = 字表[ch]; 来源 = "字表"
        else:
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
            if sm not in 声母表 or rest not in 韵母表:
                continue
            键 = (声母表.index(sm), 韵母表.index(rest), max(0, min(4, tone - 1)))
            来源 = "pinyin"
        u = 选单元(库, 键, ch)
        if u is None:
            状态.append((ch, "缺")); continue
        pos = j / max(n - 1, 1)
        f0 = u["f0"].copy()
        # 句内韵律：整体 F0 线性下降 1.2 半音（不改单元内部轮廓）
        f0 = f0 * (2.0 ** ((-1.2 * pos) / 12.0))
        E = u["E"].copy() * (1.0 - 0.25 * pos)
        segs.append(dict(f0=f0, 浊=u["浊"], F=u["F"], B=u["B"], E=E, NR=u["NR"], A=u["A"]))
        状态.append((ch, 来源))
    if not segs:
        return None, 状态
    if 逐音节:
        # 每个音节单独过器官（避免跨音节 LPC 系数跳变导致失稳），再 10 ms 交叉淡化
        ys = []
        for s in segs:
            cc = dict(T=len(s["f0"]), f0=s["f0"].copy(), 浊=np.asarray(s["浊"]).astype(bool),
                      F=s["F"].copy(), B=s["B"].copy(), E=np.maximum(s["E"], 1e-5),
                      NR=s["NR"].copy(), A=s["A"].copy())
            ys.append(器.合成(cc, 模式="全极点", **参数))
        nf = int(SR * 0.010)
        out = ys[0].copy()
        for y in ys[1:]:
            if len(out) > nf and len(y) > nf:
                w = np.linspace(0, 1, nf, dtype=np.float32)
                out[-nf:] = out[-nf:] * (1 - w) + y[:nf] * w
                out = np.concatenate([out, y[nf:]])
            else:
                out = np.concatenate([out, y])
        p = float(np.abs(out).max())
        if p > 1e-9:
            out = out / p * 0.9
        return out.astype(np.float32), 状态
    def 拼(k):
        return np.concatenate([s[k] for s in segs], axis=0)
    c = dict(T=sum(len(s["f0"]) for s in segs), f0=拼("f0"), 浊=拼("浊").astype(bool),
             F=拼("F"), B=拼("B"), E=np.maximum(拼("E"), 1e-5), NR=拼("NR"), A=拼("A"))
    # F0 交界平滑（±1 帧）
    f = c["f0"].copy()
    for i in range(1, len(f) - 1):
        if c["浊"][i]:
            f[i] = 0.5 * f[i] + 0.25 * f[i - 1] + 0.25 * f[i + 1]
    c["f0"] = f
    y = 器.合成(c, 模式="全极点", **参数)
    return y, 状态


def 主(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    出.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    扫 = 出 / "G0v3R1_扫参.json"
    参数 = dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)
    if 扫.exists():
        d = json.loads(扫.read_text(encoding="utf-8"))
        参数 = sorted(d["臂"], key=lambda r: r["均值CER"])[0]["参数"]
    库, 样本, 文 = 建库_控制()
    字表 = 建字表(库, 文)
    句表 = {
        "C主句": "今天天气很好我听见了自己的声音我想再说一遍",
        "C新句1": "今天的风很轻我慢慢说给你听",
        "C新句2": "窗外的雨停了我该休息一会",
        "C新句3": "你先别走我还想再说两句",
    }
    报 = dict(工具="控制模仿_C.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              参数=参数, 说明="A 门未过（0.3212>0.25）⇒ 本件为信息臂：定位瓶颈在控制拼接还是器官", 项=[])
    for 名, 文本 in 句表.items():
        y, 状态 = 合成句_控制(文本, 参数, 库, 样本, 文, 字表)
        if y is None:
            报["项"].append(dict(样本=名, 失败="无段")); continue
        p = 出 / f"{名}.wav"
        协.写wav(p, y)
        报["项"].append(dict(样本=名, 文本=文本, 路径=str(p), 秒=round(len(y) / SR, 3),
                            缺=sum(1 for _, s in 状态 if s == "缺"), 状态=状态))
        print("|", 名, round(len(y) / SR, 2), "s 缺", 报["项"][-1]["缺"], flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "C_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
