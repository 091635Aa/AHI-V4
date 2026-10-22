# -*- coding: utf-8 -*-
"""meeting67 · 语料_适配 · 现有数据实测盘点（**只读**，不写 meeting66/，不复制音频）。

只读来源
--------
1) ``<path>\\SpeakerA7代模型训练数据\\asr_opt\\SpeakerA7代模型训练数据.list``（167 行 wav|spk|ZH|文本）
2) ``<path>\\SpeakerA7代模型训练数据\\(N).wav`` —— **只读**（soundfile.info 读头 / 可选整段读能量）
3) ``AHI4/meeting66/产物/数据_对齐asr/{meta.json,样本/*.npz}`` —— 只读
4) ``AHI4/meeting66/产物/数据_对齐/meta.json`` —— 只读（对照：均分对齐版）

产出（只写本目录）
------------------
``盘点.json`` + ``盘点.txt``。确定性：不依赖随机；能量/VAD 口径复用 ``数据_SpeakerA`` 常量。

用法
----
    & <workspace>/\03_AHI\\.venv\\Scripts\\python.exe 盘点_只读.py            # 元数据（秒级）
    & ... 盘点_只读.py --能量 --哈希                                      # 加整段读（能量/SNR/去重）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
AHI4 = HERE.parents[2]                      # ...\AHI4
M66 = AHI4 / "meeting66"

清单路 = Path(r"<path>\SpeakerA7代模型训练数据\asr_opt\SpeakerA7代模型训练数据.list")
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")
对齐ASR = M66 / "产物" / "数据_对齐asr"
对齐均分 = M66 / "产物" / "数据_对齐"

SR = 22050
窗长 = 551
跳长 = 220

统计行: list[str] = []


def 印(s: str = "") -> None:
    print(s)
    统计行.append(s)


def 分位(v: list[float]) -> dict:
    a = np.asarray(v, dtype=np.float64)
    return dict(最小=float(a.min()), p05=float(np.percentile(a, 5)), p25=float(np.percentile(a, 25)),
                中位=float(np.median(a)), p75=float(np.percentile(a, 75)), p95=float(np.percentile(a, 95)),
                最大=float(a.max()), 均值=float(a.mean()), 合计=float(a.sum()), 条数=int(a.size))


def 分箱(v: list[float], 边: list[float]) -> dict:
    a = np.asarray(v, dtype=np.float64)
    出 = {}
    for lo, hi in zip(边[:-1], 边[1:]):
        出["[%g,%g)" % (lo, hi)] = int(np.sum((a >= lo) & (a < hi)))
    return 出


# ------------------------------------------------------------------ 1. 读清单
def 读清单():
    行s = 清单路.read_text(encoding="utf-8").splitlines()
    条目 = []
    for i, 行 in enumerate(行s):
        行 = 行.strip()
        if not 行:
            continue
        段 = 行.split("|", 3)
        if len(段) != 4:
            continue
        原路, 说话人, 语言, 文本 = 段
        名 = Path(原路.strip()).name.strip()
        条目.append(dict(序号=len(条目), 原路=原路.strip(), 名=名, wav=音频根 / 名,
                         说话人=说话人.strip(), 语言=语言.strip(), 文本=文本.strip()))
    return 条目


# ------------------------------------------------------------------ 2. 音频头
def 音频头(条目):
    头s = []
    缺 = []
    for e in 条目:
        if not e["wav"].exists():
            缺.append(e["名"])
            头s.append(None)
            continue
        i = sf.info(str(e["wav"]))
        头s.append(dict(名=e["名"], sr=int(i.samplerate), 声道=int(i.channels), 帧=int(i.frames),
                        格式=i.format, 子类型=i.subtype, 时长秒=round(i.frames / i.samplerate, 4),
                        字节=int(e["wav"].stat().st_size)))
    return 头s, 缺


def 子类型位深(子: str) -> int:
    m = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64, "PCM_U8": 8,
         "PCM_S8": 8, "PCM_24X": 24}
    return m.get(子.upper(), -1)


# ------------------------------------------------------------------ 3. 文本覆盖
def 文本覆盖(条目, 音韵):
    from pypinyin import Style, lazy_pinyin

    声计: dict = {}
    韵计: dict = {}
    调计: dict = {}
    音节计: dict = {}
    每文件 = []
    未命中 = []
    for e in 条目:
        cn = "".join(c for c in e["文本"] if ("\u4e00" <= c <= "\u9fff"))
        拼s = [p for p in lazy_pinyin(cn, style=Style.TONE3, neutral_tone_with_five=True)
               if p and p[0].isascii()] if cn else []
        音节s = [音韵.解析音节(p) for p in 拼s]
        音节s = [s for s in 音节s if s["命中"]]
        未命中 += [s for s in 音节s if not s["命中"]]
        声序 = []
        for s in 音节s:
            声计[s["声"] or "∅"] = 声计.get(s["声"] or "∅", 0) + 1
            韵计[s["韵"]] = 韵计.get(s["韵"], 0) + 1
            调计[int(s["调"])] = 调计.get(int(s["调"]), 0) + 1
            音节计[(s["声"] or "∅", s["韵"], int(s["调"]))] = 音节计.get((s["声"] or "∅", s["韵"], int(s["调"])), 0) + 1
            声序.append((s["声"] or "∅", s["韵"], int(s["调"])))
        每文件.append(dict(名=e["名"], 汉字数=len(cn), 音节数=len(音节s),
                         音节表=[[a, b, c] for a, b, c in 声序],
                         非汉字数=len(e["文本"]) - len(cn)))
    return dict(声计=声计, 韵计=韵计, 调计=调计, 音节计=音节计, 每文件=每文件, 未命中=未命中)


# ------------------------------------------------------------------ 4. 对齐产物
def 对齐结构(路: Path):
    if not (路 / "meta.json").exists():
        return dict(存在=False)
    m = json.loads((路 / "meta.json").read_text(encoding="utf-8"))
    样本 = m.get("样本", [])
    键 = dict(条数=len(样本), meta键=list(m.keys()), npz键=m.get("npz键"),
              参数=m.get("参数"), 统计=m.get("统计", {}), 划分=m.get("划分", {}).get("比例"),
              val数=m.get("划分", {}).get("val数"), train数=m.get("划分", {}).get("train数"),
              样本字段=(sorted(样本[0].keys()) if 样本 else []))
    npz = []
    for p in sorted((路 / "样本").glob("*.npz"))[:3]:
        with np.load(p, allow_pickle=False) as z:
            npz.append(dict(文件=p.name, 键=list(z.files),
                            形状={k: list(z[k].shape) for k in z.files},
                            类型={k: str(z[k].dtype) for k in z.files}))
    # 逐样本时长（来自 meta 样本表）
    时长 = [float(s.get("时长秒", 0.0)) for s in 样本]
    return dict(存在=True, 结构=键, npz样例=npz,
                时长分位=(分位(时长) if 时长 else None),
                总音节=int(sum(int(s.get("音节数", 0)) for s in 样本)),
                样本字段=sorted(样本[0].keys()) if 样本 else [],
                异常数=m.get("来源", {}).get("异常数"))


# ------------------------------------------------------------------ 5. 能量 / 去重
def 能量与哈希(条目):
    """整段读：峰值/有效值/静音比/近似 SNR + sha256 + 音频指纹（mel 均值 8 维极简指纹）。"""
    出 = []
    for e in 条目:
        x, sr = sf.read(str(e["wav"]), dtype="float32", always_2d=True)
        h = hashlib.sha256(Path(str(e["wav"])).read_bytes()).hexdigest()
        单 = np.ascontiguousarray(x.mean(axis=1, dtype=np.float32))
        if sr != SR:
            g = math.gcd(int(sr), SR)
            import scipy.signal as sps
            单 = np.ascontiguousarray(sps.resample_poly(单, SR // g, int(sr) // g).astype(np.float32))
        n = int(单.size)
        视 = np.lib.stride_tricks.sliding_window_view(单, 窗长)[::跳长]
        能 = np.mean(np.square(视.astype(np.float64)), axis=1)
        能db = 10.0 * np.log10(能 + 1e-12)
        峰 = float(能db.max())
        浊 = 能db > (峰 - 35.0)
        语音帧 = float(浊.mean()) if 浊.size else 0.0
        # 近似 SNR：浊帧能量均值 − 清帧（非浊）能量均值
        信 = float(能db[浊].mean() - (能db[~浊].mean() if (~浊).any() else 能db.min()))
        出.append(dict(名=e["名"], sha256=h, 采样=int(n), 峰值dB=round(峰, 3),
                       静态占比=round(1.0 - 语音帧, 4), 近似SNR_dB=round(信, 3),
                       指纹=[round(float(v), 3) for v in np.percentile(
                           np.log(np.asarray(能) + 1e-12), [5, 25, 50, 75, 95]).tolist()]))
    return 出


# ------------------------------------------------------------------ 主
def 主(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--能量", action="store_true", help="整段读音频：峰值/静音占比/近似SNR")
    ap.add_argument("--哈希", action="store_true", help="整段读音频：sha256 去重")
    ap.add_argument("--出", default=str(HERE / "盘点.json"))
    args = ap.parse_args(argv)

    for p in (AHI4 / "meeting61" / "实施",):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import 音韵_v5 as 音韵

    印("=" * 86)
    印("[盘点] SpeakerA 167 × 60s 语料（只读） ｜ meeting67/团队/语料_适配")
    印("清单: %s" % 清单路)
    印("音频根: %s" % 音频根)

    条目 = 读清单()
    印("\n[1] 清单：%d 行 / %d 条" % (len(清单路.read_text(encoding='utf-8').splitlines()), len(条目)))
    说话人 = sorted({e["说话人"] for e in 条目})
    语言 = sorted({e["语言"] for e in 条目})
    印("    说话人集合=%s（%d）  语言=%s" % (说话人, len(说话人), 语言))

    头s, 缺 = 音频头(条目)
    有效 = [(e, h) for e, h in zip(条目, 头s) if h]
    印("\n[2] 音频头：存在 %d / 缺失 %d%s" % (len(有效), len(缺), ("（%s）" % 缺[:5]) if 缺 else ""))

    sr计: dict = {}
    声道计: dict = {}
    位深计: dict = {}
    格式计: dict = {}
    时长 = []
    字节 = []
    for e, h in 有效:
        sr计[h["sr"]] = sr计.get(h["sr"], 0) + 1
        声道计[h["声道"]] = 声道计.get(h["声道"], 0) + 1
        位深计[子类型位深(h["子类型"])] = 位深计.get(子类型位深(h["子类型"]), 0) + 1
        格式计[h["子类型"]] = 格式计.get(h["子类型"], 0) + 1
        时长.append(h["时长秒"])
        字节.append(h["字节"])
    分 = 分位(时长)
    印("    采样率分布: %s" % sr计)
    印("    声道分布  : %s" % 声道计)
    印("    位深分布  : %s（子类型 %s）" % (位深计, 格式计))
    印("    总字节=%.2f GB" % (sum(字节) / 2 ** 30))
    印("\n[3] 时长（秒，按帧数/sr 精确算）：")
    印("    最小=%.3f p05=%.3f p25=%.3f 中位=%.3f p75=%.3f p95=%.3f 最大=%.3f 均值=%.3f 合计=%.2f s = %.4f h"
       % (分["最小"], 分["p05"], 分["p25"], 分["中位"], 分["p75"], 分["p95"], 分["最大"], 分["均值"],
          分["合计"], 分["合计"] / 3600.0))
    印("    分箱: %s" % 分箱(时长, [0, 10, 30, 60, 61, 120, 180, 3600]))

    # 文本覆盖
    覆盖 = 文本覆盖(条目, 音韵)
    声母表 = ["∅"] + list(音韵.声母序)
    韵母表 = list(音韵.韵母表)
    调表 = [1, 2, 3, 4, 5]
    音节总 = sum(覆盖["声计"].values())
    韵表外 = sorted(y for y in 覆盖["韵计"] if y not in set(韵母表))
    印("\n[4] 文本 / 音素覆盖（语料转写，非 ASR）：")
    印("    汉字音节总数=%d  非汉字字符总数=%d" % (
        音节总, sum(x["非汉字数"] for x in 覆盖["每文件"])))
    印("    声母 types=%d/%d（表内）  tokens=%d；缺失=%s" % (
        len(覆盖["声计"]), len(声母表), 音节总,
        [x for x in 声母表 if x not in 覆盖["声计"]]))
    印("    韵母 types=%d（表内 %d/%d）  tokens=%d；表内未用=%s；语料有而表外=%s" % (
        len(覆盖["韵计"]), len([y for y in 覆盖["韵计"] if y in set(韵母表)]), len(韵母表), 音节总,
        [x for x in 韵母表 if x not in 覆盖["韵计"]], 韵表外))
    印("    调   types=%d/%d  tokens=%d；缺失=%s" % (
        len(覆盖["调计"]), len(调表), 音节总,
        [x for x in 调表 if x not in 覆盖["调计"]]))
    印("    调分布: %s" % dict(sorted(覆盖["调计"].items())))
    音节型 = 覆盖["音节计"]
    印("    「声母+韵母+调」音节型 types=%d / 理论上限 %d（%d×%d×%d）；tokens=%d"
       % (len(音节型), len(声母表) * len(韵母表) * len(调表), len(声母表), len(韵母表), len(调表), 音节总))
    少用 = sorted(音节型.items(), key=lambda kv: kv[1])[:10]
    印("    最稀少音节型(前10): %s" % [(f"{a}+{b}+{c}", n) for (a, b, c), n in 少用])
    印("    每文件音节数: 最小=%d 中位=%.0f 最大=%d 均值=%.2f" % (
        min(x["音节数"] for x in 覆盖["每文件"]),
        float(np.median([x["音节数"] for x in 覆盖["每文件"]])),
        max(x["音节数"] for x in 覆盖["每文件"]),
        float(np.mean([x["音节数"] for x in 覆盖["每文件"]]))))

    # 对齐产物
    印("\n[5] 对齐产物结构：")
    al = {}
    for 名, 路 in (("数据_对齐asr", 对齐ASR), ("数据_对齐", 对齐均分)):
        d = 对齐结构(路)
        al[名] = d
        if not d["存在"]:
            印("    %-14s 不存在" % 名)
            continue
        印("    %-14s 条数=%d 总音节=%d 划分=%s(val %s) 异常=%s"
          % (名, d["结构"]["条数"], d["总音节"], d["结构"]["划分"], d["结构"]["val数"], d["异常数"]))
        印("      参数=%s" % d["结构"]["参数"])
        印("      统计=%s" % {k: v for k, v in d["结构"]["统计"].items() if k != "mel均值" and k != "mel标准差"})
        印("      样本字段=%s" % d["结构"]["样本字段"])
        for s in d["npz样例"]:
            印("      npz %s 键=%s" % (s["文件"], s["键"]))
            印("          形状=%s" % s["形状"])
            印("          类型=%s" % s["类型"])
        if d["时长分位"]:
            q = d["时长分位"]
            印("      逐样本时长(秒): 最小=%.2f 中位=%.2f 最大=%.2f 合计=%.2f s"
              % (q["最小"], q["中位"], q["最大"], q["合计"]))
    印("    注：对齐版 时长≈T×10ms（不含端点裁剪外的时间），小于原始文件时长；见 [3]")

    # 能量 / 去重
    能 = {}
    if args.能量 or args.哈希:
        print("  整段读 %d 个 wav（约 %.2f GB）……" % (len(条目), sum(字节) / 2 ** 30), flush=True)
        能 = 能量与哈希(条目)
        if args.能量:
            snr = [x["近似SNR_dB"] for x in 能]
            静 = [x["静态占比"] for x in 能]
            印("\n[6] 能量 / 近似 SNR（整段，25ms/10ms 栅格，能量阈=峰值−35dB，与数据_SpeakerA同口径）：")
            印("    近似SNR_dB: 最小=%.2f p05=%.2f 中位=%.2f 最大=%.2f 均值=%.2f"
              % (min(snr), float(np.percentile(snr, 5)), float(np.median(snr)), max(snr), float(np.mean(snr))))
            印("    静音帧占比: 最小=%.3f 中位=%.3f 最大=%.3f（>0.5 的条数=%d）"
              % (min(静), float(np.median(静)), max(静), int(sum(1 for v in 静 if v > 0.5))))
        if args.哈希:
            sha = [x["sha256"] for x in 能]
            重 = len(sha) - len(set(sha))
            印("\n[7] 去重（整文件 sha256）：%d 条 → 唯一 %d 条；重复 %d 条" % (len(sha), len(set(sha)), 重))
            指 = [tuple(x["指纹"]) for x in 能]
            印("    极简音频指纹（5 分位能量 dB）唯一 %d 条" % len(set(指)))
            # 与 L: 下 4 个子目录做「同名同大小」核对（会议66 十跑结论复核）
            for 子 in ("RVC训练集70min", "RVC训练集70min_v2", "RVC训练集_分级混合", "筛选后训练集"):
                路 = 音频根 / 子
                if not 路.exists():
                    印("    子目录 %s 不存在" % 子)
                    continue
                f = [p for p in 路.rglob("*.wav")]
                同名同大 = 0
                for p in f:
                    名 = p.name.strip()
                    原 = 音频根 / 名
                    if 原.exists() and 原.stat().st_size == p.stat().st_size:
                        同名同大 += 1
                印("    子目录 %-14s wav=%d，与根目录同名同大小=%d" % (子, len(f), 同名同大))

    # 落盘
    出 = dict(
        版本="meeting67 语料_适配 盘点 v1",
        清单=str(清单路), 音频根=str(音频根),
        清单行数=len(条目), 说话人=说话人, 语言=语言,
        音频=dict(存在=len(有效), 缺失=len(缺), 采样率=sr计, 声道=声道计, 位深=位深计, 子类型=格式计,
                 总字节=int(sum(字节)), 总GB=round(sum(字节) / 2 ** 30, 4)),
        时长=dict(分位=分, 分箱=分箱(时长, [0, 10, 30, 60, 61, 120, 180, 3600]), 逐条=时长),
        覆盖=dict(汉字音节总数=音节总, 声types=len(覆盖["声计"]), 声母表=len(声母表),
                 韵types=len(覆盖["韵计"]), 韵母表=len(韵母表),
                 调types=len(覆盖["调计"]), 调=sorted(覆盖["调计"]),
                 音节型=len(音节型), 声计=覆盖["声计"], 韵计=覆盖["韵计"], 调计=覆盖["调计"],
                 每文件=覆盖["每文件"]),
        对齐=al,
        能量=([{k: v for k, v in x.items() if k != "指纹"} for x in 能] if 能 else None),
    )
    Path(args.出).write_text(json.dumps(出, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (HERE / "盘点.txt").write_text("\n".join(统计行) + "\n", encoding="utf-8", newline="\n")
    print("\n写出: %s 和 盘点.txt" % args.出)
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
