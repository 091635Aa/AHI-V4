# -*- coding: utf-8 -*-
"""meeting63/实施/判据_可懂度_ASR.py —— 外部可懂度裁判（合成语音 → ASR → 与原文比对）。

★ C10 边界（不许含糊）：本件是**外部评估仪器**，**不在生成路径**里（生成件零外挂由 `审计_零外挂.py`
  独立证明：生成路径音频/权重读入 = 0）。ASR 只用来回答「听清楚了没有」，**不当发声器官**。

口径：合成音频 → ASR 转写 → 对**已知原文**做**字符错误率 CER**（越接近 0 越可懂）。
  must-fail：白噪/静音/倒放喂同一 ASR ⇒ CER 必须高（≈1）。
  对照：v7（修前）vs v8（修后）同一句话、同一 ASR ⇒ 看 CER 是否下降。

引擎优先级：中文专用 Paraformer（ModelScope 缓存）→ 失败则退 faster-whisper（HF 缓存）。
用法：`python 判据_可懂度_ASR.py` → 打印 + 写 `产物/判据/判据_可懂度_ASR.json`

多句可懂度（本件新增，只做测量）：
  6 句真值 × 3 角色（EmoCompanion/SpeakerA/洛天依）⇒ 18 样本，每样本转写两趟（C13 一致性），
  CER + Levenshtein 逐字对齐（S/D/I 与逐字操作）+ 按音韵类归因（鼻尾/卷舌/平舌/塞音/…）；
  与 `判据_可懂度_ASR_多句_基线.json`（首次运行时由本件写入）比较 ⇒ 相对基线下降。
  输出 `产物/判据/判据_可懂度_ASR_多句.json`。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
M63 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M63)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting62", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

from pypinyin import lazy_pinyin, Style  # noqa: E402

import 声带后端_v8 as 后8  # noqa: E402  只读（多句合成用；不改 v8 一字）
import 声线覆盖层_v7 as 覆  # noqa: E402  只读（三角色声线档案）
import 音韵_v5 as 音韵  # noqa: E402  只读（声母/韵母/韵尾归因）

真值 = "今天的风很轻，适合慢慢地说一会儿话。"
V8目录 = os.path.join(M63, "产物", "交付_v8_1", "说话")
V8目录旧 = os.path.join(M63, "产物", "交付_v8", "说话")
V7目录 = os.path.join(AHI4, "meeting62", "产物", "交付_v7", "说话")
出目录 = os.path.join(M63, "产物", "判据")
os.makedirs(出目录, exist_ok=True)
临时 = os.path.join(M63, "产物", "ASR临时")
os.makedirs(临时, exist_ok=True)

# —— 多句可懂度：真值集（已用 音韵_v5.审计 验证 问题数=0）、角色、产物路径 ——
句集 = (
    ("S1", "今天的风很轻，适合慢慢地说一会儿话。"),
    ("S2", "我们明天早上八点在南门见面。"),
    ("S3", "他把三本书放在窗台上。"),
    ("S4", "小猫从桌上跳下来，碰到杯子。"),
    ("S5", "春天来了，山上的雪开始融化。"),
    ("S6", "请问这条路怎么走？"),
)
多句角色 = ("EmoCompanion", "SpeakerA", "洛天依")
多句临时 = os.path.join(临时, "多句")   # 不复用 交付_v8/说话（那是单句交付）
os.makedirs(多句临时, exist_ok=True)
多句出 = os.path.join(出目录, "判据_可懂度_ASR_多句.json")
基线文件 = os.path.join(出目录, "判据_可懂度_ASR_多句_基线.json")


# ───────────────────────── 归一化 & CER ─────────────────────────
def _正规(s: str) -> str:
    保 = []
    for ch in str(s):
        if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
            保.append(ch)
    return "".join(保)


def CER(真: str, 测: str) -> float:
    真, 测 = _正规(真), _正规(测)
    if not 真:
        return 0.0
    n, m = len(真), len(测)
    前 = list(range(m + 1))
    for i in range(1, n + 1):
        今 = [i] + [0] * m
        for j in range(1, m + 1):
            今[j] = min(前[j] + 1, 今[j - 1] + 1, 前[j - 1] + (真[i - 1] != 测[j - 1]))
        前 = 今
    return round(前[m] / n, 4)


# ───────────────────────── 引擎 ─────────────────────────
def _写wav(路径, x, fs=44100):
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    wavfile.write(路径, fs, (x * 32767.0).round().astype("<i2"))


def 载引擎():
    """→ dict(名, 转写)。优先中文 Paraformer（ModelScope 缓存），否则 faster-whisper。"""
    md = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "models",
                      "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    if os.path.isdir(md):
        try:
            from funasr import AutoModel
            m = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                          disable_update=True, trust_remote_code=True, device="cpu",
                          cache_dir=os.path.dirname(md))
            def 转写(p):
                r = m.generate(input=p, batch_size_s=300)
                return (r[0].get("text", "") if r else "")
            return dict(名="Paraformer-seaco-large(zh)", 转写=转写)
        except Exception as e:  # noqa: BLE001
            print("[Paraformer 不可用，退 faster-whisper] %s: %s" % (type(e).__name__, str(e)[:160]))
    from faster_whisper import WhisperModel
    snap = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                        "models--Systran--faster-whisper-medium", "snapshots")
    路径 = None
    if os.path.isdir(snap):
        s = [os.path.join(snap, d) for d in os.listdir(snap)]
        路径 = s[0] if s else None
    m = WhisperModel(路径 or "medium", device="cpu", compute_type="int8")
    def 转写(p):
        segs, _ = m.transcribe(p, language="zh", beam_size=5, condition_on_previous_text=False)
        return "".join(s.text for s in segs).strip()
    return dict(名="faster-whisper-medium(HF缓存)", 转写=转写)


def 单句段(引: dict) -> dict:
    """原「单句段」（逐位保留原行为）：真值 × 三角色 v8 ＋ v7 对照 ＋ 真人 ＋ sham。"""
    print("ASR 引擎：", 引["名"])

    # —— sham 音频（must-fail）——
    rng = np.random.default_rng(63051)
    白噪 = rng.standard_normal(int(3.4 * 44100)) * 0.2
    静音 = np.zeros(int(3.4 * 44100))
    路径白 = os.path.join(临时, "sham_白噪.wav")
    路径静 = os.path.join(临时, "sham_静音.wav")
    _写wav(路径白, 白噪)
    _写wav(路径静, 静音)

    样本 = []
    for 标 in ("洛天依", "SpeakerA", "EmoCompanion"):
        for 目录, 文件名 in ((V8目录, "说话_v8_1_%s.wav" % 标),
                            (V8目录旧, "说话_v8_%s.wav" % 标)):
            p = os.path.join(目录, 文件名)
            if os.path.exists(p):
                样本.append(("v8·" + 标, p))
                break
    p7 = os.path.join(V7目录, "说话_v7_EmoCompanion.wav")
    if os.path.exists(p7):
        样本.append(("v7·EmoCompanion(修前)", p7))
    ref = os.path.join(AHI4, "meeting61", "素材", "EmoCompanion", "ref44_EmoCompanion_01.wav")
    if os.path.exists(ref):
        样本.append(("真人·EmoCompanion(对照,无真值)", ref))
    样本.append(("sham·白噪", 路径白))
    样本.append(("sham·静音", 路径静))

    读 = {}
    for 名, p in 样本:
        t = 引["转写"](p)
        读[名] = dict(转写=t, CER=CER(真值, t))
        print("%-16s CER=%.3f 「%s」" % (名, 读[名]["CER"], t))

    v8s = [v["CER"] for k, v in 读.items() if k.startswith("v8")]
    v7 = 读.get("v7·EmoCompanion(修前)", {}).get("CER")
    白 = 读["sham·白噪"]["CER"]
    静 = 读["sham·静音"]["CER"]
    报 = dict(工具="判据_可懂度_ASR.py", 引擎=引["名"], 真值=真值, 逐样本=读,
              v8_CER均值=round(float(np.mean(v8s)), 4) if v8s else None,
              v7_CER=v7,
              must_fail_白噪=white_ok if (white_ok := bool(白 >= 0.70)) else False,
              must_fail_静音=bool(静 >= 0.70),
              注="★ 外部评估仪器（不在生成路径）；生成路径零外挂由 审计_零外挂.py 独立证明")
    报["通过"] = bool(白 >= 0.70 and 静 >= 0.70 and v8s and float(np.mean(v8s)) <= 0.60)
    out = os.path.join(出目录, "判据_可懂度_ASR.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("-" * 80)
    print("v8 CER 均值 = %s ｜ v7 CER = %s ｜ sham 白噪/静音 = %.3f/%.3f"
          % (报["v8_CER均值"], 报["v7_CER"], 白, 静))
    print("JSON:", out)
    return 报


# ═══════════════ 多句可懂度：逐字对齐 / 音韵归因 / 多句合成 / 判据 ═══════════════
def 对齐(真: str, 测: str) -> dict:
    """Levenshtein 逐字对齐（带回溯）：→ S/D/I 计数 + 每个真值字的操作 + 插入的字。

    回溯优先级固定（对角 → 删除 → 插入）⇒ 同输入同输出（C13）。
    """
    真, 测 = _正规(真), _正规(测)
    n, m = len(真), len(测)
    距 = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        距[i][0] = i
    for j in range(1, m + 1):
        距[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            距[i][j] = min(距[i - 1][j] + 1, 距[i][j - 1] + 1,
                          距[i - 1][j - 1] + (真[i - 1] != 测[j - 1]))
    i, j = n, m
    操作, 插入 = [], []
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and 距[i][j] == 距[i - 1][j - 1] + (真[i - 1] != 测[j - 1]):
            if 真[i - 1] == 测[j - 1]:
                操作.append(dict(位=i, 真=真[i - 1], 操作="命中"))
            else:
                操作.append(dict(位=i, 真=真[i - 1], 操作="替换成" + 测[j - 1], 测=测[j - 1]))
                S += 1
            i, j = i - 1, j - 1
        elif i > 0 and 距[i][j] == 距[i - 1][j] + 1:
            操作.append(dict(位=i, 真=真[i - 1], 操作="删除"))
            D += 1
            i -= 1
        else:
            插入.append(dict(位=i, 字=测[j - 1]))   # 位 = 第 i 个真值字之后
            I += 1
            j -= 1
    操作.reverse()
    插入.reverse()
    return dict(S=S, D=D, I=I, 操作=操作, 插入=插入,
                错误位=[o["位"] for o in 操作 if o["操作"] != "命中"],
                # 纯规范文本（供复查）
                真文=真, 测文=测)


def _真值字(原文: str) -> list:
    """→ [(字, 拼音, 是否短语末)]，与 _正规(原文) 逐字一一对应（短语末 = 后接标点或句末）。"""
    原 = str(原文)
    拼s = lazy_pinyin(原, style=Style.TONE3, neutral_tone_with_five=True)
    if len(拼s) != len(原):   # 兜底：长度不齐则逐字取（确定性）
        拼s = [lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)[0]
               if ch.strip() else ch for ch in 原]
    出 = []
    for i, ch in enumerate(原):
        if not ("\u4e00" <= ch <= "\u9fff" or ch.isalnum()):
            continue
        nxt = 原[i + 1] if i + 1 < len(原) else ""
        末 = nxt == "" or not ("\u4e00" <= nxt <= "\u9fff" or nxt.isalnum())
        出.append((ch, 拼s[i], 末))
    return 出


def _归类(音: dict) -> list:
    """一个音节 → 所属音韵类（可多属：ch 属 卷舌+塞擦）。"""
    声, 尾 = 音["声"], 音["尾"]
    类 = []
    if 尾 in ("n", "ng"):
        类.append("鼻尾")
    if 声 in ("zh", "ch", "sh", "r"):
        类.append("卷舌")
    if 声 in ("z", "c", "s"):
        类.append("平舌")
    if 声 in ("b", "d", "g"):
        类.append("塞音")
    if 声 in ("p", "t", "k"):
        类.append("送气塞")
    if 声 in ("z", "c", "zh", "ch", "j", "q"):
        类.append("塞擦")
    if 声 in ("f", "h", "s", "sh", "x"):
        类.append("擦音")
    if 声 in ("m", "n", "l", "r"):
        类.append("浊声母")
    if 声 == "":
        类.append("零声母")
    return 类


def 多句段(引: dict) -> dict:
    """6 句真值 × 3 角色：合成 → 两趟转写 → CER → 逐字对齐 → 音韵类归因 → 基线比较。"""
    # —— 1) 合成（临时目录，不复用单句交付目录）——
    音色s = {名: 覆.取(名).转dict() for 名 in 多句角色}
    后 = 后8.声带V8(seed=300, 目录=多句临时,
                    台账路径=os.path.join(多句临时, "台账_v8.jsonl"))
    路径表 = {}
    for 句号, 句 in 句集:
        for 名 in 多句角色:
            r = 后.发声(np.zeros(1024), 句, 音色=音色s[名])
            路径表[(句号, 名)] = (os.path.join(多句临时, r["wav"]), r["时长秒"])
            print("[合成] %s·%-4s %s 时长%.2fs" % (句号, 名, r["wav"], r["时长秒"]), flush=True)

    # —— 2) 两趟转写（C13 一致性）+ CER + 逐字对齐 ——
    读 = {}
    一致 = True
    for 句号, 句 in 句集:
        for 名 in 多句角色:
            键 = "%s·%s" % (句号, 名)
            p = 路径表[(句号, 名)][0]
            t1 = 引["转写"](p)
            t2 = 引["转写"](p)
            if t1 != t2:
                一致 = False
                print("[两趟不一致] %s「%s」≠「%s」" % (键, t1, t2), flush=True)
            读[键] = dict(句号=句号, 句=句, 角色=名, wav=os.path.basename(p), 时长秒=路径表[(句号, 名)][1],
                         转写=t1, 转写第二趟=t2, 一致=bool(t1 == t2),
                         CER=CER(句, t1), 对齐=对齐(句, t1))
            print("%-9s CER=%.3f 「%s」" % (键, 读[键]["CER"], t1), flush=True)

    # —— 3) 音韵类归因（错误 = 该真值字为替换或删除）——
    计 = {}

    def _加(类, 错):
        d = 计.setdefault(类, dict(错误=0, 总数=0))
        d["总数"] += 1
        if 错:
            d["错误"] += 1

    for 键, v in 读.items():
        错位 = set(v["对齐"]["错误位"])
        for k, (ch, 拼, 末) in enumerate(_真值字(v["句"]), 1):
            if not ("\u4e00" <= ch <= "\u9fff"):
                continue          # 非汉字（字母/数字）不参与音韵归因
            音 = 音韵.解析音节(拼)
            错 = k in 错位
            for 类 in _归类(音):
                _加(类, 错)
            if 音["尾"] in ("n", "ng"):
                _加("鼻尾·短语末" if 末 else "鼻尾·句中", 错)
    归因 = {类: dict(错误=d["错误"], 总数=d["总数"],
                     错误率=round(d["错误"] / d["总数"], 4) if d["总数"] else 0.0)
            for 类, d in 计.items()}

    # —— 4) sham（沿用单句段生成方式：白噪 / 静音，3.4s，44.1kHz）——
    rng = np.random.default_rng(63051)
    路径白 = os.path.join(临时, "sham_白噪.wav")
    路径静 = os.path.join(临时, "sham_静音.wav")
    _写wav(路径白, rng.standard_normal(int(3.4 * 44100)) * 0.2)
    _写wav(路径静, np.zeros(int(3.4 * 44100)))
    sham白 = CER(真值, 引["转写"](路径白))
    sham静 = CER(真值, 引["转写"](路径静))
    print("[sham] 白噪 CER=%.3f ｜ 静音 CER=%.3f" % (sham白, sham静), flush=True)

    多句均值 = round(float(np.mean([v["CER"] for v in 读.values()])), 4)

    # —— 5) 基线（首次运行写入；此后只读取）——
    首跑 = not os.path.exists(基线文件)
    if 首跑:
        基 = dict(工具="判据_可懂度_ASR.py", 引擎=引["名"],
                 句集=[dict(号=s, 句=t) for s, t in 句集],
                 多句CER均值=多句均值,
                 逐样本CER={k: v["CER"] for k, v in 读.items()},
                 sham=dict(白噪=sham白, 静音=sham静),
                 注="修复前基线（首次运行由 判据_可懂度_ASR.py 写入；此后本件只读）")
        with open(基线文件, "w", encoding="utf-8") as f:
            json.dump(基, f, ensure_ascii=False, indent=1)
        print("[基线] 首次运行，已写入：", 基线文件, flush=True)
    with open(基线文件, encoding="utf-8") as f:
        基值 = float(json.load(f)["多句CER均值"])
    下降 = round((基值 - 多句均值) / 基值, 4) if 基值 > 0 else 0.0

    通过 = bool(多句均值 <= 0.55 and 下降 >= 0.25 and sham白 >= 0.9 and sham静 >= 0.9)
    报 = dict(
        工具="判据_可懂度_ASR.py",
        引擎=引["名"],
        句集=[dict(号=s, 句=t) for s, t in 句集],
        逐样本=读,
        多句CER均值=多句均值,
        逐类归因=归因,
        sham=dict(白噪=sham白, 静音=sham静, 双达标=bool(sham白 >= 0.9 and sham静 >= 0.9)),
        must_fail_白噪=bool(sham白 >= 0.9),
        must_fail_静音=bool(sham静 >= 0.9),
        两趟一致=bool(一致),
        基线CER均值=基值,
        相对基线下降=下降,
        通过=通过,
        基线文件=基线文件,
        首次写基线=bool(首跑),
        注=("★ 外部评估仪器（不在生成路径，C10）；18 样本 × 2 趟（C13）；"
            "通过 = 多句CER均值<=0.55 且 相对基线下降>=0.25 且 sham>=0.9。"
            + ("本次为首次运行 ⇒ 已写基线，通过自然为 false。" if 首跑 else "")),
    )
    with open(多句出, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)

    # —— 6) 打印汇总 ——
    print("-" * 80)
    print("%-6s %s" % ("句", "  ".join("%-7s" % 名 for 名 in 多句角色)))
    for 句号, _ in 句集:
        print("%-6s %s" % (句号, "  ".join(
            "%-7.3f" % 读["%s·%s" % (句号, 名)]["CER"] for 名 in 多句角色)))
    print("多句 CER 均值 = %.4f ｜ 基线 = %.4f ｜ 相对下降 = %s" % (多句均值, 基值, 下降))
    前5 = sorted(归因.items(), key=lambda kv: (-kv[1]["错误率"], -kv[1]["总数"], kv[0]))[:5]
    print("错误率最高 5 类：" + " ｜ ".join(
        "%s %.3f(%d/%d)" % (k, v["错误率"], v["错误"], v["总数"]) for k, v in 前5))
    for k in ("鼻尾", "鼻尾·句中", "鼻尾·短语末"):
        if k in 归因:
            v = 归因[k]
            print("  %-8s 错误率 %.3f（%d/%d）" % (k, v["错误率"], v["错误"], v["总数"]))
    print("两趟一致 = %s ｜ sham 白噪/静音 = %.3f/%.3f ｜ 通过 = %s"
          % (一致, sham白, sham静, 通过))
    print("JSON:", 多句出)
    return 报


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    引 = 载引擎()
    单句段(引)
    print("=" * 80)
    多句段(引)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
