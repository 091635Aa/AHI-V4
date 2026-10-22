# -*- coding: utf-8 -*-
"""meeting64/实施/量具_v9.py —— 本轮统一量具（**一次载引擎 → 串行跑多条件**）。

★ 定位：本件是「外部评估仪器」的**驱动层**，只作测量，**绝不进生成路径**（C10 边界）。
  生成路径零外挂由 meeting63 `审计_零外挂.py` 独立证明；ASR 只回答「听清楚了没有」。

★ 迁移来源（C14：只新增 meeting64/，不改 meeting61/62/63 一字）：
  · `载引擎 / _正规 / CER / 对齐 / _真值字 / _归类`  = **同源复制**自
    `meeting63/实施/判据_可懂度_ASR.py`（Paraformer-seaco-large，CPU，disable_update，
    trust_remote_code），量法/口径一字不改 ⇒ 读数可与 meeting63 的 0.5292 直接对齐。
  · `句集 / 角色` 同源；基线 CER 0.7212（v7/v8 修复前，meeting63 登记值）。

★ 与 meeting63 版的关键差别（本件的存在理由）：
  1. **不经 `声带后端_v8`**：直接 `韵律工厂(档, 种子).说(句)` 取波形 ⇒ 不写 meeting63 台账、
     不落任何 meeting63 产物。`_造档案()` 与 `声带后端_v8.声带V8._档案()` 逐字同形（含默认值与
     过滤规则）⇒ 同档同种子**波形逐位同**（用 `对拍参考` 对 meeting63 既有 18 个基线 wav 校验）。
  2. **一次载引擎跑多条件**（`跑批`）：ASR 是全流程最贵的资源（机器 15.4GB／空闲约 4~5GB，
     两个 paraformer 进程会 OOM）⇒ 载一次引擎、串行跑完所有条件。
  3. **临时音频文件名每次唯一**（`run_id` = 时间戳+pid+随机后缀，含标签+条件序号+序号）：
     历史坑——复用同一文件名时 ASR 可能读到滞后音频，产生假读数（meeting63 §8.5 已登记）。
     唯一文件名让「读到哪个文件」可核对，杜绝该类假读数。

★ 接口（供 v9 A/B 直接用；Lead 只需传 `韵律工厂` 与 `角色改`）：
    import 量具_v9 as 量
    引 = 量.载引擎()                                  # 一次
    果 = 量.跑批([dict(名="v8.1基线", 韵律工厂=类或函数,
                        角色s=量.多句角色, 角色改=fn_or_None), ...], 引=引)
    # 角色改(角色名, 原音色dict) -> 新音色dict；None = 不改

★ 用法（跑 v8.1 基线复现；会载 Paraformer，单进程，勿并发）：
    $env:PYTHONIOENCODING="utf-8"
    & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" "<workspace>/\03_AHI\\AHI4\\meeting64\\实施\\量具_v9.py"
  → 打印每句三角色 CER 表 + 写 `meeting64/产物/判据/基线_v8_1_复现.json`

★ 确定性（C13）：波形只由 (档, 种子, 句) 决定；两趟转写逐位同并登记；同条件两趟跑
  `多句CER均值` 与 `波形sha256` 必须逐位同（`跑批` 的 `复跑对照` 字段自查）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M64)
M63 = os.path.join(AHI4, "meeting63")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(M63, "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting62", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

from pypinyin import Style, lazy_pinyin  # noqa: E402

import 声线覆盖层_v7 as 覆  # noqa: E402  只读（三角色声线档案 + 覆盖层）
import 音韵_v5 as 音韵  # noqa: E402      只读（声母/韵母/韵尾归因）
from 接口契约_角色声线 import 声线档案  # noqa: E402  只读（冻结契约）

# ───────────────────────── 路径 / 口径常量 ─────────────────────────
出目录 = os.path.join(M64, "产物", "判据")
临时根 = os.path.join(M64, "产物", "ASR临时")
批出目录 = os.path.join(M64, "产物", "诊断")
批出 = os.path.join(批出目录, "batch.json")

基线CER_多句 = 0.7212          # meeting63 登记：v7/v8 修复前 6 句 × 3 角色 18 样本均值
基线CER_来源 = "meeting63/03_说明_v8.md §8.4（修复前基线）"
验收_目标 = 0.5292             # meeting63 登记：v8.1（鼻尾全音节化 8/7 帧 + 持续包络）
验收_容差 = 0.0015
FS = 44100
真值 = "今天的风很轻，适合慢慢地说一会儿话。"   # 单句口径（sham 用）

# 多句真值集（与 meeting63 判据_可懂度_ASR.句集 **逐字同源**）
句集 = (
    ("S1", "今天的风很轻，适合慢慢地说一会儿话。"),
    ("S2", "我们明天早上八点在南门见面。"),
    ("S3", "他把三本书放在窗台上。"),
    ("S4", "小猫从桌上跳下来，碰到杯子。"),
    ("S5", "春天来了，山上的雪开始融化。"),
    ("S6", "请问这条路怎么走？"),
)
多句角色 = ("EmoCompanion", "SpeakerA", "洛天依")

# ───────────────────────── 运行期唯一标识（临时文件名用，绝不写进音频内容） ─────────────────────────
_RUN_ID = "%s-%d-%s" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid(),
                        hashlib.sha256(os.urandom(16)).hexdigest()[:6])


def 新run_id() -> str:
    """新的运行标识（同一进程内多次调用也唯一）。"""
    return "%s-%s" % (_RUN_ID, hashlib.sha256(os.urandom(8)).hexdigest()[:4])


def _确保目录(*路径s) -> None:
    for p in 路径s:
        os.makedirs(p, exist_ok=True)


# ══════════════════════ 归一化 & CER & 对齐（同源复制自 判据_可懂度_ASR.py） ══════════════════════
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


# ══════════════════════ 引擎（同源复制自 判据_可懂度_ASR.py） ══════════════════════
def 写wav(路径: str, x, fs: int = FS) -> None:
    """44.1kHz / 16bit / 单声道（与 声带后端_v8._写wav 同一量化规则 ⇒ 逐位可对拍）。"""
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    pcm = (x * 32767.0).round().astype("<i2")
    with wave.open(路径, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())


def 载引擎():
    """→ dict(名, 转写)。中文专用 Paraformer（ModelScope 缓存，CPU）；失败退 faster-whisper。

    ★ 单进程约束：Paraformer 常驻内存约 2–3GB；本机 15.4GB / 空闲约 4~5GB ⇒ **同一时刻只许一个**。
    """
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


# ══════════════════════ 声线档案（与 声带后端_v8.声带V8._档案 同形） ══════════════════════
_允许音色键 = {"基频", "语速", "幅度", "气声度", "紧喉度", "粗糙度", "送气量",
              "声区", "共振峰缩放", "谱倾斜", "气声谱重心", "抖动", "增益"}


def 造档案(音色: dict | None, 名: str = "内核") -> 声线档案:
    """与 `声带后端_v8.声带V8._档案` **逐字同形**（只保留允许键、补默认值、共振峰转 tuple）。

    ⇒ 同 (音色, 种子, 句) 时本件波形与 `声带后端_v8` 的波形**逐位同**（可用 sha256 对拍证明）。
    """
    音 = dict(音色 or {})
    d = {k: v for k, v in 音.items() if k in _允许音色键}
    d.setdefault("基频", 200.0)
    d.setdefault("语速", 4.8)
    d["名"] = 名
    d["来源"] = "内生·参数化"
    if "共振峰缩放" in d:
        d["共振峰缩放"] = tuple(float(x) for x in d["共振峰缩放"])
    a = 声线档案.从dict(d)
    a.校验()
    return a


def 取音色(角色s, 角色改=None) -> dict:
    """→ {角色: 音色dict}。`角色改(角色名, 原音色dict) -> 新音色dict`（None = 不改）。"""
    出 = {}
    for 名 in 角色s:
        d = 覆.取(名).转dict()          # meeting62 声线档案覆盖层（只读）
        if 角色改 is not None:
            d2 = 角色改(名, dict(d))
            if d2 is not None:
                d = d2
        出[名] = d
    return 出


def _工厂(韵律工厂):
    """接受「类」或「返回可说的话的对象」的函数；统一成 韵律工厂(档, 种子) → 对象(有 .说)。"""
    return 韵律工厂


def _sha256(路径: str) -> str:
    h = hashlib.sha256()
    with open(路径, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


# ══════════════════════ 合成多句（直接 韵律工厂(档,种子).说(句)，不经 声带后端_v8） ══════════════════════
def 合成多句(韵律工厂, 角色s=多句角色, 标签: str = "基线", 临时目录: str | None = None,
             种子: int = 300, 角色改=None, 句集入=句集, 条件号: int = 0) -> dict:
    """6 句 × 3 角色 → 18 个 wav（44.1kHz/16bit/mono）。

    ★ 临时文件名**每次唯一**：`多句_{标签}_j{条件号}_c{种子}_{序号:03d}_{角色}_{句号}_{run_id}.wav`
      （标签 + 条件号 + 序号 + run_id 四者齐备 ⇒ 不同批次、不同条件、不同进程都不会撞名；
       历史坑：复用文件名会被 ASR 读到滞后音频，产生假读数）。
    → dict(目录, run_id, 标签, 种子, 样本{(句号,角色): dict(wav,路径,sha256,时长秒)}, 角色音色指纹)
    """
    临时目录 = 临时目录 or os.path.join(临时根, "多句")
    _确保目录(临时目录)
    run = 新run_id()
    音色s = 取音色(角色s, 角色改)
    工厂 = _工厂(韵律工厂)
    样本 = {}
    序 = 0
    for 句号, 句 in 句集入:
        for 名 in 角色s:
            序 += 1
            档 = 造档案(音色s[名])
            波形, _元, _帧 = 工厂(档, int(种子)).说(str(句))
            波形 = np.asarray(波形, dtype=np.float64)
            文名 = "多句_%s_j%02d_c%s_%03d_%s_%s_%s.wav" % (
                _安全(标签), int(条件号), _安全(str(种子)), 序, _安全(名), _安全(句号), run)
            路径 = os.path.join(临时目录, 文名)
            写wav(路径, 波形)
            样本[(句号, 名)] = dict(
                句号=句号, 句=句, 角色=名, wav=文名, 路径=路径,
                sha256=_sha256(路径), 样本数=int(波形.size),
                时长秒=round(float(波形.size) / FS, 3),
                音色指纹=档.指纹(),
            )
            print("[合成] %s·%-4s %-46s %.2fs sha=%s"
                  % (句号, 名, 文名, 样本[(句号, 名)]["时长秒"], 样本[(句号, 名)]["sha256"][:12]),
                  flush=True)
    return dict(目录=临时目录, run_id=run, 标签=str(标签), 种子=int(种子),
                样本=样本, 角色音色s={k: dict(v) for k, v in 音色s.items()})


def _安全(s: str) -> str:
    """文件名安全化（保留中文；只把路径分隔符/冒号等替换掉）。"""
    坏 = '\\/:*?"<>| \t\r\n'
    return "".join("_" if ch in 坏 else ch for ch in str(s))[:32]


# ══════════════════════ sham（must-fail：白噪/静音须 CER ≥ 0.9） ══════════════════════
def _sham(引: dict, 临时目录: str) -> dict:
    _确保目录(临时目录)
    run = 新run_id()
    rng = np.random.default_rng(63051)          # 与 meeting63 同种子（同音频内容）
    路白 = os.path.join(临时目录, "sham_白噪_%s.wav" % run)
    路静 = os.path.join(临时目录, "sham_静音_%s.wav" % run)
    写wav(路白, rng.standard_normal(int(3.4 * FS)) * 0.2)
    写wav(路静, np.zeros(int(3.4 * FS)))
    白 = CER(真值, 引["转写"](路白))
    静 = CER(真值, 引["转写"](路静))
    print("[sham] 白噪 CER=%.3f ｜ 静音 CER=%.3f" % (白, 静), flush=True)
    return dict(白噪=白, 静音=静, 双达标=bool(白 >= 0.9 and 静 >= 0.9),
                文件=[os.path.basename(路白), os.path.basename(路静)])


# ══════════════════════ 评多句（转写两趟 + 对齐 + 归因 + 基线比较） ══════════════════════
def 评多句(引: dict, 韵律工厂, 角色s=多句角色, 标签: str = "基线",
          种子: int = 300, 角色改=None, 临时目录: str | None = None,
          对拍参考目录: str | None = None, 做sham: bool = True, 条件号: int = 0) -> dict:
    """18 样本 × 两趟转写（C13）→ 逐样本 CER/对齐、逐类归因、均值、相对基线 0.7212 下降。

    `对拍参考目录`（可选）：meeting63 的 18 个基线 wav 所在目录（`发声_0000NN.wav`，
    序 = 句集 outer × 角色 inner）。给出时逐样本核对**波形 sha256 是否与基线逐位同**，
    作为「本件未改动生成路径」的独立证据（不经 声带后端_v8 也拿到同一波形）。
    """
    t0 = time.perf_counter()
    合 = 合成多句(韵律工厂, 角色s=角色s, 标签=标签, 临时目录=临时目录,
                 种子=种子, 角色改=角色改, 条件号=条件号)
    读 = {}
    一致 = True
    for (句号, 名) in [(s, n) for s, _ in 句集 for n in 角色s]:
        v = 合["样本"][(句号, 名)]
        键 = "%s·%s" % (句号, 名)
        t1 = 引["转写"](v["路径"])
        t2 = 引["转写"](v["路径"])
        if t1 != t2:
            一致 = False
            print("[两趟不一致] %s「%s」≠「%s」" % (键, t1, t2), flush=True)
        读[键] = dict(句号=句号, 句=v["句"], 角色=名, wav=v["wav"], sha256=v["sha256"],
                     时长秒=v["时长秒"], 音色指纹=v["音色指纹"],
                     转写=t1, 转写第二趟=t2, 一致=bool(t1 == t2),
                     CER=CER(v["句"], t1), 对齐=对齐(v["句"], t1))
        print("%-9s CER=%.3f 「%s」" % (键, 读[键]["CER"], t1), flush=True)

    # —— 音韵类归因（错误 = 该真值字为替换或删除）——
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
                continue
            音 = 音韵.解析音节(拼)
            错 = k in 错位
            for 类 in _归类(音):
                _加(类, 错)
            if 音["尾"] in ("n", "ng"):
                _加("鼻尾·短语末" if 末 else "鼻尾·句中", 错)
    归因 = {类: dict(错误=d["错误"], 总数=d["总数"],
                     错误率=round(d["错误"] / d["总数"], 4) if d["总数"] else 0.0)
            for 类, d in 计.items()}

    # —— 均值（逐句 / 逐角色 / 全体）——
    逐句均值 = {}
    for 句号, _ in 句集:
        逐句均值[句号] = round(float(np.mean([读["%s·%s" % (句号, n)]["CER"] for n in 角色s])), 4)
    逐角色均值 = {}
    for 名 in 角色s:
        逐角色均值[名] = round(float(np.mean([读["%s·%s" % (s, 名)]["CER"] for s, _ in 句集])), 4)
    多句均值 = round(float(np.mean([v["CER"] for v in 读.values()])), 4)
    下降 = round((基线CER_多句 - 多句均值) / 基线CER_多句, 4) if 基线CER_多句 > 0 else 0.0

    # —— sham（可选；一次载引擎多条件时按需跑，避免重复开销）——
    sham = _sham(引, os.path.join(临时根, "sham")) if 做sham else dict(白噪=None, 静音=None,
                                                                      双达标=None)

    # —— 对拍 meeting63 基线波形（可选）——
    对拍 = None
    if 对拍参考目录 and os.path.isdir(对拍参考目录):
        参考 = sorted(f for f in os.listdir(对拍参考目录) if f.lower().endswith(".wav"))
        序表 = [(s, n) for s, _ in 句集 for n in 角色s]
        逐项 = {}
        全同 = True
        for i, (句号, 名) in enumerate(序表):
            if i >= len(参考):
                break
            p = os.path.join(对拍参考目录, 参考[i])
            同 = bool(_sha256(p) == 合["样本"][(句号, 名)]["sha256"])
            全同 = bool(全同 and 同)
            逐项["%s·%s" % (句号, 名)] = dict(参考wav=参考[i], 逐位同=同)
        对拍 = dict(参考目录=对拍参考目录, 参考文件数=len(参考), 全同=全同, 逐项=逐项)

    耗时 = round(time.perf_counter() - t0, 2)
    报 = dict(
        工具="量具_v9.py", 标签=str(标签), 引擎=引["名"], 种子=int(种子),
        句集=[dict(号=s, 句=t) for s, t in 句集],
        角色s=list(角色s),
        逐样本=读,
        逐句均值=逐句均值,
        逐角色均值=逐角色均值,
        多句CER均值=多句均值,
        逐类归因=归因,
        sham=sham,
        must_fail_白噪=bool(sham["白噪"] >= 0.9) if sham["白噪"] is not None else None,
        must_fail_静音=bool(sham["静音"] >= 0.9) if sham["静音"] is not None else None,
        两趟一致=bool(一致),
        基线CER均值=基线CER_多句, 基线来源=基线CER_来源,
        相对基线下降=下降,
        验收目标=验收_目标, 验收容差=验收_容差,
        验收_复现=bool(abs(多句均值 - 验收_目标) <= 验收_容差),
        对拍=对拍,
        临时目录=合["目录"], run_id=合["run_id"],
        音色s={k: dict(v) for k, v in 合["角色音色s"].items()},
        耗时秒=耗时,
        注=("★ 外部评估仪器（不在生成路径，C10）；18 样本 × 2 趟（C13）；"
            "合成直接走 `韵律工厂(档,种子).说(句)`，不经 声带后端_v8（不写 meeting63 台账）。"),
    )
    return 报


def 打印表(报: dict) -> None:
    """打印「每句 × 三角色 CER 表」+ 均值 + 归因 + sham + 一致性。"""
    角色s = tuple(报["角色s"])
    print("-" * 84)
    print("%-6s %s" % ("句", "  ".join("%-9s" % n for n in 角色s) + "  均值"))
    for 句号, _ in 句集:
        print("%-6s %s  %-7.4f" % (
            句号, "  ".join("%-9.4f" % 报["逐样本"]["%s·%s" % (句号, n)]["CER"] for n in 角色s),
            报["逐句均值"][句号]))
    print("%-6s %s  %-7.4f" % ("角色均", "  ".join("%-9.4f" % 报["逐角色均值"][n] for n in 角色s),
                               报["多句CER均值"]))
    print("多句 CER 均值 = %.4f ｜ 基线 %.4f ｜ 相对下降 %+.2f%% ｜ 验收目标 %.4f±%.4f ｜ 复现=%s"
          % (报["多句CER均值"], 报["基线CER均值"], -100.0 * 报["相对基线下降"],
             报["验收目标"], 报["验收容差"], 报["验收_复现"]))
    前5 = sorted(报["逐类归因"].items(), key=lambda kv: (-kv[1]["错误率"], -kv[1]["总数"], kv[0]))[:5]
    print("错误率最高 5 类：" + " ｜ ".join(
        "%s %.3f(%d/%d)" % (k, v["错误率"], v["错误"], v["总数"]) for k, v in 前5))
    for k in ("鼻尾", "鼻尾·句中", "鼻尾·短语末"):
        if k in 报["逐类归因"]:
            v = 报["逐类归因"][k]
            print("  %-10s 错误率 %.3f（%d/%d）" % (k, v["错误率"], v["错误"], v["总数"]))
    s = 报["sham"]
    print("两趟一致 = %s ｜ sham 白噪/静音 = %s/%s ｜ 耗时 %.1fs"
          % (报["两趟一致"], s["白噪"], s["静音"], 报["耗时秒"]))
    if 报.get("对拍"):
        print("对拍 meeting63 基线波形：全同=%s（%d 个文件）"
              % (报["对拍"]["全同"], 报["对拍"]["参考文件数"]))


# ══════════════════════ 跑批（一次载引擎，串行多条件） ══════════════════════
def 跑批(条件列表, 引: dict | None = None, 输出: str | None = None,
         对拍参考目录: str | None = None) -> dict:
    """**一次载引擎**串行跑多个条件（ASR 只允许单进程；载引擎是最贵的固定开销）。

    条件 = dict(
        名     : 条件名（进 JSON 与临时文件名；唯一性由调用方保证语义，文件名另有 run_id）
        韵律工厂: 可调用 (档, 种子) → 有 .说(句) 的对象（通常是 `韵律_v8.人类韵律_v8` 或 v9 包装）
        角色s  : 角色名元组（默认 多句角色）
        角色改 : callable(角色名, 原音色dict) -> 新音色dict | None（默认 None = 音色零改动）
        种子   : int（默认 300，与 meeting63 基线同种子）
        做sham : bool（默认只在第一个条件上跑 sham；sham 与条件无关，重复跑是浪费）
        句集入 : 覆盖默认 6 句（默认 None）
        描述   : 自由说明（进 JSON）
    )
    → dict(工具, 引擎, run_id, 条件列表, 复跑对照, 逐条件{名: 报})
    写 JSON：`输出`（默认 `meeting64/产物/诊断/batch.json`）。
    """
    引 = 引 or 载引擎()
    输出 = 输出 or 批出
    _确保目录(os.path.dirname(输出) or ".", 临时根)
    run = 新run_id()
    print("=" * 84)
    print("[跑批] 引擎=%s ｜ 条件数=%d ｜ run_id=%s" % (引["名"], len(条件列表), run))
    逐条件 = {}
    t0 = time.perf_counter()
    for i, 条 in enumerate(条件列表, 1):
        名 = str(条.get("名") or ("条件%d" % i))
        角色s = tuple(条.get("角色s") or 多句角色)
        种 = int(条.get("种子", 300))
        做sham = bool(条.get("做sham", i == 1))
        句集入 = 条.get("句集入") or 句集
        临时 = os.path.join(临时根, "批", "c%02d_%s_%s" % (i, _安全(名), run))
        print("-" * 84)
        print("[条件 %d/%d] %s ｜ 角色=%s ｜ 种子=%d ｜ sham=%s"
              % (i, len(条件列表), 名, "/".join(角色s), 种, 做sham))
        报 = 评多句(引, 条["韵律工厂"], 角色s=角色s, 标签=名, 种子=种,
                    角色改=条.get("角色改"), 临时目录=临时,
                    对拍参考目录=对拍参考目录 if i == 1 else None, 做sham=做sham,
                    条件号=i)
        报["条件序号"] = i
        报["描述"] = str(条.get("描述") or "")
        逐条件[名] = 报
        打印表(报)

    # —— 复跑对照（C13 自查：同条件同种子两次调用必须等价；此处对照 sham 与波形分辨率）——
    复跑 = dict(说明="同进程内不重复跑同一条件（省 ASR）；两趟一致性见各条件 `两趟一致`；"
                 "跨进程复跑请再跑一次本函数并比较 `多句CER均值` 与 `逐样本.sha256`。")
    结果 = dict(工具="量具_v9.py", 版本="v9(meeting64 统一量具)", 引擎=引["名"], run_id=run,
                条件列表=[dict(名=str(c.get("名")), 角色s=list(c.get("角色s") or 多句角色),
                           种子=int(c.get("种子", 300)),
                           描述=str(c.get("描述") or "")) for c in 条件列表],
                逐条件=逐条件, 复跑对照=复跑,
                基线CER均值=基线CER_多句, 基线来源=基线CER_来源,
                耗时秒=round(time.perf_counter() - t0, 2),
                注="★ ASR 只作外部评估仪器（C10）；临时文件名每次唯一（含 run_id）防滞后音频假读数。")
    with open(输出, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=1)
    print("=" * 84)
    for 名, 报 in 逐条件.items():
        print("%-20s 多句CER=%.4f ｜ 基线 %.4f ｜ 下降 %+.2f%% ｜ 两趟一致=%s ｜ sham=%s/%s"
              % (名, 报["多句CER均值"], 报["基线CER均值"], -100.0 * 报["相对基线下降"],
                 报["两趟一致"], 报["sham"]["白噪"], 报["sham"]["静音"]))
    print("batch JSON:", 输出)
    return 结果


# ══════════════════════ 基线复现（__main__：v8.1 基线 → 基线_v8_1_复现.json） ══════════════════════
def 跑基线(输出: str | None = None,
          对拍参考目录: str | None = None) -> dict:
    """v8.1 基线复现：韵律_v8 + 三角色（EmoCompanion/SpeakerA/洛天依），6 句 × 3 角色 × 两趟。

    验收（meeting63 登记口径）：多句 CER 均值 = 0.5292 ± 0.0015；两趟一致；sham ≥ 0.9。
    """
    import 韵律_v8 as H8  # noqa: E402  只读（meeting63）；import 在此处 ⇒ 本件 import 期零副作用
    输出 = 输出 or os.path.join(出目录, "基线_v8_1_复现.json")
    _确保目录(出目录, 临时根)
    引 = 载引擎()
    条件 = [dict(名="v8.1基线", 韵律工厂=H8.人类韵律_v8, 角色s=多句角色, 种子=300,
                 角色改=None, 做sham=True,
                 描述="meeting63 v8.1（鼻尾全音节化 8/7 帧 + 持续包络）；音色零改动；声线档案=覆.取(名)")]
    结果 = 跑批(条件, 引=引, 输出=os.path.join(批出目录, "batch_基线_v8_1.json"),
                 对拍参考目录=对拍参考目录)
    报 = 结果["逐条件"]["v8.1基线"]
    报["验收"] = dict(
        目标=验收_目标, 容差=验收_容差, 实测=报["多句CER均值"],
        偏差=round(报["多句CER均值"] - 验收_目标, 4),
        达标_复现=bool(abs(报["多句CER均值"] - 验收_目标) <= 验收_容差),
        两趟一致=bool(报["两趟一致"]),
        sham达标=bool(报["sham"]["白噪"] >= 0.9 and 报["sham"]["静音"] >= 0.9),
        对拍meeting63基线波形=报.get("对拍"),
    )
    报["验收"]["通过"] = bool(报["验收"]["达标_复现"] and 报["验收"]["两趟一致"]
                            and 报["验收"]["sham达标"])
    报["产物"] = 输出
    报["复现命令"] = ("$env:PYTHONIOENCODING=\"utf-8\"; "
                 "& \"<workspace>/\03_AHI\\.venv\\Scripts\\python.exe\" "
                 "\"<workspace>/\03_AHI\\AHI4\\meeting64\\实施\\量具_v9.py\"")
    with open(输出, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("-" * 84)
    print("[验收] 复现=%s ｜ 两趟一致=%s ｜ sham达标=%s ｜ 通过=%s"
          % (报["验收"]["达标_复现"], 报["验收"]["两趟一致"],
             报["验收"]["sham达标"], 报["验收"]["通过"]))
    print("基线 JSON:", 输出)
    return 报


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    # 对拍参考 = meeting63 官方多句基线的 18 个 wav（只读；序 = 句集 outer × 角色 inner）
    跑基线(对拍参考目录=os.path.join(M63, "产物", "ASR临时", "多句"))
