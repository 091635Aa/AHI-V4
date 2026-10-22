# -*- coding: utf-8 -*-
"""步7_全量回归自证.py —— meeting39 步7 验收：全量回归 + 确定性重放 + 字回归。

契约（专家05 §2 步7：`跑 §4 回归清单；双 seed 逐位对比`｜依赖 步0–步6｜验收 §4.3 全部判据）：
  §4.2 必须回归清单：
    R1 文本态判据 V1–V7     `meeting38/实施/test_步5_文本态.py`
    R2 字回归（对基线）      `meeting38/实施/probe_字回归.py`
    R3 协议/快照            `meeting38/实施/test_步5_通用接口.py`、`探针_维度伸缩.py`
    R4 新内核逐步自证        步2/步3/步4/步5/步6 `*自证.py`（总判 PASS）
    R5 基链既有考试          meeting29/exam.py、exam2.py、meeting30/selflearn.py、
                            meeting28/v4_test.py、v5_test.py、v6_test.py（不退化）
    R6 帧冻结（基链/器官）    MD5(12) 比对 步0 指纹（10 文件，任何 M/D/R 即违规）
  §4.3 确定性门：
    C13-a 同 seed 两次运行：内核关 max|Δ|=0；内核开 max|Δ|≤1e-4（fp 地板）
    C13-b 载入/重放逐位一致（A5，由步4/步5 自证承载；此处复核 同流 ε_self 逐位）
    C13-c 新增随机扰动全走可快照独立 PRNG（新内核无全局 RNG）
    C13-d 新内核源码静态扫描无时间/拍数阈值、无常量顿悟判据、无直写状态
    + 双 seed 分化：不同 seed 应产出不同生命（防"恒等写死"）

纪律：本文件只运行/比对/扫描，**不改任何既有文件**。用法：python 步7_全量回归自证.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # AHI3 根（迁移：随文件定位；HERE=meeting39/实施）
for _p in (HERE, os.path.join(ROOT, "meeting31"), os.path.join(ROOT, "meeting38", "实施"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ★出口编码自愈（第五轮修复，2026-10-07）：不设 PYTHONIOENCODING 时，Windows 控制台默认 GBK，
#   本脚本打印子进程尾部（含 'ε/顿悟/站点' 等字符）会 UnicodeEncodeError **崩在最后一步**
#   ——于是回归结论根本没打出来（实测：崩前只到「【R 回归清单】FAIL」）。
#   这**不是**回归失败，是输出编码把结论截断了。四个模块判据早已有这条自愈，本脚本此前漏了。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from synthesis import 综合体                                    # noqa: E402
from 内核_顿悟涌现 import 顶覆盖_综合体W                          # noqa: E402

# 步0 §3 基线指纹（MD5 前 12 位）
#
# ★AHI3 迁移更新（2026-10-07，实证记录）：以下两处指纹**必须**从 AHI2.1 原值改为 AHI3 值，
#   因为 AHI3 迁移时对这两件做了「路径自包含」改动（见《改动清单》第二节）：
#     · meeting31/synthesis.py      ROOT 硬编码 "…\AHI2.1" → os.path.dirname(...__file__)
#     · meeting38/实施/文本态.py     AHI  硬编码 "…\AHI2.1" → os.path.dirname(M38)
#   我逐行 diff 核实：两件与 AHI2.1 的差异**各只有 1 行、且仅为上述路径**（unified diff 7 行含
#   上下文），**无任何语义/机制改动**。故这不是「破了帧冻结」，而是**基线需随迁移事实更新**。
#   原 AHI2.1 值保留在此备查：synthesis.py=F356CD2573C8、文本态.py=C97E00B43A7B。
#
# ★第二轮（2026-10-07 白天）再更新 `文本态.py`：除上面那行路径外，多了**一行**接线
#   —— `容量桥(种子=3800, 编码器=getattr(self.态, "编码", None))`（把语义向量接进记忆桥，
#   见《改动清单》第五节 #8/#9）。我 diff 核实：与 `备份_迁移前/实施/文本态.py`（=AHI2.1 原件）
#   的差异**仅两处**：①迁移的路径行 ②这行编码器接线。故指纹再次随之更新（非语义破坏）。
基线指纹 = {
    "meeting30/selflearn.py": "99BD219DB36F",
    "meeting29/learner.py": "5FF73FC732AC",
    "meeting28/dialogue_v2.py": "62E8D000FAE7",
    "meeting28/dialogue_v3.py": "56C88977C4D5",
    "meeting31/synthesis.py": "10632A778745",
    "meeting31/inner_speech.py": "1EC30FE980E8",
    "meeting38/实施/文本态.py": "940E2D9DF641",
    "ahi_theory/hdc_core.py": "E75189E5AACE",
    "ahi_theory/v2_engine.py": "3A351526C3DD",
    "ahi_theory/readout_head.py": "FED0374E2BCB",
}
# ★meeting43 更新（2026-10-11，实证记录）：5 个文件因「语言链修复」**有意**改动，指纹随之更新：
#   · meeting28/lang_codec.py   精化（#222）由空操作修为「IDF 加权上下文 + 去公共分量」语义码
#   · meeting28/dialogue_bridge.py 感知向量 随机探针 → 数据自定语义探针（可选 AHI_旧探针=1 回退）
#   · meeting28/dialogue_v3.py / meeting29/learner.py / meeting30/selflearn.py
#                               sys.path 注入由「ROOT=AHI2.1 的 meeting28/29/30」改为**本树**（迁移修正：
#                               原先会把整条语言链静默解析到 AHI2.1，令本树修复失效）
#   · meeting31/synthesis.py    新增 P3 解码通道 / P4 生成路 / P7 非线性决策（**均默认关**）
#   · meeting38/实施/文本态.py   新增 P8 运行期自诊钩子（**默认关**）
#   证据：AHI3/meeting43/实施/证据_语义码.md、证据_检索决策.md、证据_生成解码.md、
#         证据_非线性决策.md、证据_接口自诊.md；逐项回归见各判据脚本。
#   旧值保留备查：selflearn=B72B35AD75B7、learner=74A028481CEE、dialogue_v3=6A1EEABC5333、
#                synthesis=4817DC05B4E4、文本态=DC88B34C2496。
长流 = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。",
       "我慢慢翻了一页书。", "灯一直亮着。", "他坐在对面没说话。", "雨点敲着玻璃。",
       "为什么天空会变成红色呢", "我不太确定自己在想什么。", "突然想起很久以前的事。",
       "这和我上次以为的不一样。", "桌角的划痕是怎么来的。", "我好像漏掉了什么。"]


# ────────────────────────── 工具 ──────────────────────────
def md5_12(路径: str) -> str:
    with open(路径, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12].upper()


def 跑脚本(路径: str, cwd: str, 超时: int = 600):
    """跑一个回归子进程。

    ★第五轮修复：给子进程**显式**注入 `PYTHONIOENCODING=utf-8`。
    原先只给父进程设了 `encoding="utf-8"`（管的是**读**），子进程自己的 stdout 仍是控制台默认
    （Windows 中文环境 = GBK）⇒ 多个回归脚本（步2/步5/步6/selflearn/字回归）**崩在打印上**
    （`UnicodeEncodeError: 'gbk' codec ...`），表现为"回归 FAIL"，其实是**编码**而不是机制回归。
    实测对照：本脚本加 `PYTHONIOENCODING=utf-8` 跑 = 全 PASS；不加 = R4/R5/R2 五项假 FAIL。
    """
    环 = dict(os.environ)
    环["PYTHONIOENCODING"] = "utf-8"
    try:
        r = subprocess.run([sys.executable, 路径], cwd=cwd, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=超时,
                           env=环)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return None, "【超时>%ds】" % 超时
    except Exception as e:      # noqa: BLE001
        return None, "【异常】%r" % e


def 判无_FAIL(出: str):
    return ("FAIL" not in 出), 出.count("PASS"), 出.count("FAIL")


# ────────────────────────── R6：帧冻结（MD5 逐位） ──────────────────────────
def 判MD5():
    细 = {}
    ok = True
    for 相对, 期望 in 基线指纹.items():
        路径 = os.path.join(ROOT, 相对.replace("/", os.sep))
        实 = md5_12(路径) if os.path.exists(路径) else "MISSING"
        好 = (实 == 期望)
        ok = ok and 好
        细[相对] = dict(实=实, 期望=期望, 同=好)
    return dict(通过=ok, 条数=len(基线指纹), 细=细)


# ────────────────────────── §4.2 回归清单（子进程） ──────────────────────────
def 判回归清单():
    项 = []
    M38 = os.path.join(ROOT, "meeting38", "实施")
    M29 = os.path.join(ROOT, "meeting29")
    M30 = os.path.join(ROOT, "meeting30")
    M28 = os.path.join(ROOT, "meeting28")

    def 加(名, 路径, cwd, 期望s=(), 禁s=("FAIL",), 超时: int = 600):
        rc, 出 = 跑脚本(路径, cwd, 超时=超时)
        通过 = (rc == 0) and all(期望 in 出 for 期望 in 期望s) and \
               all(禁 not in 出 for 禁 in 禁s)
        项.append(dict(名=名, rc=rc, 通过=bool(通过),
                       期望=列表化(期望s), 尾=出.strip().splitlines()[-3:] if 出 else []))

    # R1 V1–V7：无 FAIL 且 V1..V7 齐现
    加("R1_V1-V7", os.path.join(M38, "test_步5_文本态.py"), M38,
       期望s=("V1", "V7"), 禁s=("FAIL",))
    # R3 协议/快照
    加("R3_通用接口", os.path.join(M38, "test_步5_通用接口.py"), M38, 期望s=("V10",))
    # ★R3_维度伸缩 单独放宽超时（第五轮实测）：它**自身**就要 614s（"学到 256 段自愿升维"要跑 283 轮），
    #   而默认门限 600s ⇒ **机器一忙就假超时**（实测：单独跑 614s 全 PASS；与另两个进程并发时 >600s 被判 FAIL）。
    #   这是**门限**问题，不是回归。拉到 1800s，避免"负载不同、结论不同"。
    加("R3_维度伸缩", os.path.join(M38, "探针_维度伸缩.py"), M38, 期望s=("D7",), 超时=1800)
    # R4 新内核逐步自证
    for n, f in (("R4_步2", "步2_自模型接线自证.py"), ("R4_步3", "步3_思核接线自证.py"),
                 ("R4_步4", "步4_真学习闭环自证.py"), ("R4_步5", "步5_自我回路自证.py"),
                 ("R4_步6", "步6_顿悟涌现自证.py")):
        加(n, os.path.join(HERE, f), HERE, 期望s=("总判：PASS",))
    # R5 基链既有考试
    加("R5_exam", os.path.join(M29, "exam.py"), M29)
    加("R5_exam2", os.path.join(M29, "exam2.py"), M29)
    加("R5_selflearn", os.path.join(M30, "selflearn.py"), M30)
    加("R5_v4", os.path.join(M28, "v4_test.py"), M28)
    加("R5_v5", os.path.join(M28, "v5_test.py"), M28)
    加("R5_v6", os.path.join(M28, "v6_test.py"), M28)

    # R2 字回归：解析差值行，要求全 0
    rc, 出 = 跑脚本(os.path.join(M38, "probe_字回归.py"), M38)
    行 = [l for l in 出.splitlines() if "差值" in l]
    数值 = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", 行[-1])] if 行 else []
    字通过 = (rc == 0) and bool(数值) and all(abs(x) < 1e-12 for x in 数值)
    项.append(dict(名="R2_字回归", rc=rc, 通过=bool(字通过),
                   期望="差值全0", 尾=行[-1:] if 行 else []))

    全过 = all(x["通过"] for x in 项)
    return dict(通过=全过, 项=项)


def 列表化(s):
    return list(s)


# ────────────────────────── §4.3 C13-a/b：同 seed 重放 + 双 seed 分化 ──────────────────────────
def _跑内核(seed: int, 开关: bool, N: int = 200):
    桥 = 顶覆盖_综合体W(seed=seed, 顿悟开关=开关, 学习开关=True, 自模型开关=True,
                       自我开关=True)
    for i in range(N):
        桥.步(长流[i % len(长流)])
    ε序列 = list(桥.drv.ε史) if getattr(桥, "drv", None) is not None else []
    迹W = None
    if 桥.自模型 is not None and 桥.自模型.W is not None:
        迹W = np.asarray(桥.自模型.W, dtype=np.float64)
    return dict(拍=int(桥.拍), ε序列=ε序列,
                W=迹W,
                事件拍s=list(桥.顿悟核.事件拍s) if 桥.顿悟核 else [],
                外化=出外化(桥))


def 出外化(桥):
    流 = list(桥.对话流)
    return repr([(int(桥.拍), str(流[-2:]) if 流 else "",
                  len(桥.账本), len(桥.codec._段))])


def 判C13a():
    # 内核关：两次同 seed 逐位恒等（max|Δ|=0）
    a = _跑内核(310, False)
    b = _跑内核(310, False)
    关同 = (a["ε序列"] == b["ε序列"] and a["拍"] == b["拍"] and a["外化"] == b["外化"])
    # 内核开：两次同 seed（fp 地板 ≤1e-4 的门；实测应恰好逐位=0）
    c = _跑内核(310, True)
    d = _跑内核(310, True)
    开同 = (c["ε序列"] == d["ε序列"] and c["拍"] == d["拍"] and c["事件拍s"] == d["事件拍s"]
            and c["外化"] == d["外化"])
    ΔW = 0.0
    if c["W"] is not None and d["W"] is not None:
        ΔW = float(np.max(np.abs(c["W"] - d["W"])))
    开同 = 开同 and (ΔW <= 1e-4)
    return dict(通过=(关同 and 开同), 内核关_逐位同=关同, 内核开_逐位同=开同,
                开态_W_maxΔ=ΔW, 拍=a["拍"], 事件数_开=len(c["事件拍s"]))


def 判双seed():
    """双 seed 分化：不同 seed ⇒ 不同生命（ε 序列/事件/外化不全同）；防"恒等写死"。"""
    a = _跑内核(310, True)
    b = _跑内核(311, True)
    分化 = (a["ε序列"] != b["ε序列"]) or (a["事件拍s"] != b["事件拍s"]) or (a["外化"] != b["外化"])
    return dict(通过=bool(分化), 分化=bool(分化),
                ε同=(a["ε序列"] == b["ε序列"]), 事件同=(a["事件拍s"] == b["事件拍s"]),
                事件数_310=len(a["事件拍s"]), 事件数_311=len(b["事件拍s"]))


# ────────────────────────── §4.3 C13-c/d：静态扫描（新内核） ──────────────────────────
def _剥注释文档串(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"#[^\n]*", "", src)
    return src


内核文件s = ["内核_自模型.py", "内核_相空间算子T.py", "内核_真学习.py",
            "内核_自我回路.py", "内核_顿悟涌现.py",
            "内核_画面.py"]      # ★第五轮新增第六层（画面/视觉想象）；同一套 C13-c/d 静态扫描
                                #   （扫的是"有没有时间阈值/常量顿悟阈值/直写引擎状态/全局 RNG"）


def 判静态扫描():
    细 = {}
    全过 = True
    for f in 内核文件s:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        src = _剥注释文档串(open(p, encoding="utf-8").read())
        时间阈 = re.findall(r"(?:拍数|拍|time|试次|断供拍数)\s*(?:>=|>|<=|<)\s*\d", src)
        常量顿悟 = re.findall(r"(?:ε[^\n]{0,12}|cosv|cos)\s*>=\s*0\.\d", src)
        直写 = re.findall(r"self\.(?:a_prev|i_prev|prev_c|消化债|饱足)\s*=", src)
        全局RNG = re.findall(r"(?:np\.random\.(?:rand|randn|random|seed)\b"
                            r"|(?<!default_)random\.(?:random|randint|seed)\b)", src)
        好 = not (时间阈 or 常量顿悟 or 直写 or 全局RNG)
        全过 = 全过 and 好
        细[f] = dict(好=好, 时间阈=时间阈, 常量顿悟=常量顿悟, 直写=直写, 全局RNG=全局RNG)
    return dict(通过=全过, 细=细)


# ────────────────────────── 主 ──────────────────────────
def 主():
    报 = {}
    md5 = 判MD5()
    清 = 判回归清单()
    ca = 判C13a()
    双 = 判双seed()
    扫 = 判静态扫描()
    报["R6_MD5帧冻结"] = md5
    报["R_回归清单"] = 清
    报["C13a_确定性重放"] = ca
    报["双seed分化"] = 双
    报["C13cd_静态扫描"] = 扫

    总 = (md5["通过"] and 清["通过"] and ca["通过"] and 双["通过"] and 扫["通过"])
    报["总判"] = bool(总)

    print("=" * 84)
    print("步7 · 全量回归 + 确定性重放 + 字回归（§4.2 清单 × §4.3 确定性门）自证")
    print("=" * 84)
    print("【R6 帧冻结 MD5(12)】%s（%d 文件）" % ("PASS" if md5["通过"] else "FAIL", md5["条数"]))
    for k, v in md5["细"].items():
        print("   %-32s %s %s" % (k, v["实"], "OK" if v["同"] else "MISMATCH(期望%s)" % v["期望"]))
    print("【R 回归清单】%s" % ("PASS" if 清["通过"] else "FAIL"))
    for x in 清["项"]:
        print("   %-16s %s  rc=%s  尾=%s"
              % (x["名"], "PASS" if x["通过"] else "FAIL", x["rc"], x["尾"]))
    print("【C13-a 确定性重放】%s（内核关逐位同=%s，内核开逐位同=%s，开态 W maxΔ=%s）"
          % ("PASS" if ca["通过"] else "FAIL", ca["内核关_逐位同"], ca["内核开_逐位同"],
             ca["开态_W_maxΔ"]))
    print("【双 seed 分化】%s（分化=%s；ε同=%s 事件同=%s；事件数 310=%d / 311=%d）"
          % ("PASS" if 双["通过"] else "FAIL", 双["分化"], 双["ε同"], 双["事件同"],
             双["事件数_310"], 双["事件数_311"]))
    print("【C13-c/d 静态扫描（新内核）】%s" % ("PASS" if 扫["通过"] else "FAIL"))
    for f, v in 扫["细"].items():
        print("   %-24s %s %s" % (f, "OK" if v["好"] else "VIOLATION",
                                 "" if v["好"] else str({k: v[k] for k in
                                                        ("时间阈", "常量顿悟", "直写", "全局RNG")
                                                        if v[k]})))
    print("-" * 84)
    print("总判：%s" % ("PASS" if 总 else "FAIL"))
    print("（未做任何「已完成」宣称——数字以上表为准）")
    print("=" * 84)

    出 = os.path.join(HERE, "步7_结果.json")
    with open(出, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    print("[已存] %s" % 出)
    return 报


if __name__ == "__main__":
    主()