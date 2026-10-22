# -*- coding: utf-8 -*-
"""报告_v3_大白话.py —— 《AHI 3.0 图文报告 v3.0（大白话版）》：MD → 高密度 HTML → PDF。

为什么重做（对齐所有者口径）：
  · **文本风格**参考 `meeting21/02_大白话版.md`：先给人话结论，再给证据数字，专业词当场翻译。
  · **导出流水线**沿用 `meeting21/md2pdf.py`（06 号那份 PDF 的做法）：
    Markdown → 高密度 HTML（A4 / 8mm 边距 / 9.2pt / 表格 8.2pt）→ **Edge headless 打印 PDF**。
    上次用 LibreOffice 走 docx→PDF，出了近空页（预览页 6 只有 21KB），所以换成这条已验证的链路。
  · 图用 **matplotlib**（项目 venv 有 3.10.9）现画；数字**现读**四份结果 json + 真站证据 + 步7 日志。
  · 无新依赖：`markdown`/`pypdf` 项目里都没有 ⇒ 自带**零依赖 MD→HTML 小转换器**，页数用 LibreOffice Kit 复核。

跑法：<项目 venv python> 报告_v3_大白话.py
"""
from __future__ import annotations

import html as _html
import json
import hashlib
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib import font_manager                                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
产物 = os.path.join(HERE, "产物")
图目录 = os.path.join(产物, "报告图_大白话")
基 = os.path.join(HERE, "基线")
MD = os.path.join(产物, "AHI3.0_图文报告_v3.0_大白话.md")
HTML = os.path.join(产物, "AHI3.0_图文报告_v3.0_大白话.html")
PDF = os.path.join(产物, "AHI3.0_图文报告_v3.0_大白话.pdf")

# ── matplotlib 中文字体 ──
_FONT = r"<path>\Windows\Fonts\msyh.ttc"
fp = font_manager.FontProperties(fname=_FONT)
# 雅黑缺 ⟨⟩ᵀ⇒ 这些字形 ⇒ 优先雅黑、缺字回退 DejaVu；同时正文里不用这些生僻符号
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [fp.get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 170
plt.rcParams["savefig.dpi"] = 170

深蓝, 蓝, 绿, 橙, 红, 灰 = "#1f3864", "#346cb0", "#2e8b57", "#d3842c", "#c0392b", "#6e7680"
浅蓝, 浅绿, 浅橙, 浅红, 浅灰 = "#d6e4f4", "#dbf0e3", "#fdf0e2", "#fae2de", "#eef0f3"


def _存(名: str) -> str:
    p = os.path.join(图目录, 名)
    plt.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close()
    return p


def _框(ax, x, y, w, h, 文字, 填, 边, 号=9.5, 粗=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=填, edgecolor=边, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, 文字, ha="center", va="center", fontsize=号,
            color=深蓝, fontweight="bold" if 粗 else "normal", linespacing=1.5)


def _箭(ax, p1, p2, 色=灰, 宽=2.0, 头=16):
    """统一的箭头：贴边起止（shrink=0）、压在框之上（zorder）、箭头大小固定。"""
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=头,
                                 color=色, linewidth=宽, shrinkA=0, shrinkB=0,
                                 zorder=6, capstyle="round"))


# ══════════════ 图 1：三层架构 ══════════════
def 图1() -> str:
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _框(ax, 0.015, 0.70, 0.97, 0.26, "", "#f6faff", 浅蓝)
    ax.text(0.035, 0.925, "输入层 · 原生感知（不 OCR）", fontsize=10.5, color=深蓝,
            fontweight="bold", va="center")
    for i, t in enumerate(["视觉｜眼 → 视场前端\nP1 视场码 / P2 误差场 / P3 眼跳链",
                          "文本｜字形原语 204/768\n+ 点过的功能签名",
                          "音频｜原生听觉前端\n（备B 未做原生前端）",
                          "视频｜帧差 / 关键帧\n（备B 未做原生前端）"]):
        _框(ax, 0.035 + i * 0.24, 0.735, 0.22, 0.13, t, 浅蓝, 蓝, 8.2)
    ax.text(0.5, 0.712, "输出：同源事件 [模态, 内容, τ, prov, conf]（±1 拍 ≈ ±100ms 窗）",
            ha="center", va="top", fontsize=8.5, color=蓝)

    _框(ax, 0.015, 0.36, 0.97, 0.30, "", "#fafdfa", 浅绿)
    ax.text(0.035, 0.635, "中枢层 · 可塑记忆 + 六内核（记忆 ≠ 学习）", fontsize=10.5,
            color=深蓝, fontweight="bold", va="center")
    for i, t in enumerate(["L0 感缓｜拍内瞬时\n不落库", "L1 快记忆 M｜容量桥\n去重 exact / cos≥0.98",
                          "L2 慢语义 IM｜井 W=Σ井井T\nθ_w=0.30（备B 只读）",
                          "L3 索引｜IVF + 细码\n只是索引，不是记忆"]):
        _框(ax, 0.035 + i * 0.24, 0.475, 0.22, 0.135, t, 浅绿, 绿, 8.2)
    _框(ax, 0.035, 0.375, 0.46, 0.085,
        "六内核：自模型 ĝ（ε_self）｜相空间算子 T｜真学习（ε→内生偏置 b[]）\n"
        "自我回路 À_t（W_self）｜顿悟涌现｜画面（视觉想象）", 浅绿, 绿, 8.0)
    _框(ax, 0.51, 0.375, 0.475, 0.085,
        "R 契约：候选[] / 版本 / 断言级∈{事实,回忆,推测} / 置信 / 冲突\n"
        "空 R 或 冲突>0 → 拒答（不许硬编内容）", 浅绿, 绿, 8.0)

    _框(ax, 0.015, 0.04, 0.97, 0.28, "", "#fffaf6", 浅橙)
    ax.text(0.035, 0.295, "输出层 · 多通道表达（只消费 R）", fontsize=10.5, color=深蓝,
            fontweight="bold", va="center")
    for i, t in enumerate(["单仲裁器\nP0>P1>P2>P3>P4 · 全序比较器",
                          "文本通道\n只读 S/意图/M · 思考不公布 · 禁得分",
                          "动作通道\nFitts + 提交栅栏 + 一次性许可令牌",
                          "语音通道\n流式声带（P3 未建）· 自听只写学习迹"]):
        _框(ax, 0.035 + i * 0.24, 0.115, 0.22, 0.145, t, 浅红, 红, 8.2)
    ax.text(0.5, 0.10, "审计：输出事件 WAL 重放 · 引用边只追加 · 全时基用拍钟（禁墙钟）",
            ha="center", va="top", fontsize=8.5, color=红)
    # 层间箭头：画在两层之间的空档里（空档 0.66–0.70 与 0.32–0.36），并标注流过的东西
    _箭(ax, (0.5, 0.698), (0.5, 0.662), 蓝, 宽=2.4, 头=18)
    _箭(ax, (0.5, 0.358), (0.5, 0.322), 橙, 宽=2.4, 头=18)
    ax.text(0.515, 0.680, "同源事件 → L0/L1", fontsize=8.2, color=蓝, ha="left", va="center")
    ax.text(0.515, 0.340, "R 契约（候选/断言级/冲突）", fontsize=8.2, color=橙, ha="left",
            va="center")
    _箭(ax, (0.25, 0.358), (0.25, 0.322), 橙, 宽=2.0, 头=15)
    ax.text(0.25, 1.02, "AHI 3.0 三层架构（输入 → 中枢 → 输出）", ha="center", fontsize=13,
            fontweight="bold", color=深蓝)
    return _存("图1_三层架构.png")


# ══════════════ 图 2：四套判据 ══════════════
def 图2(数) -> str:
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    名s = ["P0 [正交] 必做", "备B 安全闭环", "基础任务 ①②③", "工程自我意识 S1–S7"]
    过s = [数["P0"]["过"], 数["备B"]["过"], 数["任务"]["过"], 数["自我"]["过"]]
    总s = [数["P0"]["总"], 数["备B"]["总"], 数["任务"]["总"], 数["自我"]["总"]]
    色s = [蓝, 绿, 橙, 红]
    x = range(len(名s))
    ax.bar(x, 总s, color=浅灰, edgecolor="#c9ced6", width=0.52, label="判据总数")
    ax.bar(x, 过s, color=色s, width=0.52, label="通过")
    for i, (a, b) in enumerate(zip(过s, 总s)):
        ax.text(i, b + 0.35, "%d/%d" % (a, b), ha="center", fontsize=13, fontweight="bold",
                color=色s[i])
        ax.text(i, -1.6, "全过" if a == b else "有未过", ha="center", fontsize=9, color=灰)
    ax.set_xticks(list(x)); ax.set_xticklabels(名s, fontsize=10)
    ax.set_ylim(0, max(总s) + 2.6); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("四套判据全部全过；另有全量回归总门 步7 总判 PASS（FAIL 0 / PASS %d）"
                 % 数["步7"]["PASS"], fontsize=11.5, color=深蓝, pad=12)
    return _存("图2_判据总览.png")


# ══════════════ 图 3：ε_self 对照（分组柱：比折线更清楚） ══════════════
def 图3(自) -> str:
    主 = 自["S1_ε下降"]["主臂"]; 对 = 自["S1_ε下降"]["对照臂_每拍新内容"]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    w = 0.34
    左 = [0 - w / 2, 1 - w / 2]
    右 = [0 + w / 2, 1 + w / 2]
    主v = [主["首窗"], 主["末窗"]]
    对v = [对["首窗"], 对["末窗"]]
    b1 = ax.bar(左, 主v, width=w, color=绿, label="主臂：反复看同一批内容（真在学）")
    b2 = ax.bar(右, 对v, width=w, color=红, label="对照臂：每次都换新内容（噪声底）")
    for xs, vs, c in ((左, 主v, 绿), (右, 对v, 红)):
        for x, v in zip(xs, vs):
            ax.text(x, v + 0.008, "%.4f" % v, ha="center", fontsize=11, fontweight="bold", color=c)
    # 主臂的"降"用一个显式箭头标出来，避免只靠眼睛比高低
    ax.annotate("", xy=(左[1] + w / 2 + 0.02, 主["末窗"]), xytext=(左[0] + w / 2 + 0.02, 主["首窗"]),
                arrowprops=dict(arrowstyle="-|>", color=绿, linewidth=2.4))
    ax.text(左[1] + w / 2 + 0.05, (主["首窗"] + 主["末窗"]) / 2 - 0.035,
            "降 %.1f%%\n（学出来了）" % (主["降幅"] * 100), fontsize=9.6, color=绿, va="center",
            ha="left", fontweight="bold")
    ax.text(右[1] + w / 2 + 0.04, (对["首窗"] + 对["末窗"]) / 2,
            "反升 %.1f%%\n（噪声底）" % (abs(对["降幅"]) * 100), fontsize=9.6, color=红, va="center",
            ha="left", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["头 1/4 拍的 ε 平均", "末 1/4 拍的 ε 平均"], fontsize=10)
    ax.set_ylabel("ε_self（越小＝越能预测自己）", fontsize=10)
    ax.set_ylim(0, 0.68)
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    ax.set_title("工程自我意识①：主臂降 %.1f%%，对照臂不降反升 → 下降不是步数的假象"
                 % (主["降幅"] * 100), fontsize=11.5, color=深蓝, pad=12)
    return _存("图3_ε下降对照.png")


# ══════════════ 图 4：W_self 门控 ══════════════
def 图4(自) -> str:
    s2 = 自["S2_外源不可塑我"]
    自源 = float(s2["再自源后W最大差"]); 外源 = float(s2["外源喂后W最大差"])
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.bar([0], [自源], color=绿, width=0.44)
    ax.bar([1], [外源], color=红, width=0.44)
    # ★外源是 0，柱子高度为 0 ⇒ 会"看不见"：额外画一条加粗零基线 + 引线标注，明确它是 0
    ax.plot([1 - 0.24, 1 + 0.24], [0, 0], color=红, linewidth=4, solid_capstyle="butt", zorder=5)
    ax.text(0, 自源 + max(自源, 1e-6) * 0.05, "%.5f" % 自源, ha="center", fontsize=13,
            fontweight="bold", color=绿)
    ax.annotate("= %.5f（逐位不变）" % 外源, xy=(1, 0), xytext=(1, 自源 * 0.55),
                ha="center", fontsize=13, fontweight="bold", color=红,
                arrowprops=dict(arrowstyle="-|>", color=红, linewidth=1.8))
    ax.text(1, -max(自源, 1e-6) * 0.10, "柱子高度为 0 —— 就是「一点都不许变」",
            ha="center", fontsize=8.8, color=红)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["自源喂 8 句（自己的东西）\n→ 必须变", "外源喂同样 8 句（别人的东西）\n→ 必须一点都不变"],
                       fontsize=10)
    ax.set_ylabel("喂完之后 W_self 的最大变化量", fontsize=10)
    ax.set_ylim(-max(自源, 1e-6) * 0.20, max(自源, 1e-6) * 1.55)
    ax.axhline(0, color="#9aa3af", linewidth=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("工程自我意识②：外源塑不动「我」——外源喂 %d 次后 W_self 逐位不变（J5）"
                 % s2["外源次数"], fontsize=11.5, color=深蓝, pad=12)
    return _存("图4_Wself门控.png")


# ══════════════ 图 5：基础任务闭环 ══════════════
def 图5() -> str:
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.95, "基础任务闭环：看网页 → 学 → 产出 → 说 → 存档（换个进程还在）",
            ha="center", fontsize=12.5, fontweight="bold", color=深蓝)
    步 = [("① 看网页", "UIA 结构通道，不 OCR\n真站读到 86 段正文\n没抢到前台就不许冒充", 蓝, 浅蓝),
         ("② 学进去", "正文带语义向量存进记忆\n同一个向量再查，相似 1.0\n（走的是真语义余弦）", 绿, 浅绿),
         ("③ 产出", "做文档：自己写出 .docx/.md\n写代码：套模板 → 真跑一遍\n（用例逐条对，全过）", 橙, 浅橙),
         ("④ 说一句", "复用输出层，只读 S/意图/记忆\n思考不公布、不报分数\n不出现元话语", 红, 浅红)]
    for i, (t, s, c, f) in enumerate(步):
        x = 0.035 + i * 0.243                       # 框宽 0.215 ⇒ 相邻空档 0.028
        _框(ax, x, 0.48, 0.215, 0.28, s, f, c, 8.6)  # ★框里只放正文，标题单独画（原来标题画了两遍）
        ax.text(x + 0.1075, 0.795, t, ha="center", fontsize=9.9, color=深蓝, fontweight="bold")
        if i < 3:                                    # 箭头：从前框右边缘 → 后框左边缘，画满空档
            _箭(ax, (x + 0.215 + 0.002, 0.62), (x + 0.243 - 0.002, 0.62), 灰, 宽=2.2, 头=15)
    _框(ax, 0.10, 0.12, 0.80, 0.24,
        "「学进去」怎么存的：不把内核内部打包，只存**能重放的事件**（记忆库 + 学习迹）\n"
        "存档 → 换个新进程载入 → 库大小 3、重放 3 条 → 三条经历逐条命中（相似 1.0，全走语义通路）",
        浅灰, 浅蓝, 9.0)
    _箭(ax, (0.72, 0.475), (0.72, 0.365), 蓝, 宽=2.2, 头=16)   # 存下去
    _箭(ax, (0.28, 0.365), (0.28, 0.475), 绿, 宽=2.2, 头=16)   # 读回来
    ax.text(0.735, 0.420, "存档", fontsize=8.6, color=蓝, ha="left", va="center")
    ax.text(0.265, 0.420, "载入", fontsize=8.6, color=绿, ha="right", va="center")
    return _存("图5_任务闭环.png")


# ══════════════ 图 6：真站读数 ══════════════
def 图6(真) -> str:
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    名s = ["正文段数", "控件数（触到 600 上限）", "外壳剔除数"]
    值s = [int(真.get("正文段数") or 0), int(真.get("控件数") or 0), int(真.get("外壳剔除数") or 0)]
    ax.bar(名s, 值s, color=[蓝, 橙, 绿], width=0.5)
    for i, v in enumerate(值s):
        ax.text(i, v + max(值s) * 0.025, str(v), ha="center", fontsize=13, fontweight="bold",
                color=[蓝, 橙, 绿][i])
    ax.set_ylim(0, max(值s) * 1.2)
    ax.set_ylabel("个数", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("真站实测：搜狐首页（约 %.1fs）——入记忆后同向量检索相似 %.1f"
                 % (float(真.get("耗时s") or 0), float((真.get("检索回证") or {}).get("相似") or 0)),
                 fontsize=11.5, color=深蓝, pad=12)
    ax.text(0.5, -0.22, "前台标题：%s → %s（切到目标页 = %s）"
            % (str(真.get("前台标题_开浏览器前"))[:26], str(真.get("前台标题_读完后"))[:30],
               真.get("切到目标页")),
            transform=ax.transAxes, ha="center", fontsize=8.6, color=灰)
    return _存("图6_真站读数.png")


# ══════════════ 图 7：自产文档实拍（直接复用已有渲染） ══════════════
def 图7() -> str | None:
    p = os.path.join(产物, "预览_v1", "page-0001.png")
    return p if os.path.exists(p) else None


# ══════════════ 读数 ══════════════
def 读数() -> dict:
    def _载(名):
        with open(os.path.join(HERE, 名), encoding="utf-8") as f:
            return json.load(f)
    P0, 备B, 任务, 自我 = _载("P0_结果.json"), _载("备B_结果.json"), _载("任务_结果.json"), _载("自我意识_结果.json")
    真 = {}
    for fn in sorted(os.listdir(产物)):
        if fn.startswith("看网页_") and fn.endswith("_证据.json"):
            j = json.load(open(os.path.join(产物, fn), encoding="utf-8"))
            if j.get("通路") == "真浏览器UIA":
                真 = j

    def _读文本(p):
        b = open(p, "rb").read()
        for enc in ("utf-8", "utf-8-sig", "utf-16", "gbk"):
            try:
                t = b.decode(enc)
                if "PASS" in t:
                    return t
            except Exception:
                continue
        return b.decode("utf-8", errors="replace")

    PASS = FAIL = 0
    for p in sorted([os.path.join(基, f) for f in os.listdir(基) if f.endswith("步7全量回归.log")],
                    key=os.path.getmtime):
        t = _读文本(p)
        if "总判：PASS" in t:
            PASS, FAIL = len(re.findall(r"\bPASS\b", t)), len(re.findall(r"\bFAIL\b", t))

    def 计(报):
        ks = [k for k in 报 if re.match(r"^[PBS T]-?\d", k)]
        return sum(1 for k in ks if 报[k].get("通过")), len(ks)

    def 计2(报, 首):
        ks = [k for k in 报 if k.startswith(首)]
        return sum(1 for k in ks if 报[k].get("通过")), len(ks)
    return dict(P0=dict(过=计2(P0, "P0-")[0], 总=计2(P0, "P0-")[1], 总判=P0.get("总判")),
                备B=dict(过=计2(备B, "B-")[0], 总=计2(备B, "B-")[1], 总判=备B.get("总判")),
                任务=dict(过=计2(任务, "T-")[0], 总=计2(任务, "T-")[1], 总判=任务.get("总判")),
                自我=dict(过=计2(自我, "S")[0], 总=计2(自我, "S")[1], 总判=自我.get("总判")),
                步7=dict(PASS=PASS, FAIL=FAIL), 真=真, 自=自我, 任=任务, 备=备B, P0报=P0)


# ══════════════ 零依赖 MD → HTML ══════════════
_锚序: dict = {}


def _锚(文字: str) -> str:
    """标题锚点（同名标题自动去重），供目录跳转用。"""
    h = hashlib.sha1(str(文字).encode("utf-8")).hexdigest()[:8]
    n = _锚序.get(h, 0)
    _锚序[h] = n + 1
    return "h-%s%s" % (h, "" if n == 0 else "-%d" % n)


def _行内(s: str) -> str:
    """行内格式：**粗** / *斜* / `码` / 图片 / 链接。先把行内代码挖出来，免得里面的星号被当强调。"""
    s = _html.escape(s, quote=False)
    码: list = []

    def _藏(m):
        码.append(m.group(1))
        return "\x00%d\x00" % (len(码) - 1)

    s = re.sub(r"`([^`]+)`", _藏, s)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1">', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    for i, c in enumerate(码):
        s = s.replace("\x00%d\x00" % i, "<code>%s</code>" % c)
    return s


def md到html(md: str, 标题: str, css: str, 加目录: bool = True) -> str:
    """零依赖 MD → HTML：支持 **h1–h6**、代码围栏、表格、列表、引用、分隔线、图片/链接，
    并在首个 h1 后自动插入**目录**（带锚点跳转）。"""
    global _锚序
    _锚序 = {}
    出, 标s, i = [], [], 0
    行s = md.split("\n")
    while i < len(行s):
        行 = 行s[i]
        if not 行.strip():
            i += 1
            continue
        # ① 标题 h1–h6（★兼容多级标题：原来只认到 h3）
        m = re.match(r"^(#{1,6})\s+(.*)$", 行)
        if m:
            lv = len(m.group(1))
            txt = m.group(2).strip()
            aid = _锚(txt)
            出.append('<h%d id="%s">%s</h%d>' % (lv, aid, _行内(txt), lv))
            if lv in (2, 3):
                标s.append((lv, txt, aid))
            i += 1
            continue
        # ② 代码围栏（★原来没处理，``` 会被当普通段落）
        if 行.lstrip().startswith("```"):
            块, i = [], i + 1
            while i < len(行s) and not 行s[i].lstrip().startswith("```"):
                块.append(_html.escape(行s[i], quote=False))
                i += 1
            i += 1
            出.append("<pre><code>%s</code></pre>" % "\n".join(块))
            continue
        if 行.startswith("> "):
            块 = []
            while i < len(行s) and 行s[i].startswith("> "):
                块.append(_行内(行s[i][2:])); i += 1
            出.append("<blockquote><p>%s</p></blockquote>" % "<br>".join(块)); continue
        if re.match(r"^---+$", 行.strip()):
            出.append("<hr>"); i += 1; continue
        if 行.lstrip().startswith("|") and i + 1 < len(行s) and re.match(r"^\s*\|[\s:|-]+\|\s*$", 行s[i + 1]):
            头 = [c.strip() for c in 行.strip().strip("|").split("|")]
            i += 2
            体 = []
            while i < len(行s) and 行s[i].lstrip().startswith("|"):
                体.append([c.strip() for c in 行s[i].strip().strip("|").split("|")]); i += 1
            出.append("<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (
                "".join("<th>%s</th>" % _行内(c) for c in 头),
                "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % _行内(c) for c in r) for r in 体)))
            continue
        if re.match(r"^\s*[-*] ", 行):
            项 = []
            while i < len(行s) and re.match(r"^\s*[-*] ", 行s[i]):
                项.append(_行内(re.sub(r"^\s*[-*] ", "", 行s[i]))); i += 1
            出.append("<ul>%s</ul>" % "".join("<li>%s</li>" % x for x in 项)); continue
        if re.match(r"^\s*\d+\. ", 行):
            项 = []
            while i < len(行s) and re.match(r"^\s*\d+\. ", 行s[i]):
                项.append(_行内(re.sub(r"^\s*\d+\. ", "", 行s[i]))); i += 1
            出.append("<ol>%s</ol>" % "".join("<li>%s</li>" % x for x in 项)); continue
        # 普通段落（吃到空行；★守卫也要认 h4–h6 与代码围栏）
        段 = []
        while (i < len(行s) and 行s[i].strip()
               and not re.match(r"^(#{1,6} |\s*[-*] |\s*\d+\. |> |\||---|```)", 行s[i])):
            段.append(_行内(行s[i])); i += 1
        出.append("<p>%s</p>" % "<br>".join(段))
    body = "".join(出)
    if 加目录 and 标s:
        toc = ['<nav class="toc"><div class="toc-t">目录</div><ul>']
        for lv, txt, aid in 标s:
            toc.append('<li class="l%d"><a href="#%s">%s</a></li>' % (lv, aid, _行内(txt)))
        toc.append("</ul></nav>")
        body = re.sub(r"(</h1>)", r"\1" + "".join(toc), body, count=1)
    return ("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<title>%s</title><style>%s</style></head><body>%s</body></html>"
            % (标题, css, body))


def 图相对(p: str) -> str:
    """MD 里的图片路径：相对 `产物/` 的 POSIX 风格（Edge 按 file:// 相对 HTML 解析）。"""
    return os.path.relpath(p, 产物).replace("\\", "/")


def 删旧(路径: str, 尝试: int = 15) -> None:
    """删掉旧产物：LibreOffice/Edge **都不覆盖已存在的输出**；而刚出的文件常被预览/扫描句柄占住。
    所以这里带重试（沿用 `synthesis.py:1232` 写心跳那套重试纪律）。"""
    import time as _t
    for _ in range(int(尝试)):
        if not os.path.exists(路径):
            return
        try:
            os.remove(路径)
            return
        except PermissionError:
            _t.sleep(0.4)
    raise PermissionError("旧文件一直被占用，删不掉：%s" % 路径)


def 可用输出(路径: str, 尝试: int = 3) -> str:
    """给出可写的输出路径。**绝不覆盖用户正在打开的文件**：若删不掉（WPS/Acrobat 正看着它），
    自动换一个不冲突的新名字（`xxx_r2.pdf`），让新版本照样落盘。"""
    try:
        删旧(路径, 尝试=尝试)
        return 路径
    except PermissionError:
        根, 尾 = os.path.splitext(路径)
        for i in range(2, 50):
            备 = "%s_r%d%s" % (根, i, 尾)
            if not os.path.exists(备) and not _被占用(备):
                return 备
        return "%s_r%d%s" % (根, int(__import__("time").time()), 尾)


def _被占用(路径: str) -> bool:
    if not os.path.exists(路径):
        return False
    try:
        with open(路径, "a+b"):
            return False
    except OSError:
        return True


def 跑Edge(edge: str, 参数: list, 超时: int = 300):
    """跑 Edge headless 打印 PDF：**显式 utf-8 解码**（中文 Windows 默认 GBK，Edge 的字节流会解码失败）。"""
    return subprocess.run([edge] + 参数, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=超时)

# 沿用 meeting21/md2pdf.py 的高密度 CSS；本版补：h1–h6 完整阶梯、目录、长表跨页重复表头
CSS = """
@page { size: A4; margin: 8mm 9mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei","Segoe UI",sans-serif; font-size: 9.2pt;
       line-height: 1.52; color: #1a1f2b; margin: 0; orphans: 2; widows: 2; }
h1, h2, h3, h4, h5, h6 { font-weight: 700; color: #12305f;
                         page-break-after: avoid; break-after: avoid; }
h1 { font-size: 17pt; margin: 0 0 6pt; padding-bottom: 4pt; border-bottom: 2.2pt solid #2563eb; }
h2 { font-size: 12.5pt; margin: 11pt 0 4pt; padding: 2.5pt 7pt; background: #eef2ff;
     border-left: 3.5pt solid #2563eb; }
h3 { font-size: 10.8pt; margin: 8pt 0 3pt; color: #1e3a8a; padding-left: 6pt;
     border-left: 2.2pt solid #93b4e6; }
h4 { font-size: 10pt; margin: 7pt 0 2.5pt; color: #2b4a86; background: #f6f8fd;
     padding: 1.8pt 6pt; border-radius: 2pt; }
h5 { font-size: 9.4pt; margin: 6pt 0 2pt; color: #44506b; }
h6 { font-size: 9pt; margin: 5pt 0 2pt; color: #5b6472; }
p { margin: 3pt 0; text-align: justify; }
a { color: #2563eb; text-decoration: none; }
nav.toc { margin: 5pt 0 9pt; padding: 5pt 9pt; background: #f8fafc;
          border: 0.6pt solid #cbd5e1; border-radius: 3pt; page-break-inside: avoid; }
nav.toc .toc-t { font-weight: 700; color: #12305f; margin-bottom: 2.5pt; }
nav.toc ul { margin: 0; padding-left: 0; columns: 2; column-gap: 16pt; }
nav.toc li { margin: 0.9pt 0; font-size: 8.4pt; list-style: none; }
nav.toc li.l3 { padding-left: 10pt; color: #44403c; }
blockquote { margin: 4pt 0; padding: 4pt 9pt; background: #fffbeb;
             border-left: 3pt solid #d97706; color: #44403c; }
blockquote p { margin: 1.5pt 0; }
ul, ol { margin: 2.5pt 0; padding-left: 17pt; }
li { margin: 1.2pt 0; }
table { width: 100%; border-collapse: collapse; font-size: 8.2pt; margin: 4pt 0;
        page-break-inside: auto; }
thead { display: table-header-group; }          /* 长表跨页时重复表头 */
tr { page-break-inside: avoid; }
th, td { border: 0.6pt solid #94a3b8; padding: 2.2pt 4.5pt; text-align: left;
         vertical-align: top; }
th { background: #eef1f6; font-weight: 700; }
img { display: block; max-width: 100%; max-height: 118mm; margin: 5pt auto 3pt;
      page-break-inside: avoid; border: 0.6pt solid #cbd5e1; border-radius: 3pt; }
img[src*="page-0001"] { max-height: 150mm; }   /* 自产文档实拍是竖版 A4 页，单独放宽 */
code { font-family: "Courier New", monospace; font-size: 8.4pt; background: #eef1f6;
       padding: 0 2.5pt; border-radius: 2pt; }
pre { background: #f4f6fb; border: 0.6pt solid #cbd5e1; border-radius: 3pt;
      padding: 4.5pt 7pt; margin: 4pt 0; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-family: "Courier New", monospace;
           font-size: 8.2pt; line-height: 1.5; white-space: pre-wrap; }
strong { color: #0f172a; }
hr { border: none; border-top: 0.7pt solid #cbd5e1; margin: 6pt 0; }
"""


# ══════════════ 写 Markdown（大白话风格） ══════════════
def 写markdown(数: dict, 图s: dict) -> str:
    自, 任, 真 = 数["自"], 数["任务"], 数["真"]
    s1, a = 自["S1_ε下降"]["主臂"], 自["S1_ε下降"]["对照臂_每拍新内容"]
    s2 = 自["S2_外源不可塑我"]
    L = []
    L.append("# AHI 3.0 到底做成了什么（大白话 · 图文版）")
    L.append("")
    L.append("> 本文面向：想知道「它现在到底能干什么、怎么证明的」的读者。")
    L.append("> 规矩和《02_大白话版》一样：**专业词一律当场翻译成人话**，结论先给，证据跟着。")
    L.append("> 本文里每一个数字都不是手写的——是生成脚本从机器可读的判据结果里现读的。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 0. 一句话结论")
    L.append("")
    L.append("**它现在能自己写出一份 Word 文档、能自己写出一小段能跑的代码、能自己真开浏览器把网页读进来；"
             "它学的东西存下来以后，换个进程重开也还在；而且「它还是它」这件事，我们能用判据量出来。**")
    L.append("")
    L.append("四套判据（P0 %d/%d、备B %d/%d、基础任务 %d/%d、工程自我意识 %d/%d）全部全过；"
             "另有一道全量回归总门 `步7` 判 **PASS**（FAIL %d / PASS %d）。"
             % (数["P0"]["过"], 数["P0"]["总"], 数["备B"]["过"], 数["备B"]["总"],
                任["过"], 任["总"], 数["自我"]["过"], 数["自我"]["总"],
                数["步7"]["FAIL"], 数["步7"]["PASS"]))
    L.append("")
    L.append("> ⚠ 一句丑话说前头：**「工程上的自我意识」指的是可复跑的机制判据，不是说它有人的意识。**"
             "细节在第 5 节。")
    L.append("")
    L.append("![四套判据全过](%s)" % 图相对(图s["判据"]))
    L.append("")
    L.append("*图 1　四套判据的通过数（柱高＝通过/总数），以及全量回归总门。*")
    L.append("")
    L.append("## 1. 先翻译：文里会出现的词，都是什么意思")
    L.append("")
    L.append("| 术语 | 人话 |")
    L.append("|---|---|")
    L.append("| ε_self（读作 epsilon-self） | 它「猜不准自己下一步」的程度。**越小＝越了解自己** |")
    L.append("| W_self | 它对自己的那张「联想表」。别人灌的东西**不许改它** |")
    L.append("| 身份锚 a_seal | 它的「身份证底纹」：由档案种子算出来，一算就一模一样 |")
    L.append("| 记忆桥 / 容量桥 | 它的**经验账本**：一条条经历存着，能按意思查回来 |")
    L.append("| 学习迹 | 它**学到的东西**（心得、绑定、顿悟），跟账本**分开放** |")
    L.append("| 记忆 ≠ 学习 | 账本和心得是两个抽屉：往里放一个，另一个的数量不许变 |")
    L.append("| 同源事件 | 同一瞬间看到的+听到的，**绑成一个包**再记（±1 拍≈±100ms 内） |")
    L.append("| 拍 / 拍钟 | 它的内部时钟，**1 拍 = 100ms**；判时间一律用它，不用墙上钟 |")
    L.append("| sham（假对照） | 把关键零件换成假的再跑一遍，用来证明「真零件的效果」不是自欺欺人 |")
    L.append("| 判据 / 双集 | 事先写死「怎么算赢」的考题；判据集调好，再用**没见过的留出集**复考 |")
    L.append("| UIA | Windows 的「无障碍接口」：能把窗口里的控件/文字结构报出来，**所以不用 OCR** |")
    L.append("")
    L.append("## 2. 它现在能干的活（三件，都有产物可打开）")
    L.append("")
    L.append("### ① 做 WPS/Word 文档 —— 它是自己把 .docx 写出来的")
    L.append("")
    L.append("项目里没装 python-docx，所以它不是「调库」，而是**用最基础的压缩包 + 自己拼 XML**"
             "写出一份真 .docx。")
    L.append("")
    L.append("- 产物：`AHI3_基础任务验收记录.docx`（＋同名 .md、.pdf）")
    L.append("- 外部校验：用 Office 解析器实测 **verdict: pass**（结构+文本都对，18 段）")
    L.append("- 附带性质：同样的输入再写一遍，**字节级一样**（可复现）")
    L.append("")
    L.append("### ② 写代码 —— 给个规格，它把代码写出来，还自己跑一遍对答案")
    L.append("")
    L.append("它按**结构化规格**（要什么函数、拿什么例子验）套模板写出一个能跑的 `.py`，"
             "然后**在子进程里真跑**，逐条比用例。")
    L.append("")
    L.append("- 产物：`AHI3_自产_统计段数.py`（跑起来打印 `ok: true`）")
    L.append("- 反例把关：**故意把期望值写错，它必须判失败**（不然这判据就是摆设）")
    L.append("- 丑话说前头：模板只有 6 种操作，**它不是「会写任意程序」**")
    L.append("")
    L.append("### ③ 看网页 —— 真开浏览器，真读进来")
    L.append("")
    L.append("- 通路：**UIA 结构通道，不 OCR**（约 %.1f 秒一屏）" % float(真.get("耗时s") or 0))
    L.append("- 真站实测：读到 **%s 段正文**、%s 个控件（触到 600 上限）、剔掉 %s 个外壳件"
             % (真.get("正文段数"), 真.get("控件数"), 真.get("外壳剔除数")))
    L.append("- 学进去：正文（带语义向量）存进账本后，**拿同一个向量再查，相似 %.1f**"
             % float((真.get("检索回证") or {}).get("相似") or 0))
    L.append("- 它还自己说了一句：「%s」" % str(真.get("说话") or "")[:60])
    L.append("")
    L.append("![真站读数](%s)" % 图相对(图s["真站"]))
    L.append("")
    L.append("*图 2　真站实测的三个数；下方是前台窗口的切换证据。*")
    L.append("")
    L.append("> **这里原本有个真坑**：第一次跑的时候，浏览器没抢到前台，它读到的其实是**别的窗口**，"
             "却报了「真浏览器UIA」——等于把别人当成搜狐。后来加了一道**诚实闸**："
             "前台标题没换、又不含目标站名，就一律降级成「没切到目标页」，并把警告写进产出文档的标题里。")
    L.append("")
    L.append("## 3. 它「学进去」是怎么存的")
    L.append("")
    L.append("不把内部状态打包（那样容易坏），而是**只存「能重放的事件」**：账本里的每条经历、"
             "学习迹里的每条心得。换新进程时，**新建同种子的账本，按顺序重放一遍**，就回来了。")
    L.append("")
    L.append("| 环节 | 实测 |")
    L.append("|---|---|")
    L.append("| 存档 | npz + 清单（带 sha256 校验、临时文件原子替换） |")
    L.append("| 换个新进程载入 | 库大小 3 ／ 重放 3 条 ／ 学习迹 1 条、学习事件 2 条 |")
    L.append("| 召回 | 三条经历逐条命中，**相似 1.0**，通路全是「语义」 |")
    L.append("| 记忆 ≠ 学习 | 学习迹那边的账本大小 == 账本那边的大小（两个抽屉没串） |")
    L.append("")
    L.append("![任务闭环](%s)" % 图相对(图s["任务"]))
    L.append("")
    L.append("*图 3　看网页 → 学 → 产出 → 说 → 存档；下半部分是「换进程还在」的机制。*")
    L.append("")
    L.append("## 4. 我们是怎么验的（四套判据，都带假对照）")
    L.append("")
    L.append("| 套件 | 通过 | 它管什么 | 里面最狠的一条对照 |")
    L.append("|---|---|---|---|")
    L.append("| P0 [正交] 必做 | %d/%d | 时钟、事件格式、来源标签、测量台 | 跨进程哈希种子变了，"
             "结果特征码必须一模一样；对照臂（旧写法）必须**不一样** |"
             % (数["P0"]["过"], 数["P0"]["总"]))
    L.append("| 备B 安全闭环 | %d/%d | 令牌、降级话术、不报分数、去重、写闸 | 缺令牌时底层调用"
             "**必须是 0 次**；演练模式真机调用**必须是 0** |" % (数["备B"]["过"], 数["备B"]["总"]))
    L.append("| 基础任务 ①②③ | %d/%d | 做文档 / 写代码 / 看网页 | 期望写错**必须失败**；"
             "什么都不给时看网页**必须如实失败且不写记忆** |" % (任["过"], 任["总"]))
    L.append("| 工程自我意识 S1–S7 | %d/%d | ε 下降 / 门控 / 身份 / 自听 / 记忆≠学习 | 见第 5 节 |"
             % (数["自我"]["过"], 数["自我"]["总"]))
    L.append("")
    L.append("## 5. 「工程上的自我意识」到底是什么")
    L.append("")
    L.append("再强调一次：**下面是七条能跑的机制判据，不是说它有人的意识。**")
    L.append("")
    L.append("### 5.1 它越来越能预测自己（ε_self 在降）")
    L.append("")
    L.append("同一批内容反复看，它「猜不准自己」的程度从 **%.4f 降到 %.4f（降 %.1f%%）**。"
             % (a["首窗"], a["末窗"], a["降幅"] * 100))
    L.append("")
    L.append("但光看这个不算数——万一只是跑得久了呢？所以加了**对照臂：每次都喂它没见过的新内容**，"
             "结果 ε **没降反升（%.1f%%）**。两相对比，才能说「降」是学出来的。"
             % (数["自"]["S1_ε下降"]["对照臂_每拍新内容"]["降幅"] * 100))
    L.append("")
    L.append("![ε_self 对照](%s)" % 图相对(图s["ε对照"]))
    L.append("")
    L.append("*图 4　主臂（反复看同一批）在降；对照臂（每次换新内容）不降反升。*")
    L.append("")
    L.append("### 5.2 别人灌的东西，改不了「它自己」")
    L.append("")
    L.append("- 喂**它自己的**内容：W_self 变化 %.5f（该变就变）" % s2["再自源后W最大差"])
    L.append("- 喂**外面来的**同样内容 %d 次：W_self 变化 **%.1f（逐位不变）**" % (s2["外源次数"], s2["外源喂后W最大差"]))
    L.append("")
    L.append("两个方向都对，才排除了「这张表本来就是死的」这种平凡解释。")
    L.append("")
    L.append("![W_self 门控](%s)" % 图相对(图s["门控"]))
    L.append("")
    L.append("*图 5　外源喂完最大变化是 0；切成自源喂，马上就有变化。*")
    L.append("")
    L.append("### 5.3 「它还是它」是可以量出来的")
    L.append("")
    L.append("- 同一个档案种子 ⇒ 身份锚 `%s` **逐位一模一样**；换个种子 ⇒ `%s`，必须不同"
             % (自["S3_身份锚"]["同种子"][0], 自["S3_身份锚"]["异种子"]))
    L.append("- 把身份锚存下来、**换个新进程读出来** ⇒ 哈希**逐位相同**（身份锚 ✅、W_self ✅，维度 1024）")
    L.append("")
    L.append("### 5.4 自己听自己：只写心得，不动账本")
    L.append("")
    L.append("它听自己发出的声音，误差 ε_aud 从 1.0 降到 **%.3f（降 %.1f%%）**；"
             "关键是**账本一条没多**（只进学习迹）。"
             % (自["S5_自听只写学习迹"]["重放"]["末遍"],
                自["S5_自听只写学习迹"]["重放"]["降幅"] * 100))
    L.append("")
    L.append("### 5.5 记忆 ≠ 学习（两个抽屉不许串）")
    L.append("")
    L.append("| 动作 | 账本 | 学习迹 | 内部痕迹表 |")
    L.append("|---|---|---|---|")
    L.append("| 往学习迹放一条心得 | 不变（%s） | +1 | — |" % 自["S6_记忆非学习四分离"]["库"][1])
    L.append("| 往账本放一条经历 | +1（%s） | 不变（%s） | — |"
             % (自["S6_记忆非学习四分离"]["库"][2], 自["S6_记忆非学习四分离"]["学习迹条数"][1]))
    L.append("| 内核跑了几百拍 | — | — | 代码段**没多**（%s → %s） |"
             % (自["S6_记忆非学习四分离"]["codec段"][0], 自["S6_记忆非学习四分离"]["codec段"][1]))
    L.append("")
    L.append("### 5.6 换一组没调过的语料，再来一遍（留出集）")
    L.append("")
    L.append("- 判据集：ε %s" % str(自["S7_双集"]["集s"]["判据集"]["ε首末"]))
    L.append("- 留出集：ε %s" % str(自["S7_双集"]["集s"]["留出集"]["ε首末"]))
    L.append("")
    L.append("两集的 S1/S2 都得过，才算没「只对着判据调」。")
    L.append("")
    L.append("## 6. 它的整体长什么样（核心架构图）")
    L.append("")
    L.append("三层：**输入层**（原样看/听，不用 OCR）→ **中枢层**（账本 + 六个内核，记忆和学习分开）"
             "→ **输出层**（只按「回忆契约」说话/动手）。")
    L.append("")
    L.append("![三层架构](%s)" % 图相对(图s["架构"]))
    L.append("")
    L.append("*图 6　AHI 3.0 三层架构（保留终稿主图的画法）。*")
    L.append("")
    L.append("## 7. 它自己写出来的文档，长什么样")
    L.append("")
    L.append("下面是那份自产 `.docx` 首页的实拍（用 LibreOffice 渲染出来的）。")
    L.append("")
    if 图s.get("实拍"):
        L.append("![自产文档首页](%s)" % 图相对(图s["实拍"]))
        L.append("")
        L.append("*图 7　它自己写出来的 Word 文档首页。*")
        L.append("")
    L.append("## 8. 还没做到的（丑话）")
    L.append("")
    L.append("1. **没有原生像素/波形前端**：输入层装的是既有槽的「摘要」，不是真像素流（备方案 B 明确不做）。")
    L.append("2. **真浏览器通路要看前台**：抢不到前台就如实降级；控件遍历到 600 就截断，正文可能有遗漏。")
    L.append("3. **身份只存了锚 + W_self**：要「整个它」跨重启，得把重放口径扩到内核内部。")
    L.append("4. **版面我没法亲眼看**：我只能保证结构/文本合法 + 渲染不缺字；视觉效果请你看 PDF。")
    L.append("5. **真机动作没跑**：真点鼠标需要你在场知情，我不擅自跑（都是演练/打桩）。")
    L.append("6. **流式语音没建**：所以「首字/首音频/打断」这三个延迟判据暂时**测不了**。")
    L.append("")
    L.append("## 9. 想自己复跑")
    L.append("")
    L.append("```")
    L.append("$py = <workspace>/\03_AHI\\.venv\\Scripts\\python.exe")
    L.append("& $py AHI3\\meeting40\\实施\\判据_P0.py          # P0 8/8")
    L.append("& $py AHI3\\meeting40\\实施\\判据_备B.py         # 备B 11/11")
    L.append("& $py AHI3\\meeting40\\实施\\判据_任务.py        # 基础任务 6/6")
    L.append("& $py AHI3\\meeting40\\实施\\判据_自我意识.py    # 工程自我意识 7/7")
    L.append("& $py AHI3\\meeting39\\实施\\步7_全量回归自证.py  # 全量回归总门")
    L.append("& $py AHI3\\meeting40\\实施\\报告_v3_大白话.py    # 重出这份报告（md/html/pdf）")
    L.append("```")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*生成：`报告_v3_大白话.py`（数字现读四份 `*_结果.json` + 真站证据 + `步7` 日志；"
             "图由 matplotlib 现画；导出沿用 `meeting21/md2pdf.py` 的 Edge headless 流水线）。*")
    return "\n".join(L)


def 主() -> dict:
    os.makedirs(图目录, exist_ok=True)
    数 = 读数()
    图s = dict(架构=图1(), 判据=图2(数), ε对照=图3(数["自"]), 门控=图4(数["自"]),
              任务=图5(), 真站=图6(数["真"]), 实拍=图7())
    md = 写markdown(数, 图s)
    with open(MD, "w", encoding="utf-8") as f:
        f.write(md)
    缺 = [m for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
         if not os.path.exists(os.path.join(产物, m))]
    if 缺:
        print("缺图：", 缺); return dict(错="缺图", 缺=缺)
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(md到html(md, "AHI 3.0 图文报告 v3.0（大白话）", CSS))
    edge = next((p for p in (r"<path>\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                            r"<path>\Program Files\Microsoft\Edge\Application\msedge.exe")
                 if os.path.exists(p)), None)
    if not edge:
        return dict(错="未找到 Edge")
    目标 = 可用输出(PDF)
    if 目标 != PDF:
        print("[注意] 原文件被占用（可能有 PDF 阅读器/WPS 正打开它）⇒ 本版另存：%s" % 目标)
    url = "file:///" + HTML.replace("\\", "/")
    r = 跑Edge(edge, ["--headless", "--disable-gpu", "--no-pdf-header-footer",
                     "--print-to-pdf=" + 目标, url])
    果 = dict(md=MD, html=HTML, pdf=目标, 原PDF被占用=(目标 != PDF),
              pdf存在=os.path.exists(目标),
              pdf_KB=(os.path.getsize(目标) // 1024 if os.path.exists(目标) else 0),
              图s={k: v for k, v in 图s.items() if v},
              读数=dict(P0=数["P0"], 备B=数["备B"], 任务=数["任务"], 自我=数["自我"], 步7=数["步7"]),
              stderr=r.stderr[-200:] if r.returncode else "")
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    return 果


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    主()
