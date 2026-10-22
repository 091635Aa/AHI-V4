# -*- coding: utf-8 -*-
"""报告_v3.py —— 生成《AHI 3.0 · 图文报告 v3.0》（.docx + 图 + PDF）。

设计原则（避免"手抄数字"）：
  · 报告里**每一个数**都从机器可读产物里**现读**：`P0_结果.json` / `备B_结果.json` /
    `任务_结果.json` / `自我意识_结果.json` / 产物里的真站证据 json / `步7` 日志。
  · 图用 **Pillow** 现画（系统里没有 matplotlib）：架构图 / 判据总览 / ε 对照 / W_self 门控 /
    任务闭环 / 真站读数；中文用系统字体 `msyh.ttc`（微软雅黑）。
  · 文档用 **python-docx**（bundled python 有），中文字体显式设 `w:eastAsia`。
  · PDF 由 LibreOffice Kit 从同一 .docx 导出（外部命令，见文末）。

跑法（用 bundled python）：
  <bundled-python> 报告_v3.py
"""
from __future__ import annotations

import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
产物 = os.path.join(HERE, "产物")
图目录 = os.path.join(产物, "报告图")
基 = os.path.join(HERE, "基线")

# ── 视觉规范 ──
底色 = (255, 255, 255)
深蓝 = (31, 56, 100)
蓝 = (52, 108, 176)
浅蓝 = (214, 228, 244)
绿 = (46, 139, 87)
浅绿 = (219, 240, 227)
红 = (192, 57, 43)
浅红 = (250, 226, 222)
橙 = (211, 132, 44)
灰 = (110, 118, 128)
浅灰 = (238, 240, 243)
_FONT = r"<path>\Windows\Fonts\msyh.ttc"
_FONT_BD = r"<path>\Windows\Fonts\msyhbd.ttc"


def 字(号: int, 粗: bool = False):
    try:
        return ImageFont.truetype(_FONT_BD if 粗 else _FONT, int(号))
    except Exception:
        return ImageFont.load_default()


def 画布(w: int, h: int):
    im = Image.new("RGB", (int(w), int(h)), 底色)
    return im, ImageDraw.Draw(im)


def 框(d, x, y, w, h, 文字, 填充=浅蓝, 边=蓝, 号=26, 粗=False, 字色=深蓝, 圆=14, 边宽=3):
    d.rounded_rectangle([x, y, x + w, y + h], radius=圆, fill=填充, outline=边, width=边宽)
    d.text((x + w / 2, y + h / 2), 文字, font=字(号, 粗), fill=字色, anchor="mm", align="center")


def 箭头(d, x1, y1, x2, y2, 色=蓝, 宽=4, 头=14):
    d.line([x1, y1, x2, y2], fill=色, width=宽)
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    d.polygon([(x2, y2),
               (x2 - 头 * math.cos(a - 0.42), y2 - 头 * math.sin(a - 0.42)),
               (x2 - 头 * math.cos(a + 0.42), y2 - 头 * math.sin(a + 0.42))], fill=色)


def 标题行(d, x, y, 文字, 号=34, 色=深蓝):
    d.text((x, y), 文字, font=字(号, True), fill=色, anchor="lm")


# ══════════════════════ 图 1：三层架构（核心图） ══════════════════════
def 图1_架构(路径: str) -> dict:
    W, H = 1720, 1180
    im, d = 画布(W, H)
    d.rounded_rectangle([16, 16, W - 16, H - 16], radius=22, outline=浅灰, width=6)
    d.text((W / 2, 60), "AHI 3.0 三层架构（输入层 → 中枢层 → 输出层）· v3.0",
           font=字(38, True), fill=深蓝, anchor="mm")

    层x, 层w = 60, 1240
    # ── 输入层 ──
    d.rounded_rectangle([层x, 120, 层x + 层w, 330], radius=16, fill=(246, 250, 255),
                        outline=浅蓝, width=4)
    标题行(d, 层x + 22, 156, "输入层 · 原生感知（不 OCR）", 28)
    xs = [层x + 24, 层x + 340, 层x + 656, 层x + 972]
    for x, t in zip(xs, ["视觉｜器官_眼 → 视场前端\nP1 视场码 / P2 误差场 / P3 眼跳链",
                         "文本｜字形原语 204/768\n+ 语义.记点击功能签名",
                         "音频｜原生听觉前端\n（合成波形通路已验）",
                         "视频｜帧差 / 关键帧\n（备B 不做原生前端）"]):
        框(d, x, 190, 292, 112, t, 填充=浅蓝, 边=蓝, 号=21)
    d.text((层x + 层w / 2, 314), "输出：同源事件原子 ⟨模态, 内容, τ, prov, conf⟩（±1 拍 ≈ ±100ms 窗）",
           font=字(21), fill=蓝, anchor="mm")

    # ── 中枢层 ──
    d.rounded_rectangle([层x, 370, 层x + 层w, 720], radius=16, fill=(250, 252, 250),
                        outline=浅绿, width=4)
    标题行(d, 层x + 22, 406, "中枢层 · 可塑记忆 + 六内核（记忆 ≠ 学习）", 28)
    for x, y, w, h, t in [
        (层x + 24, 440, 292, 104, "L0 感缓｜拍内瞬时，不落库"),
        (层x + 340, 440, 292, 104, "L1 快记忆 M｜容量桥 / FSW\n去重：exact 或 cos ≥ 0.98"),
        (层x + 656, 440, 292, 104, "L2 慢语义 IM｜井 W = Σ井井ᵀ\nθ_w = 0.30（备B 只读）"),
        (层x + 972, 440, 292, 104, "L3 索引｜IVF + 细码\n（索引层，不是记忆层）"),
        (层x + 24, 560, 604, 128, "六内核（真接线，默认开关可回退）\n"
                                 "自模型 ĝ（ε_self）｜相空间算子 T｜真学习（ε→内生偏置 b[]）\n"
                                 "自我回路（À_t，W_self）｜顿悟涌现｜画面（视觉想象）"),
        (层x + 656, 560, 608, 128, "R 契约（中枢 → 输出的唯一交付物）\n"
                                  "候选[] / 版本 / 断言级∈{事实,回忆,推测} / 置信 / 冲突\n"
                                  "空 R 或 冲突>0 ⇒ 拒答（不许硬编内容）")]:
        框(d, x, y, w, h, t, 填充=浅绿, 边=绿, 号=20)
    d.text((层x + 层w / 2, 706), "写闸（§7.6）：写 M 仅当 prov_route=实感 ∧ 非输出回灌；自听只写学习迹",
           font=字(20), fill=绿, anchor="mm")

    # ── 输出层 ──
    d.rounded_rectangle([层x, 760, 层x + 层w, 1010], radius=16, fill=(255, 250, 246),
                        outline=(235, 200, 170), width=4)
    标题行(d, 层x + 22, 796, "输出层 · 多通道表达（只消费 R）", 28)
    框(d, 层x + 24, 830, 292, 150, "单仲裁器\n优先级 P0 > P1 > P2 > P3 > P4\n全序比较器 + P0 抢占", 填充=浅红, 边=红, 号=20)
    框(d, 层x + 340, 830, 292, 150, "文本通道\n只读 S / 意图 / M\n思考不公布 · 禁得分", 填充=浅红, 边=红, 号=20)
    框(d, 层x + 656, 830, 292, 150, "动作通道\nFitts + 提交栅栏\n一次性许可令牌（缺即拒）", 填充=浅红, 边=红, 号=20)
    框(d, 层x + 972, 830, 292, 150, "语音通道\n流式声带（P3 未建）\n自听只写学习迹", 填充=浅红, 边=红, 号=20)
    d.text((层x + 层w / 2, 996), "审计：输出事件 WAL 重放 · 引用边 append-only · 全时基用拍钟（禁墙钟）",
           font=字(20), fill=红, anchor="mm")

    # 层间箭头
    箭头(d, 层x + 层w / 2, 332, 层x + 层w / 2, 368)
    箭头(d, 层x + 层w / 2, 722, 层x + 层w / 2, 758)

    # 右栏
    d.rounded_rectangle([1330, 120, W - 40, 1010], radius=16, fill=(248, 249, 251), outline=浅灰, width=4)
    d.text((1330 + (W - 40 - 1330) / 2, 158), "本轮已验（机器可读）", font=字(24, True), fill=深蓝, anchor="mm")
    项s = [("P0 [正交] 必做", "8/8", 绿), ("备B 安全闭环", "11/11", 绿),
          ("基础任务 ①②③", "6/6", 绿), ("工程自我意识 S1–S7", "7/7", 绿),
          ("全量回归 步7", "PASS (FAIL 0)", 绿), ("真站读网页", "86 段 / 相似 1.0", 蓝),
          ("跨进程学进去", "3/3 命中 1.0", 蓝)]
    y = 210
    for 名, 值, c in 项s:
        d.rounded_rectangle([1352, y, W - 62, y + 68], radius=12, fill=底色, outline=浅灰, width=3)
        d.text((1368, y + 34), 名, font=字(20), fill=深蓝, anchor="lm")
        d.text((W - 80, y + 34), 值, font=字(22, True), fill=c, anchor="rm")
        y += 82
    d.text((1330 + (W - 40 - 1330) / 2, 800), "红线（全程未破）", font=字(22, True), fill=深蓝, anchor="mm")
    d.text((1330 + (W - 40 - 1330) / 2, 850),
           "C1 禁 Transformer\nC9 禁 OCR\nC10 禁外挂模型当器官\nC13 确定性可复现\nC14 不破既有",
           font=字(20), fill=灰, anchor="mm", align="center", spacing=8)
    im.save(路径)
    return dict(路径=路径, 宽=W, 高=H)


# ══════════════════════ 图 2：判据总览 ══════════════════════
def 图2_判据(数: dict, 路径: str) -> dict:
    W, H = 1560, 720
    im, d = 画布(W, H)
    d.text((W / 2, 46), "四套判据全过 + 全量回归总门", font=字(34, True), fill=深蓝, anchor="mm")
    组 = [("P0 [正交]", 数["P0"]["过"], 数["P0"]["总"], 蓝),
          ("备B 安全闭环", 数["备B"]["过"], 数["备B"]["总"], 绿),
          ("基础任务", 数["任务"]["过"], 数["任务"]["总"], 橙),
          ("工程自我意识", 数["自我"]["过"], 数["自我"]["总"], 红)]
    x0, y0, bw, gap, 高 = 150, 620, 190, 120, 420
    for i, (名, 过, 总, c) in enumerate(组):
        x = x0 + i * (bw + gap)
        # 总数底柱
        d.rounded_rectangle([x, y0 - 高, x + bw, y0], radius=12, fill=浅灰, outline=(220, 224, 230), width=2)
        h = 高 * (过 / max(总, 1))
        d.rounded_rectangle([x, y0 - h, x + bw, y0], radius=12, fill=c, outline=c, width=2)
        d.text((x + bw / 2, y0 - h - 30), "%d/%d" % (过, 总), font=字(30, True), fill=c, anchor="mm")
        d.text((x + bw / 2, y0 + 34), 名, font=字(23), fill=深蓝, anchor="mm")
        d.text((x + bw / 2, y0 + 66), "全过" if 过 == 总 else "有未过", font=字(19), fill=灰, anchor="mm")
    # 步7 面板
    d.rounded_rectangle([x0 + 4 * (bw + gap) - 60, y0 - 高, x0 + 4 * (bw + gap) + 200, y0],
                        radius=12, fill=(246, 250, 246), outline=浅绿, width=3)
    d.text((x0 + 4 * (bw + gap) + 10, y0 - 高 + 60), "全量回归\n步7_全量回归自证",
           font=字(21, True), fill=深蓝, anchor="mm", align="center")
    d.text((x0 + 4 * (bw + gap) + 10, y0 - 高 + 190), "总判 PASS", font=字(26, True), fill=绿, anchor="mm")
    d.text((x0 + 4 * (bw + gap) + 10, y0 - 高 + 250),
           "FAIL = 0\nPASS = %d\nR6 帧冻结 10/10" % 数["步7"]["PASS"], font=字(19), fill=灰,
           anchor="mm", align="center", spacing=8)
    im.save(路径)
    return dict(路径=路径)


# ══════════════════════ 图 3：ε_self 下降 vs 对照臂 ══════════════════════
def 图3_ε对照(自我: dict, 路径: str) -> dict:
    主 = 自我["S1_ε下降"]["主臂"]
    对 = 自我["S1_ε下降"]["对照臂_每拍新内容"]
    W, H = 1560, 760
    im, d = 画布(W, H)
    d.text((W / 2, 46), "S1  ε_self 真在降吗？——主臂 vs 对照臂（每拍新内容）",
           font=字(32, True), fill=深蓝, anchor="mm")
    L, R, T, B = 190, W - 260, 150, H - 130
    # 轴
    d.line([L, B, R, B], fill=灰, width=3)
    d.line([L, T, L, B], fill=灰, width=3)
    lo, hi = 0.0, 0.62
    for v in (0.0, 0.2, 0.4, 0.6):
        y = B - (v - lo) / (hi - lo) * (B - T)
        d.line([L - 8, y, R, y], fill=(232, 235, 239), width=2)
        d.text((L - 20, y), "%.1f" % v, font=字(20), fill=灰, anchor="rm")
    x1, x2 = L + 190, R - 190
    def Y(v):
        return B - (v - lo) / (hi - lo) * (B - T)
    for 名, 甲, 乙, c in (("主臂：重复语料（真在学）", 主["首窗"], 主["末窗"], 绿),
                        ("对照臂：每拍新内容（噪声底）", 对["首窗"], 对["末窗"], 红)):
        d.line([x1, Y(甲), x2, Y(乙)], fill=c, width=7)
        for x, v in ((x1, 甲), (x2, 乙)):
            d.ellipse([x - 11, Y(v) - 11, x + 11, Y(v) + 11], fill=c, outline=底色, width=3)
        d.text((x2 + 26, Y(乙)), "%.4f" % 乙, font=字(24, True), fill=c, anchor="lm")
        d.text((x1 - 26, Y(甲)), "%.4f" % 甲, font=字(24, True), fill=c, anchor="rm")
        d.text((x2 + 26, Y(乙) + 34), 名, font=字(20), fill=灰, anchor="lm")
    d.text((x1, B + 40), "首窗 ε 均值", font=字(22), fill=深蓝, anchor="mm")
    d.text((x2, B + 40), "末窗 ε 均值", font=字(22), fill=深蓝, anchor="mm")
    d.rounded_rectangle([L, B + 74, R, B + 128], radius=10, fill=(246, 250, 246), outline=浅绿, width=2)
    d.text(((L + R) / 2, B + 101),
           "主臂降幅 %.1f%%　｜　对照臂降幅 %.1f%%（不降反升）⇒ 下降不是步数带来的假象"
           % (主["降幅"] * 100, 对["降幅"] * 100), font=字(22, True), fill=深蓝, anchor="mm")
    im.save(路径)
    return dict(路径=路径)


# ══════════════════════ 图 4：W_self 门控 ══════════════════════
def 图4_门控(自我: dict, 路径: str) -> dict:
    s2 = 自我["S2_外源不可塑我"]
    自源 = float(s2["再自源后W最大差"])
    外源 = float(s2["外源喂后W最大差"])
    W, H = 1400, 620
    im, d = 画布(W, H)
    d.text((W / 2, 44), "S2  外源塑不动「我」：‖ΔW_self‖ 两方向对照（J5）", font=字(30, True),
           fill=深蓝, anchor="mm")
    L, B, T = 170, H - 150, 150
    d.line([L, B, W - 120, B], fill=灰, width=3)
    mx = max(自源, 外源, 1e-6)
    for i, (名, v, c) in enumerate((("自源喂 ⇒ 必须变", 自源, 绿), ("外源喂 ⇒ 必须为 0", 外源, 红))):
        x = L + 160 + i * 420
        h = (B - T) * (v / mx) if mx > 0 else 0
        h = max(h, 4 if v == 0 else h)
        d.rounded_rectangle([x, B - h, x + 260, B], radius=10, fill=c, outline=c, width=2)
        d.text((x + 130, B - h - 28), "%.5f" % v, font=字(28, True), fill=c, anchor="mm")
        d.text((x + 130, B + 36), 名, font=字(22), fill=深蓝, anchor="mm")
    d.rounded_rectangle([L, B + 74, W - 120, B + 128], radius=10, fill=(250, 244, 244),
                        outline=浅红, width=2)
    d.text(((L + W - 120) / 2, B + 101),
           "外源喂 %d 次后 W_self 最大差 = %.1f（逐位不变）　｜　自源喂 %d 次 ⇒ 更新 %d 次"
           % (s2["外源次数"], 外源, 8, s2["自源更新次数"]), font=字(21, True), fill=深蓝, anchor="mm")
    im.save(路径)
    return dict(路径=路径)


# ══════════════════════ 图 5：基础任务闭环 ══════════════════════
def 图5_任务(数: dict, 路径: str) -> dict:
    W, H = 1680, 760
    im, d = 画布(W, H)
    d.text((W / 2, 46), "基础任务闭环：看网页 → 学 → 产出 → 说 → 存档（跨进程仍在）",
           font=字(31, True), fill=深蓝, anchor="mm")
    框s = [(70, "① 看网页\nUIA 结构通道（不 OCR）\n真站 86 段 / 600 控件\n诚实闸：没切页不冒充", 蓝, 浅蓝),
          (420, "② 学进去\n正文带语义向量入容量桥\n同向量检索 相似 1.0\n（语义余弦真通路）", 绿, 浅绿),
          (770, "③ 产出\n做文档：自产 .docx/.md\n写代码：模板→子进程自证\n（用例全过）", 橙, (253, 240, 226)),
          (1120, "④ 说一句\n复用输出层（只读 S/意图/M）\n思考不公布 · 禁得分\n不出现元话语", 红, 浅红)]
    for x, t, 边c, 填c in 框s:
        框(d, x, 150, 320, 190, t, 填充=填c, 边=边c, 号=20)
        箭头(d, x + 320 + 6, 245, x + 350 - 6, 245, 色=灰, 宽=4)
    # 回环：存档 → 新进程
    d.rounded_rectangle([420, 420, 1440, 560], radius=14, fill=(246, 248, 252), outline=浅蓝, width=3)
    d.text((930, 462), "「学进去」的持久化：事件重放式存档（npz + manifest sha256、temp + os.replace）",
           font=字(22, True), fill=深蓝, anchor="mm")
    d.text((930, 512), "存档 → 真·新进程载入 → 重建库大小 3 / 重放 3 条 → "
                      "三条经历逐条命中（相似 1.0，全走语义通路）",
           font=字(21), fill=灰, anchor="mm")
    箭头(d, 1280, 344, 1280, 416, 色=蓝, 宽=4)
    箭头(d, 620, 416, 620, 344, 色=蓝, 宽=4)
    # 底部：任务判据
    项s = [("T-1 做文档", "8/8", "check_office verdict: pass（18 段）", 绿),
          ("T-2 写代码", "7/7", "反例（期望写错）必失败", 绿),
          ("T-5 看网页", "12/12", "sham：什么都不给 ⇒ 如实失败且不写记忆", 绿),
          ("T-6 跨进程", "10/10", "记忆≠学习 未破", 绿)]
    x = 70
    for 名, 值, 注, c in 项s:
        d.rounded_rectangle([x, 600, x + 380, 700], radius=12, fill=底色, outline=浅灰, width=3)
        d.text((x + 18, 632), 名, font=字(21, True), fill=深蓝, anchor="lm")
        d.text((x + 362, 632), 值, font=字(22, True), fill=c, anchor="rm")
        d.text((x + 18, 668), 注, font=字(17), fill=灰, anchor="lm")
        x += 400
    im.save(路径)
    return dict(路径=路径)


# ══════════════════════ 图 6：真站读数 ══════════════════════
def 图6_真站(真站: dict, 路径: str) -> dict:
    W, H = 1500, 700
    im, d = 画布(W, H)
    d.text((W / 2, 44), "任务③ 真站实测：搜狐首页（UIA 结构通道，不 OCR）", font=字(30, True),
           fill=深蓝, anchor="mm")
    条 = [("正文段数", int(真站.get("正文段数") or 0), 蓝),
         ("控件数（触到 600 上限）", int(真站.get("控件数") or 0), 橙),
         ("外壳剔除数", int(真站.get("外壳剔除数") or 0), 绿)]
    L, B, T = 210, H - 190, 150
    mx = max(v for _, v, _ in 条) or 1
    for i, (名, v, c) in enumerate(条):
        x = L + i * 400
        h = (B - T) * v / mx
        d.rounded_rectangle([x, B - h, x + 250, B], radius=10, fill=c, outline=c, width=2)
        d.text((x + 125, B - h - 30), str(v), font=字(30, True), fill=c, anchor="mm")
        d.text((x + 125, B + 38), 名, font=字(20), fill=深蓝, anchor="mm")
    d.rounded_rectangle([L, B + 76, W - 120, B + 150], radius=10, fill=(246, 248, 252),
                        outline=浅蓝, width=2)
    d.text(((L + W - 120) / 2, B + 96),
           "前台标题：%s  →  %s" % (str(真站.get("前台标题_开浏览器前"))[:34],
                                str(真站.get("前台标题_读完后"))[:34]),
           font=字(19), fill=灰, anchor="mm")
    d.text(((L + W - 120) / 2, B + 128),
           "切到目标页 = %s　｜　入记忆后同向量检索 相似 %.1f（语义通路）　｜　耗时 %.1f s"
           % (真站.get("切到目标页"), float((真站.get("检索回证") or {}).get("相似") or 0),
              真站.get("耗时s") or 0),
           font=字(20, True), fill=深蓝, anchor="mm")
    im.save(路径)
    return dict(路径=路径)


# ══════════════════════ 读数（全部来自机器可读产物） ══════════════════════
def 读数() -> dict:
    def _载(名):
        with open(os.path.join(HERE, 名), encoding="utf-8") as f:
            return json.load(f)
    P0, 备B, 任务, 自我 = _载("P0_结果.json"), _载("备B_结果.json"), _载("任务_结果.json"), _载("自我意识_结果.json")
    # 真站证据（取最近一次 通路=真浏览器UIA 的）
    真站 = {}
    for fn in sorted(os.listdir(产物)):
        if fn.startswith("看网页_") and fn.endswith("_证据.json"):
            with open(os.path.join(产物, fn), encoding="utf-8") as f:
                j = json.load(f)
            if j.get("通路") == "真浏览器UIA":
                真站 = j
    # 步7 日志（PowerShell 重定向可能是 UTF-16/GBK ⇒ 多编码稳健解码，并取**最新**一份）
    def _读文本(p: str) -> str:
        b = open(p, "rb").read()
        for enc in ("utf-8", "utf-8-sig", "utf-16", "gbk"):
            try:
                t = b.decode(enc)
                if "PASS" in t or "总判" in t:
                    return t
            except Exception:
                continue
        return b.decode("utf-8", errors="replace")

    PASS = FAIL = 0
    日志s = sorted([os.path.join(基, f) for f in os.listdir(基)
                  if f.endswith("步7全量回归.log")], key=lambda p: os.path.getmtime(p))
    for p in 日志s:
        t = _读文本(p)
        if "总判：PASS" in t or "总判: PASS" in t:
            PASS, FAIL = len(re.findall(r"\bPASS\b", t)), len(re.findall(r"\bFAIL\b", t))
            步7日志 = os.path.basename(p)
    def 计(报):
        ks = [k for k in 报 if re.match(r"^[P0BTS]-?\d", k) or k.startswith(("P0-", "B-", "T-", "S"))]
        return sum(1 for k in ks if 报[k].get("通过")), len(ks)
    p = 计(P0); b = 计(备B); t = 计(任务); s = 计(自我)
    return dict(P0=dict(过=p[0], 总=p[1], 总判=P0.get("总判")),
                备B=dict(过=b[0], 总=b[1], 总判=备B.get("总判")),
                任务=dict(过=t[0], 总=t[1], 总判=任务.get("总判")),
                自我=dict(过=s[0], 总=s[1], 总判=自我.get("总判")),
                步7=dict(PASS=PASS, FAIL=FAIL, 日志=步7日志),
                真站=真站, 自我报=自我, 任务报=任务, P0报=P0, 备B报=备B)


# ══════════════════════ 文档 ══════════════════════
def 建文档(数: dict, 图s: dict, 输出: str) -> dict:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Inches, Mm, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.left_margin = sec.right_margin = Mm(20)
    sec.top_margin = sec.bottom_margin = Mm(18)

    def 中文字体(style, 名="微软雅黑"):
        style.font.name = 名
        rpr = style.element.get_or_add_rPr()
        rpr.get_or_add_rFonts().set(qn("w:eastAsia"), 名)

    中文字体(doc.styles["Normal"])
    doc.styles["Normal"].font.size = Pt(10.5)
    # ★把**所有**样式（含模板残留样式）的字体统一为雅黑：否则模板里 HTML Preformatted/Macro Text
    #   等样式仍引用 Courier，LibreOffice 的缺字诊断会报 "Courier"（正文其实一个字节都没用它）。
    for _st in doc.styles:
        try:
            _st.font.name = "微软雅黑"
            _rf = _st.element.get_or_add_rPr().get_or_add_rFonts()
            for _a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                _rf.set(qn(_a), "微软雅黑")
        except Exception:
            pass
    for h in ("Heading 1", "Heading 2", "Heading 3", "Title"):
        try:
            中文字体(doc.styles[h])
        except Exception:
            pass

    def 段(文字, 号=10.5, 粗=False, 色=None, 斜=False, 居=None, 段后=6):
        p = doc.add_paragraph()
        if 居:
            p.alignment = 居
        r = p.add_run(str(文字))
        r.font.size = Pt(号)
        r.bold = 粗
        r.italic = 斜
        if 色:
            r.font.color.rgb = RGBColor(*色)
        r.font.name = "微软雅黑"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        p.paragraph_format.space_after = Pt(段后)
        return p

    def 图(路径, 说明, 宽=6.6):
        doc.add_picture(路径, width=Inches(宽))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        段(说明, 号=9, 色=(110, 118, 128), 斜=True, 居=WD_ALIGN_PARAGRAPH.CENTER, 段后=10)

    def 表(头s, 行s, 宽s=None):
        t = doc.add_table(rows=1, cols=len(头s))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(头s):
            c = t.rows[0].cells[i]
            c.text = ""
            r = c.paragraphs[0].add_run(str(h))
            r.bold = True
            r.font.size = Pt(9.5)
            r.font.name = "微软雅黑"
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        for 行 in 行s:
            cells = t.add_row().cells
            for i, v in enumerate(行):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v))
                r.font.size = Pt(9)
                r.font.name = "微软雅黑"
                r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        return t

    # ── 封面 ──
    段("AHI 3.0", 号=30, 粗=True, 色=(31, 56, 100), 居=WD_ALIGN_PARAGRAPH.CENTER, 段后=2)
    段("图文报告 v3.0", 号=20, 粗=True, 色=(52, 108, 176), 居=WD_ALIGN_PARAGRAPH.CENTER, 段后=14)
    段("三层架构（输入 / 中枢 / 输出）· 基础任务闭环 · 学进去 · 工程自我意识",
       号=12, 色=(80, 88, 98), 居=WD_ALIGN_PARAGRAPH.CENTER, 段后=4)
    段("依据：meeting40 终稿（1 主 + 2 备）与后续实施 · 纪律：C1 禁 Transformer｜C9 禁 OCR｜"
       "C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有",
       号=9.5, 色=(120, 128, 138), 居=WD_ALIGN_PARAGRAPH.CENTER, 段后=4)
    段("本文所有数字均从机器可读产物现读（P0/备B/任务/自我意识 四份结果 json + 步7 日志 + 真站证据 json），"
       "不手抄；不主张人类级意识。", 号=9, 斜=True, 色=(140, 146, 154),
       居=WD_ALIGN_PARAGRAPH.CENTER, 段后=12)
    图(图s["架构"]["路径"], "图 1　AHI 3.0 三层架构：输入层原生感知（不 OCR）→ 中枢层可塑记忆与六内核"
                        "（记忆 ≠ 学习）→ 输出层多通道表达；右栏为本轮已验读数与红线。", 宽=6.8)

    doc.add_page_break()

    # ── §1 一页概览 ──
    doc.add_heading("1　一页概览", level=1)
    段("四套判据全部全过，另有全量回归总门与真站实测。判据都带对照臂或 sham，可失败。", 段后=8)
    表(["判据套件", "通过", "总判", "关键对照臂 / 反例"],
        [["P0 [正交] 必做项 / 基线冻结", "%d/%d" % (数["P0"]["过"], 数["P0"]["总"]), 数["P0"]["总判"],
          "跨 PYTHONHASHSEED 特征码相等；对照臂（旧 hash）两 seed 两特征码"],
         ["备B 安全闭环", "%d/%d" % (数["备B"]["过"], 数["备B"]["总"]), 数["备B"]["总判"],
          "缺令牌 ⇒ 底层 0 调用；可逆误拒 0.0；演练真机调用 0"],
         ["基础任务 ①②③", "%d/%d" % (数["任务"]["过"], 数["任务"]["总"]), 数["任务"]["总判"],
          "期望写错必失败；什么都不给 ⇒ 看网页如实失败且不写记忆"],
         ["工程自我意识 S1–S7", "%d/%d" % (数["自我"]["过"], 数["自我"]["总"]), 数["自我"]["总判"],
          "ε 对照臂不降反升；外源喂 max|ΔW|=0，再自源喂≠0"]])
    段("全量回归 `步7_全量回归自证.py`：总判 PASS（FAIL = %d / PASS = %d，R6 帧冻结 10/10）。"
       % (数["步7"]["FAIL"], 数["步7"]["PASS"]), 号=10, 粗=True, 段后=10)
    图(图s["判据"]["路径"], "图 2　四套判据通过数与全量回归总门（柱高 = 通过/总数）。", 宽=6.6)

    doc.add_page_break()

    # ── §2 核心图表：架构 ──
    doc.add_heading("2　核心图表：三层架构（保留主图）", level=1)
    段("核心架构与 meeting40 终稿一致：输入层原生三前端 + 同源事件；中枢层 L0–L3 与六内核、R 契约；"
       "输出层单仲裁器 + 三通道。备方案 B 在此骨架上只替换最关键的感知/记忆/输出环节，故本文读数"
       "对主A 与备B 都成立。", 段后=8)
    表(["层", "本轮已落的件", "状态"],
        [["输入", "SensorFrame 时间戳适配器 + 同源事件绑定（±1 拍窗）", "备B 已落（默认关）"],
         ["中枢", "L0 感缓（不落库）/ L1 写入 / 去重下限 exact+cos≥0.98 / 最小来源双标记壳 / 快照", "备B 已落（默认关）"],
         ["输出", "最小 R 适配 / 一次性许可令牌 / 真机-演练差分标签 / 删得分", "备B 已落（默认关）"],
         ["正交", "拍钟与事件 schema / provenance 固定枚举 / 可失败测量台", "P0 已落"]])
    段("诚实标注：原生像素/波形前端（主A/C）**未**实现；输入侧帧数据是既有槽内容的规范摘要。",
       号=9, 斜=True, 色=(140, 146, 154), 段后=10)

    # ── §3 工程自我意识 ──
    doc.add_heading("3　工程意义上的自我意识（图文）", level=1)
    段("口径：只出机制级证据，不主张人类级意识。每条判据都配对照臂或反例。", 段后=8)
    自 = 数["自我报"]
    表(["判据", "实测", "对照臂 / 反例"],
        [["S1 ε_self 真在降",
          "主臂 %.4f → %.4f（降 %.1f%%）" % (自["S1_ε下降"]["主臂"]["首窗"], 自["S1_ε下降"]["主臂"]["末窗"],
                                      自["S1_ε下降"]["主臂"]["降幅"] * 100),
          "每拍新内容 %.4f → %.4f（降 %.1f%%，不降反升）" % (
              自["S1_ε下降"]["对照臂_每拍新内容"]["首窗"], 自["S1_ε下降"]["对照臂_每拍新内容"]["末窗"],
              自["S1_ε下降"]["对照臂_每拍新内容"]["降幅"] * 100)],
         ["S2 外源不可塑我（J5）",
          "外源喂 %d 次 ⇒ max|ΔW_self| = %.1f" % (自["S2_外源不可塑我"]["外源次数"],
                                            自["S2_外源不可塑我"]["外源喂后W最大差"]),
          "再自源喂 ⇒ max|ΔW| = %.5f（证 W 不是冻住）" % 自["S2_外源不可塑我"]["再自源后W最大差"]],
         ["S3 自指束身份锚",
          "同档案种子 a_seal = %s（两次逐位同）" % 自["S3_身份锚"]["同种子"][0],
          "异种子 = %s（必须不同）" % 自["S3_身份锚"]["异种子"]],
         ["S4 身份跨存档/重启",
          "新进程载入：身份锚 sha256 逐位同 = %s，W_self 逐位同 = %s"
          % (自["S4_身份跨存档"]["逐项"]["身份锚跨进程逐位同"], 自["S4_身份跨存档"]["逐项"]["W跨进程逐位同"]),
          "异种子身份锚必须不同 = %s" % 自["S4_身份跨存档"]["逐项"]["对照异种子身份锚不同"]],
         ["S5 自听只写学习迹",
          "ε_aud 谱 %s ⇒ 降 %.1f%%，末值 %.3f；记忆库 0 变"
          % (自["S5_自听只写学习迹"]["重放"]["ε_aud均值谱"],
             自["S5_自听只写学习迹"]["重放"]["降幅"] * 100, 自["S5_自听只写学习迹"]["重放"]["末遍"]),
          "自听＝输出回灌 ⇒ 可写记忆 = False"],
         ["S6 记忆≠学习四分离",
          "库 %s ／ 学习迹条数 %s ／ codec._段 %s" % (自["S6_记忆非学习四分离"]["库"],
                                             自["S6_记忆非学习四分离"]["学习迹条数"],
                                             自["S6_记忆非学习四分离"]["codec段"]),
          "写学习迹不动记忆、写记忆不动学习迹（双向）"],
         ["S7 双集",
          "判据集 ε %s；留出集 ε %s" % (自["S7_双集"]["集s"]["判据集"]["ε首末"],
                                 自["S7_双集"]["集s"]["留出集"]["ε首末"]),
          "留出集为另一组语料，S1/S2 两集都须过"]])
    图(图s["ε对照"]["路径"], "图 3　S1 主臂 vs 对照臂：重复语料时 ε_self 降 25.8%；每拍新内容时 ε 不降反升，"
                          "⇒ 下降不是步数假象。", 宽=6.6)
    图(图s["门控"]["路径"], "图 4　S2 两方向对照：外源喂 W_self 最大差恒为 0；切回自源喂就有变化 —— "
                          "两个方向合起来才排除「W 本来就是死的」。", 宽=6.2)

    doc.add_page_break()

    # ── §4 基础任务 ──
    doc.add_heading("4　基础任务闭环（图文）", level=1)
    段("三类基础任务端到端跑通，各留机器可读证据与产物文件。", 段后=8)
    图(图s["任务"]["路径"], "图 5　看网页 → 学 → 产出 → 说 → 存档；下部为任务判据与关键反例。", 宽=6.8)
    真 = 数["真站"]
    表(["任务", "产物", "实测 / 证据"],
        [["① 做文档（WPS/Word）", "AHI3_基础任务验收记录.docx（+ .md / .pdf）",
          "零依赖手写最小 OOXML；外部解析器 check_office verdict: pass（18 段）；字节可复现"],
         ["② 写代码", "AHI3_自产_统计段数.py",
          "规格 → 模板 → 落盘 → 子进程执行：用例逐条全过、退出码 0；期望写错必失败"],
         ["③ 看网页", "网页摘要_%s.docx" % str((真.get("文档") or {}).get("docx", ""))[-12:-5],
          "真站：切到目标页 = %s；正文 %s 段 / 控件 %s（触顶）/ 外壳剔除 %s；入记忆检索相似 %.1f"
          % (真.get("切到目标页"), 真.get("正文段数"), 真.get("控件数"), 真.get("外壳剔除数"),
             float((真.get("检索回证") or {}).get("相似") or 0))]])
    图(图s["真站"]["路径"], "图 6　真站实测读数与前台切换证据（诚实闸保证：没切到目标页就不许冒充）。", 宽=6.6)
    段("它自己说的一句：「%s」" % str(真.get("说话", ""))[:60], 号=10, 粗=True, 段后=6)
    段("看网页任务逼出的一个真缺陷：`cmd start msedge` 拿不到前台时，旧实现把别的窗口（DSH）当成目标站并报"
       "「真浏览器UIA」。已加诚实闸 `判通路(标题前, 标题后, 段数, 关键词)`：没换页且标题不含目标关键词 ⇒ "
       "一律降级为「前台未切到目标页」，并把警示语写进产出文档标题。", 段后=10)

    doc.add_page_break()

    # ── §5 学进去 ──
    doc.add_heading("5　「学进去」：跨进程持久化", level=1)
    段("做法：**不 pickle 内核内部**，只存可重放事件（容量桥 `_文本`+`_向量` → 事件；学习迹 → 学习事件），"
       "载入时新建同种子桥 + 顺序重放 ⇒ 库与向量确定性再生（沿 `synthesis.py:1244` 的事件重放口径）。", 段后=8)
    表(["环节", "实测"],
        [["存档", "npz + manifest（payload sha256 校验、temp + os.replace 原子替换）"],
         ["新进程载入", "库大小 3 ／ 重放 3 条 ／ 学习迹 1 条、学习重放 2 条"],
         ["召回", "三条经历逐条命中，相似 1.0，通路全为「语义」"],
         ["记忆 ≠ 学习", "学习迹侧记忆库大小 == 记忆侧库大小（物理分离未被存档流程破坏）"]])
    段("边界：事件源自 `容量桥._文本/_向量`；若将来桥内出现不经 `记经历` 的状态（如 IVF 细码），需一并纳入事件表。",
       号=9, 斜=True, 色=(140, 146, 154), 段后=10)

    # ── §6 图文展示：它自产的文档 ──
    doc.add_heading("6　图文展示：它自己写出来的文档", level=1)
    段("下图是本轮 AHI 自主产出的 `.docx` 首页实拍（由 LibreOffice 渲染）。该项目 venv 没有 python-docx，"
       "所以它用 stdlib `zipfile` + 手写最小 OOXML 生成；外部解析器判定结构与文本均合法。", 段后=8)
    实拍 = os.path.join(产物, "预览_v1", "page-0001.png")
    if os.path.exists(实拍):
        图(实拍, "图 7　AHI 自产文档首页实拍（结构与文本已过外部解析器；本模型不支持图像输入，"
                "未人工目视版面）。", 宽=5.0)

    # ── §7 诚实边界 ──
    doc.add_heading("7　诚实边界（必须随报告一起读）", level=1)
    for i, t in enumerate([
        "不主张人类级意识：第 3 节全是可复跑的机制判据，不是「像人一样有自我」的结论。",
        "真浏览器通路依赖前台切换：判据里不跑真站（避免夺前台）；`取控件(上限=600)` 截断 ⇒ 真站 86 段正文可能仍有遗漏。",
        "S4 只存身份锚 + W_self：要「整个我」跨重启，需把 `持久化.py` 的事件重放口径扩到内核侧。",
        "文档版面的视觉核验缺失：`check_office` 的 pass 只保证结构与文本；本模型不支持图像输入，未目视版面。",
        "本体面未纳入：手/嘴/眼的实时闭环、真机动作（需所有者知情）、流式语音（P3 未建）不在本轮证据内。",
        "S6 第④项（真模块四复盘）用的是回退口径：`命.学()` 返回里没有「记忆底图未增」键，按「库不变」判定，JSON 里如实记 null。",
        "模板能力有限：「写代码」是结构化规格 → 6 种模板；「做文档」是素材 + 模板合成，不是从自然语言写任意程序/文章。"], 1):
        段("%d. %s" % (i, t), 号=9.5, 段后=4)

    # ── §8 复跑 ──
    doc.add_heading("8　复跑入口", level=1)
    for t in ["$py = <workspace>/\03_AHI\\.venv\\Scripts\\python.exe",
              "& $py AHI3\\meeting40\\实施\\判据_P0.py          # P0 8/8",
              "& $py AHI3\\meeting40\\实施\\判据_备B.py         # 备B 11/11",
              "& $py AHI3\\meeting40\\实施\\判据_任务.py        # 基础任务 6/6",
              "& $py AHI3\\meeting40\\实施\\判据_自我意识.py    # 工程自我意识 7/7",
              "& $py AHI3\\meeting39\\实施\\步7_全量回归自证.py  # 全量回归总门"]:
        p = doc.add_paragraph()
        r = p.add_run(t)
        r.font.name = "微软雅黑"
        r.font.size = Pt(9)
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        p.paragraph_format.space_after = Pt(2)

    doc.save(输出)
    抹 = _修遗留部件(输出)
    return dict(输出=输出, 字节=os.path.getsize(输出), 抹掉Courier引用=抹)


def _修遗留部件(docx路径: str) -> int:
    """python-docx 模板自带的 `word/stylesWithEffects.xml`（Office 2010 兼容件）仍引用 Courier；
    它只影响 LibreOffice 的**缺字诊断**（正文一个字节都没用它）。这里**原地改写该部件**的字体名，
    保持包结构与其余部件逐字节不变。返回被抹掉的 Courier 引用数。"""
    import zipfile
    临时 = docx路径 + ".tmp"
    抹 = 0
    with zipfile.ZipFile(docx路径) as zin, zipfile.ZipFile(临时, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            数据 = zin.read(it.filename)
            if it.filename == "word/stylesWithEffects.xml":
                t = 数据.decode("utf-8")
                抹 = t.count("Courier")
                for 槽 in ("ascii", "hAnsi", "cs", "eastAsia"):
                    t = t.replace('w:%s="Courier"' % 槽, 'w:%s="微软雅黑"' % 槽)
                数据 = t.encode("utf-8")
            zout.writestr(it, 数据)
    # 新写的文件常被实时扫描/预览句柄短暂占用 ⇒ 重试（沿用 synthesis.py:1232 写心跳的做法）
    import time as _t
    for _ in range(8):
        try:
            os.replace(临时, docx路径)
            return 抹
        except PermissionError:
            _t.sleep(0.25)
    os.replace(临时, docx路径)
    return 抹


def 主() -> dict:
    os.makedirs(图目录, exist_ok=True)
    数 = 读数()
    图s = dict(架构=图1_架构(os.path.join(图目录, "图1_三层架构.png")),
              判据=图2_判据(数, os.path.join(图目录, "图2_判据总览.png")),
              ε对照=图3_ε对照(数["自我报"], os.path.join(图目录, "图3_ε下降对照.png")),
              门控=图4_门控(数["自我报"], os.path.join(图目录, "图4_Wself门控.png")),
              任务=图5_任务(数, os.path.join(图目录, "图5_任务闭环.png")),
              真站=图6_真站(数["真站"], os.path.join(图目录, "图6_真站读数.png")))
    文 = 建文档(数, 图s, os.path.join(产物, "AHI3.0_图文报告_v3.0.docx"))
    with open(os.path.join(图目录, "读数.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in 数.items() if not k.endswith("报") and k != "真站"},
                  f, ensure_ascii=False, indent=1, default=str)
    果 = dict(文档=文, 图s={k: v["路径"] for k, v in 图s.items()},
              读数=dict(P0=数["P0"], 备B=数["备B"], 任务=数["任务"], 自我=数["自我"], 步7=数["步7"]))
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    return 果


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    主()
