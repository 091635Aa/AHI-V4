# -*- coding: utf-8 -*-
"""meeting38/语义.py —— AHI 语义层：真 affordance 图（非 Transformer）

本层只用 HDC 双极码与余弦，无 Transformer/无梯度；语义来自亲历点击而非名字长相。

语义的真值来源是"世界"而非"我们"：
  · 一个按钮"是什么"，由它**自己亲历的点击结果**（点击前→后的屏幕差分）决定，
    不是它的名字、不是它的长相、不是它的坐标。
  · 容器层级（窗口>面板>按钮）来自几何"最小包含者"，是纯粹的结构事实。
  · 叠加时"谁接住点击"，不预设（不写死"后画的在上"），起初如实返回"未知"，
    只有真实点击裁决不断累积，才收敛出确定的上/下关系。
  · 功能词的划分：把各按钮的"功能签名"（HDC 双极码）做余弦聚类（≥阈值归并），
    同名异效自然分开、异名同效自然归并——名字与长相都不进入签名判断。

数据流：建容器树 / 叠加优先级 每拍从 态.取视觉() 涌现；记点击 是学习入口，
真实屏幕点击后由外部把头真值（真实命中句柄）喂进来批改支配计数与功能聚类。
"""
from __future__ import annotations

import threading
import time

import numpy as np


# ══════════════════════ 常量（结构常量，非先验自由度） ══════════════════════
聚类阈值 = 0.55          # 两个功能签名余弦 ≥ 0.55 → 同一"功能词"
最大层深 = 8             # 容器树深度保护（防环）


def _面积(框) -> float:
    if not 框:
        return 0.0
    _, _, w, h = 框
    return float(max(w, 0)) * float(max(h, 0))


def _框含点(框, pt) -> bool:
    if not 框 or pt is None:
        return False
    x, y, w, h = 框
    return (x <= pt[0] <= x + w) and (y <= pt[1] <= y + h)


def _框交(框a, 框b) -> float:
    if not 框a or not 框b:
        return 0.0
    ax, ay, aw, ah = 框a
    bx, by, bw, bh = 框b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


class 语义图:
    """真 affordance 图：容器树（几何最小包含者）+ 功能签名聚类（HDC 余弦）
    + 叠加支配（由真实点击裁决批改）。全部只用双极超向量与计数，无任何网络/梯度。"""

    def __init__(self, 态, 台账=None):
        self.态 = 态
        self.台账 = 台账                      # 外部持久台账（可选 dict），用于落盘学习结果

        self.按钮s: dict = {}                  # 句柄 -> {功能签名, 功能词, 功能置信, 履历, 点击次数}
        self.词表: list = []                   # [{词, 码, 成员, 大小}]
        self.支配: dict = {}                   # (上,下) -> int；>0 表示前者支配后者

        self._裁决数 = 0
        self._容器树: list = []
        self._树索引: dict = {}
        self._最大深度 = 0

        # 若给了台账，导入既有学习（按钮s / 支配 / 裁决数）
        if isinstance(台账, dict):
            for h, r in (台账.get("按钮s") or {}).items():
                if isinstance(r, dict):
                    self.按钮s[h] = r
            self.支配 = dict(台账.get("支配") or {})
            self._裁决数 = int(台账.get("点击裁决") or 0)
            if self.按钮s:
                self._重聚类()

    # ══════════════════════ 容器树：最小包含者负责 ══════════════════════
    @staticmethod
    def _名(e: dict) -> str:
        if not e:
            return "元素"
        字 = str(e.get("字", "") or "").strip()
        if 字:
            return 字
        形状 = str(e.get("形状", "") or "").strip()
        return 形状 or "元素"

    def 建容器树(self, 元素s) -> list:
        """父 = 面积严格更大、且包含本元素中心点的最小元素（最小包含者负责）。
        返回并写回 态.写语义(容器树=...)；每项 {句柄, 父, 层级路径}。最大层深保护防环。"""
        元素s = [e for e in (元素s or []) if isinstance(e, dict)]
        索引 = {}
        for e in 元素s:
            h = e.get("句柄")
            if h is not None:
                索引[h] = e

        父表: dict = {}
        for e in 元素s:
            h = e.get("句柄")
            if h is None:
                continue
            ea = _面积(e.get("框"))
            候选 = []
            for p in 元素s:
                ph = p.get("句柄")
                if ph is None or ph == h:
                    continue
                pa = _面积(p.get("框"))
                if pa <= ea:                      # 严格更大：从根上杜绝环
                    continue
                if _框含点(p.get("框"), e.get("心")):
                    候选.append((pa, ph))
            候选.sort(key=lambda t: t[0])         # 面积最小者
            父表[h] = 候选[0][1] if 候选 else None

        树 = []
        最大深度 = 0
        for e in 元素s:
            h = e.get("句柄")
            if h is None:
                continue
            路径 = []
            已访 = set()
            cur = h
            depth = 0
            while (cur is not None) and (cur not in 已访) and (depth < 最大层深):
                已访.add(cur)
                路径.append(self._名(索引.get(cur)))
                cur = 父表.get(cur)
                depth += 1
            路径.reverse()
            最大深度 = max(最大深度, len(路径))
            树.append(dict(句柄=h, 父=父表.get(h), 层级路径=" > ".join(路径)))

        self._容器树 = 树
        self._树索引 = {it["句柄"]: it for it in 树}
        self._最大深度 = 最大深度
        self.态.写语义(容器树=树)
        return 树

    # ══════════════════════ 叠加：谁接住点击，由裁决数据说话 ══════════════════════
    def 取上层(self, a, b):
        """A、B 叠着时谁在上。完全由真实点击裁决累积出的支配计数决定：
        net>0 判 a、net<0 判 b、net==0 如实返回"未知"（绝不瞎猜硬规则）。"""
        if a is None or b is None or a == b:
            return "未知"
        net = int(self.支配.get((a, b), 0)) - int(self.支配.get((b, a), 0))
        if net > 0:
            return a
        if net < 0:
            return b
        return "未知"

    def 叠加优先级(self, 元素s) -> dict:
        """列出所有真实重叠的元素对及其学到的上/下关系（未学到即为"未知"）。
        写回 态.写语义(叠加=...)；返回可序列化摘要 + 取上层 判决函数。"""
        元素s = [e for e in (元素s or []) if isinstance(e, dict) and e.get("句柄") is not None]
        列表 = []
        for i in range(len(元素s)):
            for j in range(i + 1, len(元素s)):
                a, b = 元素s[i], 元素s[j]
                ha, hb = a.get("句柄"), b.get("句柄")
                if ha == hb:
                    continue
                交 = _框交(a.get("框"), b.get("框"))
                if 交 <= 0:
                    continue
                上 = self.取上层(ha, hb)
                证据 = int(self.支配.get((ha, hb), 0)) - int(self.支配.get((hb, ha), 0))
                列表.append(dict(对=(ha, hb), 上=上, 证据=证据, 重叠=round(交, 1)))
        # 已确定的排前，其次看证据强度/重叠面积
        列表.sort(key=lambda r: (r["上"] == "未知", -abs(r["证据"]), -r["重叠"]))
        已定 = sum(1 for r in 列表 if r["上"] != "未知")
        摘要 = dict(列表=列表, 已定=已定, 未知=len(列表) - 已定)
        self.态.写语义(叠加=摘要)
        return dict(摘要, 取上层=self.取上层)

    def _批支配(self, 上, 下, 胜者):
        """(上,下) 为一对被评估的方向；实际点中的是 上 → +1，是 下 → -1。"""
        k = (上, 下)
        self.支配[k] = int(self.支配.get(k, 0)) + (1 if 胜者 == 上 else -1)
        self._裁决数 += 1

    def _批改叠加(self, 句柄, 点击前元素s, 真实命中句柄) -> int:
        """用真实命中者批改支配计数：句柄是其所在重叠栈里的"待评上候选"。
        以句柄中心为实际点击点，找出该点上所有被覆盖的其它元素作为"下候选"。"""
        前 = {}
        for e in (点击前元素s or []):
            if isinstance(e, dict) and e.get("句柄") is not None:
                前[e["句柄"]] = e
        e0 = 前.get(句柄)
        pt = e0.get("心") if e0 else None

        候选 = []
        if pt is not None:
            for h, e in 前.items():
                if h == 句柄:
                    continue
                if _框含点(e.get("框"), pt):
                    候选.append(h)

        批改 = 0
        for o in 候选:
            if 真实命中句柄 == 句柄 or 真实命中句柄 == o:
                self._批支配(句柄, o, 真实命中句柄)
                批改 += 1
        # 几何不可用（或命中者不在候选内）时退化为直接评估这一对
        if 批改 == 0 and 真实命中句柄 is not None and 真实命中句柄 != 句柄:
            self._批支配(句柄, 真实命中句柄, 真实命中句柄)
            批改 += 1
        return 批改

    # ══════════════════════ 学习入口：事后归因 ══════════════════════
    def _元素码(self, e: dict) -> np.ndarray:
        """一个元素的 HDC 码：几何/颜色/形状 的特征码 与 文字的文本码 **按位绑定**。
        用绑定而非捆绑：绑定是两个码逐位相乘（含负号，保持零均值），
        不同几何/不同文字自然去相关；同一元素则逐位一致——这样"异效"分离、"同效"重合。"""
        x, y, w, h = e.get("框", (0, 0, 0, 0))
        主色 = e.get("主色", (0, 0, 0))
        if isinstance(主色, (int, float)):
            主色 = (主色, 0, 0)
        主色 = tuple(主色)[:3]
        while len(主色) < 3:
            主色 = tuple(主色) + (0,)
        形状 = str(e.get("形状", "") or "")
        形状id = (sum(ord(c) for c in 形状) % 97) / 97.0
        特征 = [float(x), float(y), float(w), float(h),
                float(主色[0]), float(主色[1]), float(主色[2]), 形状id]
        特征码 = self.态.编码.特征码(特征)
        字 = str(e.get("字", "") or "")
        if not 字:
            return 特征码
        文本码 = self.态.编码.文本码(字)
        return self.态.编码.绑(特征码, 文本码, 移位=3)

    def 记点击(self, 句柄, 点击前元素s, 点击后元素s, 真实命中句柄=None) -> dict:
        """学习入口。功能签名 = (新增⊕消失⊕文字变化) 的 HDC 束；
        功能词 = 全体功能签名的余弦聚类；真实命中句柄 用于批改叠加支配。"""
        前 = {}
        for e in (点击前元素s or []):
            if isinstance(e, dict) and e.get("句柄") is not None:
                前[e["句柄"]] = e
        后 = {}
        for e in (点击后元素s or []):
            if isinstance(e, dict) and e.get("句柄") is not None:
                后[e["句柄"]] = e

        新增 = [后[h] for h in 后 if h not in 前]
        消失 = [前[h] for h in 前 if h not in 后]
        变化 = [(前[h], 后[h]) for h in 前
                if h in 后 and str(前[h].get("字", "")) != str(后[h].get("字", ""))]

        码s = []
        for e in 新增:
            码s.append(self._元素码(e))
        for e in 消失:
            码s.append(self._元素码(e))
        for eb, ea in 变化:
            码s.append(self.态.编码.绑(self._元素码(eb), self._元素码(ea), 移位=17))
        签名 = self.态.编码.束(码s) if 码s else self.态.编码.文本码("无变化")

        记录 = self.按钮s.get(句柄)
        if not isinstance(记录, dict):
            记录 = dict(句柄=句柄, 履历=[], 点击次数=0)
        记录["功能签名"] = 签名
        记录["点击次数"] = int(记录.get("点击次数", 0)) + 1
        履历 = 记录.setdefault("履历", [])
        履历.append(dict(t=round(self.态.秒(), 2), 新增=len(新增), 消失=len(消失),
                         变化=len(变化), 真实命中=真实命中句柄))
        if len(履历) > 64:
            del 履历[:-64]
        self.按钮s[句柄] = 记录

        批改 = 0
        if 真实命中句柄 is not None:
            批改 = self._批改叠加(句柄, 点击前元素s, 真实命中句柄)

        self._重聚类()
        self.态.写语义(按钮s=dict(self.按钮s), 词表=list(self.词表),
                      点击裁决=int(self._裁决数))
        if isinstance(self.台账, dict):
            self.台账.update(按钮s=dict(self.按钮s), 支配=dict(self.支配),
                             点击裁决=int(self._裁决数))

        return dict(句柄=句柄, 功能签名=签名, 功能词=记录.get("功能词"),
                    新增=len(新增), 消失=len(消失), 变化=len(变化),
                    批改次数=批改, 点击次数=记录["点击次数"])

    # ══════════════════════ 功能聚类（纯 HDC 余弦 + 计数） ══════════════════════
    def _重聚类(self):
        """按插入序对全部功能签名做单遍余弦聚类；功能词本身也是 bundle 出的 HDC 码。
        名称/长相不参与，只有"亲历点击得到的功能签名"参与。"""
        项 = [(h, r.get("功能签名")) for h, r in self.按钮s.items()
              if isinstance(r, dict) and r.get("功能签名") is not None]
        项.sort(key=lambda t: str(t[0]))
        项序 = {h: i for i, (h, _) in enumerate(项)}

        簇s = []                                  # [{成员, 码}]
        for h, sig in 项:
            if 簇s:
                中心 = np.stack([c["码"] for c in 簇s], 0)
                sims = self.态.编码.批相似(sig, 中心)
                k = int(np.argmax(sims))
                s = float(sims[k])
            else:
                k, s = -1, 0.0
            if k >= 0 and s >= 聚类阈值:
                簇s[k]["成员"].append(h)
                簇s[k]["码"] = self.态.编码.束(
                    [self.按钮s[m]["功能签名"] for m in 簇s[k]["成员"]])
            else:
                簇s.append(dict(成员=[h], 码=sig))

        # 以成员中最小插入序为词序，保证标签稳定
        簇s.sort(key=lambda c: min(项序[m] for m in c["成员"]))
        self.词表 = []
        for i, c in enumerate(簇s, 1):
            标签 = "W%d" % i
            中心 = c["码"]
            for m in c["成员"]:
                sim = self.态.编码.相似(self.按钮s[m]["功能签名"], 中心)
                self.按钮s[m]["功能词"] = 标签
                self.按钮s[m]["功能置信"] = round(float(sim), 3)
            self.词表.append(dict(词=标签, 码=中心,
                                  成员=list(c["成员"]), 大小=len(c["成员"])))

    # ══════════════════════ 读取 ══════════════════════
    def 取按钮(self, 句柄) -> dict | None:
        r = self.按钮s.get(句柄)
        if not isinstance(r, dict):
            return None
        it = self._树索引.get(句柄)
        return dict(功能签名=r.get("功能签名"), 功能词=r.get("功能词"),
                    层级路径=(it.get("层级路径") if it else ""),
                    置信=float(r.get("功能置信", 0.0)),
                    点击次数=int(r.get("点击次数", 0)))

    def 判据(self) -> dict:
        return dict(按钮数=len(self.按钮s), 词数=len(self.词表),
                    容器根数=sum(1 for it in self._容器树 if it.get("父") is None),
                    最大深度=int(self._最大深度), 点击裁决数=int(self._裁决数))

    # ══════════════════════ 连续循环（约 2–4Hz，不阻塞） ══════════════════════
    def 运行(self, to_stop: threading.Event):
        """每拍读 态.取视觉() → 建容器树 → 叠加优先级 → 写回语义槽。只读最新值，不阻塞他路。"""
        周期 = 0.3
        while not to_stop.is_set():
            try:
                v = self.态.取视觉()
                元素s = v.get("元素s") or []
                if 元素s:
                    self.建容器树(元素s)
                    self.叠加优先级(元素s)
            except Exception:
                pass
            to_stop.wait(周期)


# ══════════════════════ 离线自检（不碰真屏幕） ══════════════════════
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from 共存态 import 共存态

    def _元素(句柄, 字, 框, 主色=(120, 120, 120), 形状="按钮"):
        x, y, w, h = 框
        return dict(句柄=句柄, 框=(x, y, w, h), 心=(x + w / 2.0, y + h / 2.0),
                    字=字, 主色=tuple(主色), 形状=形状, 置信=0.9)

    态 = 共存态(种子=3800, 用gpu=False)
    图 = 语义图(态)
    标题 = "════════ 语义.py 离线自检（HDC 双极码 · 无 Transformer/无梯度）════════"
    print(标题)

    # ── ① 最小包含层级 ──
    窗口 = _元素("w", "窗口", (0, 0, 1000, 700), 形状="窗口")
    面板 = _元素("p", "面板", (100, 100, 400, 300), 形状="面板")
    按钮 = _元素("b", "按钮", (150, 150, 80, 30))
    裸钮 = _元素("b2", "确定", (700, 600, 90, 30))
    树 = 图.建容器树([窗口, 面板, 按钮, 裸钮])
    mp = {it["句柄"]: it for it in 树}
    层级正确 = (mp["b"]["父"] == "p" and mp["p"]["父"] == "w"
              and mp["w"]["父"] is None
              and mp["b"]["层级路径"] == "窗口 > 面板 > 按钮"
              and mp["b2"]["父"] == "w"
              and mp["b2"]["层级路径"] == "窗口 > 确定")
    print("[1] 容器树:")
    for it in 树:
        print("      %-3s 父=%-5s 路径=%s" % (it["句柄"], it["父"], it["层级路径"]))
    print("    最小包含层级正确 =", 层级正确,
          "| 根数=", sum(1 for it in 树 if it["父"] is None),
          "| 最大深度=", 图._最大深度)

    # ── ② 同名异效分开 / 异名同效归并 ──
    b_s1 = _元素("s1", "保存", (520, 150, 80, 30))   # 异名
    b_s2 = _元素("s2", "存盘", (520, 200, 80, 30))   # 异名
    b_n1 = _元素("n1", "新建", (520, 250, 80, 30))   # 同名
    b_n2 = _元素("n2", "新建", (520, 300, 80, 30))   # 同名
    基础 = [窗口, 面板, b_s1, b_s2, b_n1, b_n2]
    已保存 = _元素("r_save", "已保存", (300, 300, 100, 24), (0, 180, 0), "标签")
    文档 = _元素("r_doc", "文档", (300, 340, 100, 24), (30, 90, 220), "标签")
    图表 = _元素("r_chart", "图表", (700, 40, 100, 24), (220, 120, 30), "标签")

    # "保存"与"存盘"：效果完全相同（都让"已保存"出现）→ 异名同效应归并
    图.记点击("s1", 基础, 基础 + [已保存])
    图.记点击("s2", 基础, 基础 + [已保存])
    # 两个都叫"新建"：效果不同（一个出现"文档"、一个出现"图表"）→ 同名异效应分开
    图.记点击("n1", 基础, 基础 + [文档])
    图.记点击("n2", 基础, 基础 + [图表])

    词表快照 = list(态.取语义().get("词表") or [])
    r_s1, r_s2 = 图.取按钮("s1"), 图.取按钮("s2")
    r_n1, r_n2 = 图.取按钮("n1"), 图.取按钮("n2")
    异名同效归并 = r_s1["功能词"] == r_s2["功能词"]
    同名异效分开 = r_n1["功能词"] != r_n2["功能词"]
    同效余弦 = 图.态.编码.相似(图.按钮s["s1"]["功能签名"], 图.按钮s["s2"]["功能签名"])
    异效余弦 = 图.态.编码.相似(图.按钮s["n1"]["功能签名"], 图.按钮s["n2"]["功能签名"])
    print("[2] 功能聚类（词表 %d 个）:" % len(词表快照))
    for it in 词表快照:
        print("      %s 大小=%d 成员=%s" % (it["词"], it["大小"], it["成员"]))
    print("    '保存'(s1)=%s  '存盘'(s2)=%s  → 异名同效归并=%s (余弦=%.3f)"
          % (r_s1["功能词"], r_s2["功能词"], 异名同效归并, 同效余弦))
    print("    '新建'(n1)=%s  '新建'(n2)=%s  → 同名异效分开=%s (余弦=%.3f)"
          % (r_n1["功能词"], r_n2["功能词"], 同名异效分开, 异效余弦))

    # ── ③ 叠加：由真实点击裁决收敛（初始必须"未知"） ──
    甲 = _元素("A", "顶层", (100, 100, 200, 60))
    乙 = _元素("B", "底层", (120, 120, 200, 60))
    前景 = [窗口, 甲, 乙]
    初 = 图.叠加优先级(前景)
    初始判决 = 初["取上层"]("A", "B")               # 冻结初始读数（此时裁决数=0）
    初始未知 = (初始判决 == "未知" and 初["未知"] >= 1)
    裁决前 = 图.判据()["点击裁决数"]
    for _ in range(5):
        图.记点击("B", 前景, 前景, 真实命中句柄="A")   # 手瞄B，实际被A接住
    后 = 图.叠加优先级(前景)
    收敛 = (后["取上层"]("A", "B") == "A" and 后["取上层"]("B", "A") == "A")
    裁决后 = 图.判据()["点击裁决数"]
    print("[3] 叠加裁决:")
    print("    初始取上层(A,B)=%s（未知=%s）" % (初始判决, 初始未知))
    print("    5次真实命中后 取上层(A,B)=%s 取上层(B,A)=%s → 收敛=%s"
          % (后["取上层"]("A", "B"), 后["取上层"]("B", "A"), 收敛))
    print("    支配计数[(B,A)]=%s  裁决数 %d→%d"
          % (图.支配.get(("B", "A")), 裁决前, 裁决后))

    print("[4] 判据:", 图.判据())
    全过 = 层级正确 and 异名同效归并 and 同名异效分开 and 初始未知 and 收敛
    print("════════ 自检结果: %s ════════" % ("全部通过" if 全过 else "存在失败"))
    sys.exit(0 if 全过 else 1)
