# -*- coding: utf-8 -*-
"""meeting38/实施/学习_三阶段.py —— V3 模块四：学习能力强化（消化层重构，纯新增）。

问题（所有者 2026-10-06）：知识面窄、学习能力弱、学 Python 有问题。
根因：现在学习是「存进去」，不是「消化掉」。

V3 三阶段消化：
  阶段一 摄入：看文档/刷网页 → 提取文本向量 → 存入**缓冲舱**（未消化暂存）。
  阶段二 关联：新向量与**记忆底图 M** 做相似度匹配 → 找到「这个新知识和我知道的什么有关」
              → 建立绑定「新知识 ↔ 旧知识」。
  阶段三 复盘：空闲时对当天摄入做**内部递归** → 提取核心概念 → 压缩成**快照** → 存入长期底图。

红线：C1（无 Transformer/无梯度）｜C10（无外挂模型——概念提取用字符 bigram 统计＋HDC 束中心，
     全是既有算子）｜C13（确定性：聚类按插入序、束中心=sign(和)，同输入同输出）。
"""
from __future__ import annotations

from collections import Counter

import numpy as np

# 学习迹独立存储（所有者硬命题「记忆≠学习」，裁定 H5：物理分离）
try:
    from 学习迹 import 学习迹
except Exception:                                        # 兼容被以包方式导入的情形
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from 学习迹 import 学习迹


class 三阶段消化:
    """摄入 → 关联 → 复盘。与引擎的接口只用：编码器（文本→HDC 向量）、记忆桥（M 检索）。

    ★记忆 ≠ 学习（所有者硬命题；`meeting39/01_...自裁备忘.md` H5）：
      · 记忆（M）= `记忆桥`（容量桥/FSW 长时库）——「记着」，本类**只读不写**；
      · 学习 = `self.学迹`（`学习迹` 独立存储）——「变会」，所有学习产物只进这里。
      V3 原先在复盘里 `记忆.记经历(...)` 把学习产物写回记忆底图（审计 P0-2），现已切断。
    """

    def __init__(self, 编码器, 记忆桥=None, 综合体=None, 缓冲上限: int = 512):
        self.编码 = 编码器
        self.记忆 = 记忆桥                      # ★只读（检索用）；不再作为学习产物出口
        self.桥 = 综合体                       # 可选的 meeting31 综合体（提供 _检索，作 M 的补充）
        self.学迹 = 学习迹(背衬=记忆桥)          # ★学习产物独立存储（与记忆物理分离）
        self.缓冲上限 = int(缓冲上限)
        self.缓冲: list = []                   # 阶段一产物：dict(文本,向量,来源,已关联)
        self._已见: set = set()                 # ★去重（导航项在每屏重复，见 摄入()）
        self._近重向量: list = []               # ★近重复去重（UIA 会把同一句报多遍）
        self.绑定: list = []                   # 阶段二产物：(新摘要, 旧摘要, 相似)（镜像 学迹.绑定s）
        self.快照s: list = []                  # 阶段三产物（镜像 学迹.快照s，保留既有字段名）
        self.统计 = dict(摄入=0, 关联=0, 复盘=0, 快照=0)

    # ══════════ 阶段一：摄入 ══════════
    def 摄入(self, 文本: str, 来源: str = "网页") -> dict:
        文 = str(文本 or "").strip()
        if not 文:
            return dict(存=False)
        # ★去重（第二轮实测纠正）：真站点的导航项（"新闻"/"体育"/"登录"…）会在每屏重复出现，
        #   若不去重会把 512 条缓冲塞满导航词，**真实正文句反而进不来**，关联自然只能绑噪声。
        #   确定性：首见即存，重复只计数（不影响任何向量/统计口径的确定性）。
        if 文[:400] in self._已见:
            self.统计["摄入重复"] = self.统计.get("摄入重复", 0) + 1
            return dict(存=False, 重复=True, 缓冲=len(self.缓冲))
        self._已见.add(文[:400])
        向量 = None
        try:
            向量 = self.编码.文本码(文[:512])
        except Exception:
            向量 = None
        # ★近重复去重：UIA 会把同一句在多个节点上重复上报（实测同页同句 cos=1.0000 多次），
        #   精确去重挡不住「同一句带不同空白/标点」的变体。近重复阈值取 0.98（几乎同一句）。
        if 向量 is not None and self._近重向量:
            import numpy as _np
            码本 = _np.stack(self._近重向量, 0)
            分 = (码本 @ 向量) / _np.maximum(_np.linalg.norm(码本, axis=1)
                                          * float(_np.linalg.norm(向量)), 1e-9)
            if float(_np.max(分)) >= 0.98:
                self.统计["摄入近重复"] = self.统计.get("摄入近重复", 0) + 1
                return dict(存=False, 近重复=True, 缓冲=len(self.缓冲))
        if 向量 is not None:
            self._近重向量.append(向量)
            if len(self._近重向量) > 4000:
                self._近重向量 = self._近重向量[-4000:]
        项 = dict(文本=文[:400], 向量=向量, 来源=来源, 已关联=False)
        self.缓冲.append(项)
        if len(self.缓冲) > self.缓冲上限:
            self.缓冲 = self.缓冲[-self.缓冲上限:]
        self.统计["摄入"] += 1
        return dict(存=True, 缓冲=len(self.缓冲), 有向量=(向量 is not None))

    # ══════════ 阶段二：关联 ══════════
    def _查M(self, 向量, 文本: str = "", k: int = 3) -> list:
        """向记忆底图 M 检索相关旧知识：[{文本,相似}]。优先真记忆桥，退化用综合体 _检索。

        ★`向量` 走**语义通路**（`容量桥.查经历` 自动判定；见容量桥文件头「语义检索真通路」）——
        这是「理解语义」的关键：原先只喂 FSW 地址通路，相似度落在噪声底（0.03 量级，
        无关文本反而更高），关联因此是随机的。
        """
        出 = []
        if self.记忆 is not None and 向量 is not None:
            try:
                r = self.记忆.查经历(向量)
                if isinstance(r, dict) and r.get("文本"):
                    出.append(dict(文本=str(r["文本"])[:80], 相似=float(r.get("相似") or 0.0),
                                   通路=r.get("通路")))
            except Exception:
                pass
        if not 出 and self.桥 is not None and 文本:
            try:
                for 段, 分 in self.桥._检索(文本, k=k):
                    出.append(dict(文本=str(段)[:80], 相似=float(分), 通路="综合体检索"))
            except Exception:
                pass
        return 出

    def 关联(self, k: int = 3, 相似下限: float = 0.2) -> list:
        """对未关联的摄入项逐个与 M 匹配，建立「新↔旧」绑定。

        ★阈值（审计 P1-3）：原先「命中即绑定、不设下限」，实测会把语义无关的旧知
        （相似 −0.01/0.02）也绑上，等于「见谁绑谁」。现设**相似度下限**（默认 0.2），
        低于下限的命中**不产生绑定** —— 判据因此可失败。
        """
        新绑定 = []
        for 项 in self.缓冲:
            if 项["已关联"]:
                continue
            项["已关联"] = True
            命中 = self._查M(项["向量"], 项["文本"], k=k)
            if not 命中:
                continue
            for h in 命中:
                s = float(h.get("相似") or 0.0)
                if s < float(相似下限):          # ★无关不绑（可失败判据）
                    continue
                项绑 = self.学迹.放绑定(项["文本"], h["文本"], s)
                新绑定.append(dict(项绑, 相似=项绑["相似"]))
        self.绑定.extend([dict(新=b["新"], 旧=b["旧"], 相似=b["相似"]) for b in 新绑定])
        self.绑定 = self.绑定[-2000:]
        self.统计["关联"] += len(新绑定)
        return 新绑定

    # ══════════ 阶段三：复盘（内部递归 → 概念快照）══════════
    @staticmethod
    def _关键词s(文本s, top: int = 6) -> list:
        """统计式「核心概念」（非模型）：拉丁标识符整词 + 中文 2-gram，按频次/长度取前 top。
        排序确定性：频次降序 → 词长降序 → 字典序。"""
        import re
        计 = Counter()
        for 文 in 文本s:
            s = str(文)
            for m in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", s):
                计[m.lower()] += 1
            for run in re.findall(r"[\u4e00-\u9fff]{2,}", s):
                for i in range(len(run) - 1):
                    计[run[i:i + 2]] += 1
        return [w for w, _ in sorted(计.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[:top]]

    def 复盘(self, k簇相似: float = 0.3, top: int = 6) -> dict:
        """对缓冲里的摄入做内部递归：提取核心概念 + 向量单遍聚类取代表（medoid），
        压缩成一个概念快照（束）→ 入长期底图。"""
        项s = [x for x in self.缓冲 if x["向量"] is not None]
        if not 项s:
            return dict(复盘=False, 原因="缓冲无可用向量")
        文本s = [x["文本"] for x in 项s]
        关键词 = self._关键词s(文本s, top=top)
        # 单遍余弦聚类（插入序，确定性）→ 每簇 medoid
        簇s = []                                   # [{成员idx, 中心束}]
        for i, x in enumerate(项s):
            if 簇s:
                中心 = np.stack([c["中心"] for c in 簇s], 0)
                # 批相似
                try:
                    sims = np.asarray([float(np.dot(x["向量"], c) / (np.linalg.norm(x["向量"]) *
                                     np.linalg.norm(c) + 1e-9)) for c in 中心])
                except Exception:
                    sims = np.zeros(len(簇s), np.float32)
                j = int(np.argmax(sims))
                s = float(sims[j])
            else:
                j, s = -1, 0.0
            if j >= 0 and s >= k簇相似:
                簇s[j]["成员"].append(i)
            else:
                簇s.append(dict(成员=[i], 中心=x["向量"]))
        medoids = []
        for c in 簇s:
            本 = 项s[c["成员"][0]]
            最 = -1.0
            for i in c["成员"]:
                x = 项s[i]
                try:
                    分 = float(np.dot(x["向量"], c["中心"]) / (np.linalg.norm(x["向量"]) *
                          np.linalg.norm(c["中心"]) + 1e-9))
                except Exception:
                    分 = 0.0
                if 分 > 最:
                    最, 本 = 分, x
            medoids.append(本)
        # 概念快照 = 关键词码 + 各簇代表向量 的束
        码s = []
        for w in 关键词:
            try:
                码s.append(self.编码.文本码(w))
            except Exception:
                pass
        for m in medoids:
            码s.append(m["向量"])
        try:
            束 = self.编码.束(码s) if 码s else None
        except Exception:
            束 = None
        摘要 = "概念快照：" + "、".join(关键词[:top])
        # ★学习产物只进独立学习迹，**不写记忆底图**（H5 物理分离；审计 P0-2）
        库0 = self.学迹.记忆库大小()
        号 = self.学迹.放快照(关键词, len(簇s), 摘要, 束=束,
                              代表s=[m["文本"][:40] for m in medoids])
        库1 = self.学迹.记忆库大小()
        快 = dict(号=号, 关键词=关键词, 簇数=len(簇s),
                  代表s=[m["文本"][:40] for m in medoids],
                  摘要=摘要, 束=束, 入库=None, 学习迹=号,
                  记忆库前=库0, 记忆库后=库1)
        self.快照s.append(快)
        self.统计["复盘"] += 1
        self.统计["快照"] += 1
        return dict(复盘=True, 关键词=关键词, 簇数=len(簇s),
                    摘要=摘要, 入库=None, 学习迹=号,
                    记忆底图未增=bool(库0 == 库1),
                    有束=(束 is not None),
                    代表s=[m["文本"][:40] for m in medoids])

    # ══════════ 一环消化（摄入→关联→复盘）══════════
    def 消化一环(self, 文本s, 来源: str = "网页", 最少字: int = 8) -> dict:
        """摄入 → 关联 → 复盘。

        `最少字`：短于该长度的片段不入缓冲（导航项/标签多为 2~6 字，如「新闻」「体育」；
        真实正文句通常 ≥8 字）。这是**结构规则**，不是语义阈值——用来把噪声挡在门外，
        让「关联」有可比的内容粒度（第二轮真站实测：不去短句时缓冲被导航词塞满）。
        """
        存 = 0
        for t in 文本s:
            s = str(t or "").strip()
            if len(s) < int(最少字):
                continue
            if self.摄入(s, 来源=来源).get("存"):
                存 += 1
        绑 = self.关联()
        复 = self.复盘()
        return dict(摄入=存, 投喂=len(文本s), 缓冲=len(self.缓冲), 新绑定=len(绑),
                    绑定样本=绑[:4], 复盘=复)

    def 快照(self) -> dict:
        return dict(统计=dict(self.统计), 缓冲=len(self.缓冲),
                    绑定=len(self.绑定), 快照=len(self.快照s),
                    学习迹=self.学迹.摘要(),
                    库大小=(self.记忆.库大小() if self.记忆 is not None else None))