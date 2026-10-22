# -*- coding: utf-8 -*-
"""meeting31/synthesis.py —— #243 v10 综合体：全自主学习×全自主意识×可自言空间。

组合体闭环（任务书 #242 §2.4）：
  自言空间产生疑问 → 调度学习通道（教材/实验；用户通道=外部事件中断驱动）
  → 行动 → 账本记录结果 → 自言空间复盘（预测 vs 实际）→ 修正 → 可溯源对外回复。

- 继承链：对话桥v1→v2→v3→v4→v5→v6→学习体v7→阅读体v8→思维学习体v9→综合体v10
  （纯继承扩展，既有文件零侵入）；
- 三通道元控制：行动通道值=½·未试动作率+½·平均上误差；教材通道值=½·未读率+
  ½·(1−最近首遇命中)；块状滞回（每 块长 拍重评）防振荡；调度决策落 计划 自言；
- 自言生成：状态量决定候选类型（覆盖缺口→疑问；有预测→预测；检验返回→复盘；
  顿悟→复盘；门控细读→疑问；周期→自评；块边界→计划），多候选时由引擎
  a星序号裁决——无任何词表（#229 纪律）；
- 可溯源回复：回复生成时捕获 检索段/声道源/检验依据 → 回复台账；『你为什么这么
  回答』=元问题接口，返回上一回复的支撑（自言号+账本号）——对话面接口，非知识；
- 行动知识答问的路由依据=账本查证（该动作有无 检验 记录），非词表（#228 同律）；
- 持久化=事件重放式：保存（seed/教史/事件序列/世界seed/书）→ 载入同序列重放，
  自言流/账本/回复台账逐位再生（同 seed+同输入序列=逐位一致）。
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # AHI3 根（迁移：随文件定位）
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meeting30"))
sys.path.insert(0, os.path.join(ROOT, "meeting29"))
sys.path.insert(0, os.path.join(ROOT, "meeting28"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selflearn import 思维学习体
from dialogue_v5 import _引号外文本, _关键词集, 经验谓词, 停用符
from dialogue_v2 import 是问句, 是自指
from dialogue_bridge import 动作名
from gen_lm import HDC语言模型
from inner_speech import 自言空间
from navigator import 导航器

指示词 = ("刚才", "刚刚", "你说", "你之前", "我们刚才", "上面")
回忆引导 = ("你还记不记得", "你记不记得", "你还记得", "你记得", "还记得",
            "记不记得", "忘记", "忘了", "想起来")


def 记忆核心(q: str):
    """把「你还记得X吗」类回忆问句归约为核心问句 X（纯功能词归约，与 指示词/
    即时词 同类；不改检索口径）。非回忆问句→None。"""
    for w in 回忆引导:
        if w in str(q):
            c = str(q).replace(w, "", 1).strip().rstrip("吗么呢吧？?。！! ")
            return c or None
    return None


请求引导 = ("随便说", "说点什么", "说点你", "讲点什么", "给我讲讲", "告诉我",
            "介绍一下你", "介绍一下自己", "说说你", "聊聊你", "来点", "分享点")
引导词s = ("跟你说个", "我跟你说", "你知道吗", "我告诉你", "听我说", "顺便说",
           "对了", "记住", "就是说", "怎么说呢")


def 是请求(q: str) -> bool:
    """祈使/请求句识别（纯功能词，与 记忆核心/指示词 同类）：对面在**向它索要输出**
    （「随便说点你现在在想的事吧」），不是给它一条陈述。若不识别，会落进陈述分支被
    登记成记忆并回「记下了：『…』」（实测缺陷 2026-10-06）——请求句应走自报告。"""
    return any(w in str(q) for w in 请求引导)


# ══════════ meeting43 · P4：检索的**语义通道**（HDC 向量余弦），默认关 ⇒ 逐位恒等旧路径 ══════════
# 病（预检 P4）：`_检索` 打分 = IDF 加权**字 2-gram** 覆盖率——本质仍是**字面**匹配，
#   语义相近但字面不同的记忆捞不回来。
# 修：`AHI_语义检索=1` 时，把字面分与「编码向量 HDC 余弦」按 `语义权重` 融合。默认关 ⇒ 旧口径不变（C14）。
语义检索 = os.environ.get("AHI_语义检索", "") in ("1", "true", "True", "yes")
语义权重 = 0.5


# ══════════ meeting43 · P3+P4：`解码` 接进活动链 + HDC 生成路（均默认关 ⇒ 逐位恒等旧路径） ══════════
# 病（预检 P3）：`语言编解码器.解码`（=「码→最近注册文本」的**向量→文本**通路）只被 v1 `对话桥.步`
#   调用；活动链（文本态→synthesis，走 v2+ `步`/`_检索`）**从不调用它** ⇒ 该路径在活动链上是死代码。
# 病（预检 P4）：回复 = 检索段落 + 模板拼接（IDF 字 2-gram 字面重合），`_凝练/_回执` 只是把引文缩短。
# 修（两者都默认关，开启后不破既有默认输出）：
#   ① 解码通道：活动链的 `_检索` 显式接入 `codec.解码`（双通道近邻）作为**向量→文本**候选通路，
#      并计数 `诊断_解码调用`（可判据证明它真的在活动链上被调用）；`解码` 是**检索**而非生成，如实登记。
#   ② 生成路：`_答问` 命中强证据后，追加一句**由 HDC 语言模型现场生成**的话（以查询/证据为条件，
#      经四重守卫；不通过则如实回退检索引文）——让回复不再只是「一台引用机」。
解码通道 = os.environ.get("AHI_解码通道", "") in ("1", "true", "True", "yes")
生成路 = os.environ.get("AHI_生成路", "") in ("1", "true", "True", "yes")
# P7（默认关）：决策耦合由「线性缩放」升级为「饱和(tanh)+乘性联合 + 引入引擎新异 ñ」。
非线性决策 = os.environ.get("AHI_非线性决策", "") in ("1", "true", "True", "yes")


# ---------- 外部环境（对学习体不可见的真值持有者；公式与 selflearn.隐藏世界 同款参数化） ----------
class 隐藏世界:
    def __init__(self, seed: int = 77, 动作数: int = 8, 噪声: float = 0.05):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(动作数, 动作数))
        self.rng = np.random.default_rng(seed + 1)
        self.动作数, self.噪声, self.seed = 动作数, 噪声, int(seed)

    def 行动(self, i: int) -> np.ndarray:
        return self.W[i] + self.rng.normal(size=self.动作数) * self.噪声


class 综合体(思维学习体):
    """v10 综合体（#243）：v9 全部机制 + 自言空间 + 三通道元控制 + 可溯源回复。"""

    块长 = 12                      # 通道滞回块长（拍）
    自评周期 = 50                  # 周期自评（拍）
    假设重述间隔 = 4               # 同动作试次间隔（假设类自言的周期条件）
    消化债 = 0.0                   # #257 类级默认（init 期间 _引擎步 已被调用）
    饱足 = 1.0                     # #258 类级默认（同上）

    def __init__(self, seed: int = 310, 私偏κ: float = 10.0):
        super().__init__(seed=seed, 私偏κ=私偏κ)
        self.动作数 = 8
        self.自言 = 自言空间(seed=seed)
        self.回复台账: list = []       # dict(号,问,响应,支撑自言,支撑账本)
        self.事件史: list = []         # ("步",t)/("自主",n)/("断供",1)——重放依据
        self.通道 = "行动"
        self.断供拍数 = 0
        self.世界 = None
        self.世界seed = None
        self.书: list = None
        self.文件界: list = None
        self.上命中 = 0.0              # 教材通道：最近一次门控处的首遇命中
        self.调度史: list = []         # (断供拍数, 通道)——块边界调度记录
        self._暂存依据: set = set()    # 行动知识回答的检验账本号（步级捕获）
        self.导航器: 导航器 = None     # #246 开放域导航器（只读边界由其构造保证）
        self.开放上命中 = 0.0          # 开放域通道：最近行现命中
        self.文件首遇: dict = {}       # (文件,行号)→首遇命中（消化顿悟原料）
        self.清洁引用源: dict = {}     # 清洁引用块→声道源（#248：归属核对用）
        self.舱s: dict = {}            # #265 知识舱室（名字→知识舱）
        self.当前舱 = None             # 当前读取目标舱（None=核心库）
        self.消化债 = 0.0              # #257 消化内驱：未消化量（自测预测误差积累）
        self.饱足 = 1.0                # #258 饱足因子：消化行为后吸引力衰减（防固着，
                                       #  16:37 动作熵塌缩实证：内压过强锁死单一动作）
        self.自评偏差 = 0.0            # meeting43·P6：自评（行为集中度）折成的**引擎可消费**信号
        self.诊断_解码调用 = 0         # meeting43·P3：活动链上 `codec.解码` 的调用计数（判据可查）
        self.行债: dict = {}           # (文件,行号)→(行文本, 债额)——逐行挂账，回放验证销账
        # #251 对话流机制：跨轮条件模型 + 上下文记忆窗口
        from collections import deque
        self.lm_d = HDC语言模型(self.codec, seed=seed + 3)   # 对话流模型（计数条件生成）
        self.对话流 = deque(maxlen=6)  # ("你"/"我", 文本, 会话)——最近 6 条上下文（带窗口）
        self.充分阈值 = 0.45           # #270 证据充分度（自发延答的内驱信号）
        self.思考中: dict = None       # dict(问题,到达拍,证据,拍数,会话)——思考态
        self.思考队列: deque = deque(maxlen=8)
        self.待投递: str = None        # 思考完成待投递的作答
        self.说话人: str = "我"        # 本拍提问者（多账号共用记忆池：检索隔离用）
        self.段说话人: dict = {}       # 记忆段→发言人（未标注=提问者本人）
        self._当前会话 = "主"          # 本拍所属窗口（多窗口：上下文窗口按会话隔离；记忆共享）
        # Phase3·自发承载：维度不是外部硬塞给它的条件，而是它自己自审「装得下装不下」后
        # 自愿提出的扩容请求。自审发生在**它自己的节律**里（学一句后 / 自主拍），落成它自己的
        # 自言；宿主（文本态）只做机械重放重建，不做条件判断。
        self.待扩容: int = None        # 生命自审后给出的目标维（None=它觉得自己还装得下）
        self._重放中 = False           # 载入重放期间关掉自审（重放=重建，不是新的学习）

    # ---------- 资源接入 ----------
    def 设世界(self, 世界: 隐藏世界) -> None:
        self.世界, self.动作数, self.世界seed = 世界, 世界.动作数, 世界.seed

    def 设教材(self, 书: list, 文件界: list) -> None:
        self.书, self.文件界 = 书, 文件界

    def 设开放域(self, 根s: list, 屏蔽s: list = ()) -> None:
        """#246：开放域授权（只读）。授权本身入账（接收类），边界由导航器构造保证。"""
        self.导航器 = 导航器(根s, 屏蔽s)
        self.账本.append(("接收", "开放域授权：允许=%s；屏蔽=%s（只读）" %
                          (self.导航器.根s, self.导航器.屏蔽)))

    # ---------- 知识舱室（#265：可插拔隔离） ----------
    def 挂舱(self, 名字: str) -> None:
        from 知识舱 import 知识舱 as _舱
        if 名字 not in self.舱s:
            self.舱s[名字] = _舱(名字, self.codec, self.seed + 7)
        self.舱s[名字].挂载恢复()
        self.当前舱 = self.舱s[名字]
        self.账本.append(("自我", "挂载知识舱：%s" % 名字))

    def 卸舱(self, 名字: str) -> None:
        舱 = self.舱s.get(名字)
        if 舱 is None:
            return
        行数 = len(舱.行s)
        舱.摘除()
        if self.当前舱 is 舱:
            self.当前舱 = None
        self.账本.append(("自我", "卸载知识舱：%s（剥离 %d 行知识）" % (名字, 行数)))

    # ---------- 自言工具 ----------
    def _引(self, 号: int):
        return (号, self.账本[号][1][:16])

    def _自言(self, 类型: str, 文本: str, 触发源, 引用号s) -> int:
        return self.自言.追加(self.拍, 类型, 文本, 触发源,
                              [self._引(i) for i in 引用号s])

    def _账本计数(self, 类型: str) -> int:
        return sum(1 for t, _ in self.账本 if t == 类型)

    # ---------- 假设环一拍（v9 扩展：假设入账 + 引擎步进；束更新数学与 v9 逐位同） ----------
    def 假设环一拍(self, 世界: 隐藏世界) -> dict:
        n = self.动作数
        未测 = [i for i in range(n) if i not in self.试次]
        if 未测:
            i = 未测[0]
        else:
            i = max(range(n), key=lambda j: (self.上误差.get(j, 0), -self.试次[j]))
        p = self._预测(i)
        if p is None:
            假设文本 = "动作%d：还没试过，没有可预测的方向。" % i
        else:
            序 = np.argsort(-np.abs(p))[:3]
            假设文本 = "动作%d：我预测 %s。" % (
                i, "、".join("第%d维往%s" % (d + 1, "正" if p[d] > 0 else "负")
                             for d in 序))
        self.账本.append(("假设", 假设文本))
        假设号 = len(self.账本) - 1
        实际 = 世界.行动(i)
        实向 = 实际 / (np.linalg.norm(实际) + 1e-9)
        cosv = None
        if p is not None:
            cosv = float(p @ 实向)
            self.上误差[i] = 1.0 - cosv
            if cosv >= 0.8 and self.试次.get(i, 0) >= 1:
                首达 = not any(t == "顿悟" and x.get("动作") == i
                              for t, x in [(e[0], e[1]) for e in self.顿悟账本])
                if 首达:
                    self.顿悟账本.append(("顿悟", dict(动作=i, cos=round(cosv, 3)),
                                          (self.试次[i],)))
        self.束[i] = self.束.get(i, np.zeros(n)) + np.sign(实际)
        self.试次[i] = self.试次.get(i, 0) + 1
        检验文本 = "动作%d cos=%s" % (i, "未测" if cosv is None else round(cosv, 3))
        self.账本.append(("检验", 检验文本))
        检验号 = len(self.账本) - 1
        rec = self._引擎步(self.感知向量(self.codec.编码("检验 " + 检验文本)))
        self._吃(rec)
        return dict(动作=i, 预测cos=cosv, 假设号=假设号, 检验号=检验号,
                    预测文本=假设文本)

    # ---------- 三通道元控制（+#246 开放域=四通道） ----------
    def _通道值(self) -> dict:
        if self.世界 is None:
            v行动 = 0.0                                  # 行动通道未接入（无世界可实验）
        else:
            未试 = sum(1 for i in range(self.动作数) if self.试次.get(i, 0) == 0)
            误差s = list(self.上误差.values())
            v行动 = 0.5 * (未试 / self.动作数) + \
                0.5 * (sum(误差s) / len(误差s) if 误差s else 1.0)
        if self.书 and len(self.书) > 1:
            未读率 = 1.0 - self.位置 / (len(self.书) - 1)
            v教材 = 0.5 * max(未读率, 0.0) + 0.5 * (1.0 - self.上命中)
        else:
            v教材 = 0.0
        v文件 = 0.0
        if self.导航器 is not None:
            有活 = (bool(self.导航器.栈)
                    or self.导航器.当前文件 is not None
                    or self.导航器.根游标 < len(self.导航器.根s))
            if 有活:
                栈内文件 = sum(1 for _, w in self.导航器.栈
                              for e in w if not e[2])
                遇 = self.导航器.已读文件 + 栈内文件
                未看率 = 1.0 - self.导航器.已读文件 / max(遇, 1)
                v文件 = 0.5 * 未看率 + 0.5 * (1.0 - self.开放上命中)
        return dict(行动=v行动, 教材=v教材, 用户=0.0, 文件域=v文件)
        # 用户通道=中断驱动（断供期无外部事件）

    def _自评条目(self, 触发源) -> int:
        desc, H = self.行为分布()
        文本 = ("第 %d 拍：记忆 %d 段、账本 %d 条（检验 %d、假设 %d、教材 %d）、"
                "能量 %.2f、行为熵 %.2f bit、消化债 %.2f、自言 %d 条。" %
                (self.拍, len(self.codec._段), len(self.账本),
                 self._账本计数("检验"), self._账本计数("假设"),
                 self._账本计数("教材"), self.e if self.e is not None else -1.0,
                 H, self.消化债, len(self.自言.流)))
        最近 = len(self.账本) - 1
        # meeting43·P6：自评不止"写一句话"——把**行为集中度**折成引擎可消费的自评偏差
        # （熵越低 = 行为越集中 = 越该放开探索）；供决策层 _应答计划 读入（默认 0 无影响）。
        import math
        Hmax = math.log2(max(int(getattr(self, "动作数", 2)), 2))
        self.自评偏差 = float(np.clip(1.0 - H / Hmax, -1.0, 1.0)) if Hmax > 0 else 0.0
        return self._自言("自评", 文本, 触发源, [最近])

    # ---------- 自发承载自审（Phase3：维度伸缩的「自愿」面） ----------
    def 自审承载(self, 触发源=("内生", "承载自审")) -> dict:
        """生命自己掂量「我装得下吗」——不是外部塞给它的条件，是它自审后的自愿。

        驱动量 = 它自己学到的记忆量（codec._段）；「需要多大」由 维度伸缩.维度伸缩器 给出
        （纯确定性，只涨不缩）。**判据的归属在本生命内部**：当它自审出「学的东西已经超出
        我当前的承载」时，它把「想把自己撑大一点」写成自己的计划自言，并留下 待扩容 请求；
        宿主只机械执行重建。学得越多→越可能自愿扩容（所有者：「学得越多维度越多」）。"""
        if self._重放中:                                       # 重放=重建，不是新学习
            return dict(自愿=False, 学习量=len(self.codec._段), 原因="重放中")
        from 维度伸缩 import 维度伸缩器
        from 通用接口 import 当前D
        当D = 当前D()
        学习量 = len(self.codec._段)
        目 = 维度伸缩器().目标D(学习量)
        if 目 > 当D:
            self.账本.append(("自我", "承载自审：已学 %d 段，自愿把承载从 %d 维撑到 %d 维。"
                              % (学习量, 当D, 目)))
            引 = [len(self.账本) - 1]
            self._自言("计划",
                       "我学到的东西越来越多了（%d 段），感觉快装不下了——我想把自己的"
                       "容量撑大一点：%d→%d。" % (学习量, 当D, 目),
                       触发源, 引)
            self.待扩容 = int(目)
            return dict(自愿=True, 学习量=学习量, 当前D=当D, 目标D=int(目))
        return dict(自愿=False, 学习量=学习量, 当前D=当D, 目标D=目)

    # ---------- 断供一拍（无外部输入；引擎步进+通道动作+自言） ----------
    def 断供一拍(self) -> dict:
        assert (self.世界 is not None) or (self.导航器 is not None) or \
            (self.书 is not None), "断供前必须 设世界()/设教材()/设开放域() 至少其一"
        self.事件史.append(("断供", 1))
        if self.思考中 is not None:                         # #270 思考态优先（像人放下文件先想）
            会话 = self.思考中.get("会话", "主")             # 多窗口：答完回原窗口
            self._思考一拍()
            投递 = self.待投递                               # 思考完成 → 作答回传（延答投递）
            self.待投递 = None
            return dict(通道="思考", 投递=投递, 会话=会话)
        if self.断供拍数 % self.块长 == 0:                  # 块边界：重评+计划自言
            值 = self._通道值()
            可选 = [(k, v) for k, v in 值.items() if k in ("行动", "教材", "文件域")
                    and v > 0]
            if 可选:
                新通道 = max(可选, key=lambda kv: kv[1])[0]
            else:                                           # 全零平局→按已接入资源回退
                新通道 = ("文件域" if self.导航器 is not None else
                          "教材" if self.书 is not None else "行动")
            引号s = [len(self.账本) - 1] if self.账本 else None
            if 引号s is None:
                self.账本.append(("检验", "动作- cos=未测"))
                引号s = [len(self.账本) - 1]
            self._自言("计划",
                       "通道值：行动 %.2f / 教材 %.2f / 文件域 %.2f / 用户 %.2f"
                       "（无外部事件）——接下来 %d 拍走%s通道。" %
                       (值["行动"], 值["教材"], 值["文件域"], 值["用户"],
                        self.块长, 新通道),
                       ("内生", "块边界调度"), 引号s)
            if 新通道 != self.通道:
                self.通道 = 新通道
            self.调度史.append((self.断供拍数, self.通道))
        if self.通道 == "文件域":
            self.开放拍()
        elif self.通道 == "行动":
            r = self.假设环一拍(self.世界)
            i = r["动作"]
            if r["预测cos"] is None:
                self._自言("疑问", "动作 %d 我还没试过——它会把世界往哪边推？" % i,
                           ("内生", "覆盖缺口 %d/%d" % (
                               sum(1 for k in range(self.动作数)
                                   if self.试次.get(k, 0) == 0), self.动作数)),
                           [r["假设号"]])
            else:
                self._自言("预测", r["预测文本"],
                           ("内生", "试次=%d" % self.试次[i]), [r["假设号"]])
                self._自言("复盘", "动作 %d 实际 cos %.2f——误差 %.2f。" %
                           (i, r["预测cos"], 1.0 - r["预测cos"]),
                           ("内生", "检验返回"), [r["检验号"], r["假设号"]])
                if self.试次[i] % self.假设重述间隔 == 0:
                    self._自言("假设", "我猜动作 %d 的效果方向已经稳定——上次 cos %.2f。" %
                               (i, r["预测cos"]),
                               ("内生", "束累计=%d 试" % self.试次[i]),
                               [r["检验号"]])
        else:
            n0, d0 = len(self.账本), len(self.顿悟账本)
            rb = self.读一拍(self.书, self.文件界)
            rec = dict(动作=rb["动作"])
            if len(self.账本) > n0:                          # 细读发生了（门控 h<0.5）
                行2 = self.账本[n0][1]
                self.上命中 = self.首遇命中.get(rb["位置"] - 1, 0.0) \
                    if rb["位置"] > 0 else self.上命中
                self._自言("疑问", "『%s…』我还预测不了——细读收进教材声道。" % 行2[:10],
                           ("内生", "注意门控"), [n0])
            if len(self.顿悟账本) > d0:                      # 自查顿悟（消化/连接）
                for t, x, num in self.顿悟账本[d0:]:
                    引 = [n0] if len(self.账本) > n0 else [len(self.账本) - 1]
                    if t == "消化":
                        self._自言("复盘", "消化顿悟：『%s…』从命中 %.2f 到 %.2f——"
                                   "当时不懂、现在懂了。" % (str(x)[:10], num[0], num[1]),
                                   ("内生", "重读命中跃迁"), 引)
                    else:
                        self._自言("复盘", "连接顿悟：『%s…』与『%s…』两段强命中——"
                                   "发现跨段连接。" % (str(x[0])[:10], str(x[1])[:10]),
                                   ("内生", "联想检索"), 引)
        if self.断供拍数 % self.自评周期 == self.自评周期 - 1:
            self._自评条目(("内生", "周期自评"))
        self.断供拍数 += 1
        self.自审承载(("内生", "断供承载自审"))   # Phase3·自发：没人说话时它也自己掂量承载
        return dict(通道=self.通道)

    # ---------- 开放域（#246：导航式自主探索；决策先落自言，后调导航器原语） ----------
    def 开放拍(self) -> None:
        导 = self.导航器
        if not self.账本:
            return
        引 = [len(self.账本) - 1]
        if 导.当前文件 is not None:                         # ① 行级阅读（文件未完）
            self._文件读一拍()
            return
        if 导.栈 and not 导.栈[-1][1]:                      # ② 目录耗尽 → 复盘+回退
            目录, _ = 导.弹栈()
            self._自言("复盘", "『%s』看完了——回上一层继续。" %
                       (os.path.basename(目录) or 目录),
                       ("内生", "目录耗尽"), 引)
            return
        if not 导.栈:                                       # ③ 启动下一个根
            根 = 导.下一根()
            if 根 is None:
                self._自言("自评", "开放域 %d 个根我都走过了——覆盖完成。" % len(导.根s),
                           ("内生", "开放域覆盖完成"), 引)
                return
            条目, 屏蔽数 = 导.列目录(根)
            if 屏蔽数 is None:                               # 有界 IO：超时/异常 → 跳过
                self._自言("复盘", "【调用】list_dir('%s')【观察】失败（%s）——剔除根。"
                           % (根, str(条目)[:16]), ("内生", "有界 IO 跳过"), 引)
                return
            导.推栈(根, 条目)
            self.账本.append(("行动", "列目录 %s：%d 项（屏蔽 %d）" %
                              (根, len(条目), 屏蔽数)))
            self._自言("疑问", "【思考】开放域还有没看过的根——【调用】list_dir('%s')"
                       "【观察】%d 项、屏蔽 %d——先看什么？" % (根, len(条目), 屏蔽数),
                       ("内生", "开放域启动"), [len(self.账本) - 1])
            return
        # ④ 选题：下一待访条目（子目录优先、字典序——固定序内生选题）
        _, 待访 = 导.栈[-1]
        名, 全路径, 是否目录 = 待访.pop(0)
        if 是否目录:
            条目, 屏蔽数 = 导.列目录(全路径)
            if 屏蔽数 is None:                               # 有界 IO：超时/异常 → 跳过
                self._自言("复盘", "【思考】要进『%s』——【调用】list_dir('%s')"
                           "【观察】失败（%s）——跳过。" % (名, 全路径, str(条目)[:16]),
                           ("内生", "有界 IO 跳过"), [len(self.账本) - 1])
                return
            导.推栈(全路径, 条目)
            self.账本.append(("行动", "列目录 %s：%d 项（屏蔽 %d）" %
                              (全路径, len(条目), 屏蔽数)))
            self._自言("计划", "【思考】『%s』还没看过——【调用】list_dir('%s')"
                       "【观察】%d 项。" % (名, 全路径, len(条目)),
                       ("内生", "导航选题"), [len(self.账本) - 1])
        else:
            n = 导.打开文件(全路径)
            if n is None or n == 0:
                if n == 0:
                    导.当前文件 = None
                self._自言("复盘", "【思考】想读『%s』——【调用】read_file('%s')"
                           "【观察】无可读文本行（二进制/超限/空）——跳过。" % (名, 名),
                           ("内生", "跳过判定"), [len(self.账本) - 1])
                return
            self._自言("计划", "【思考】要看内容——【调用】read_file('%s')"
                       "【观察】%d 行可读。" % (名, n),
                       ("内生", "导航选题"), [len(self.账本) - 1])

    def _文件读一拍(self) -> None:
        导 = self.导航器
        行 = 导.下一行()
        if 行 is None:
            return
        f = 导.当前文件
        pos = 导.行位置 - 1
        u = self.感知向量(self.codec.编码(行))
        rec = self._引擎步(u)
        self._吃(rec)
        h = self.当前舱.命中(行) if self.当前舱 is not None \
            else self.lm_m.三元命中率(行)
        key = (f, pos)
        self.文件首遇.setdefault(key, h)
        教材号 = None
        if h < 0.5:                                         # 注意门控：预测不了才细读
            if self.当前舱 is not None:                     # #265 舱化：知识入舱不混核心
                self.当前舱.存行(行)
                self.账本.append((self.当前舱.名字, 行))
            else:
                if self.注册去重(行):
                    self.段源[行] = "教材"
                self.lm_m.学句(行)
                self.账本.append(("教材", 行))
            教材号 = len(self.账本) - 1
            导.细读行 += 1
            导.文件细读 += 1
            导.读迹.append((行, f))
            self.行债[(f, pos)] = (行, 1.0 - h)             # #257：逐行挂账
            # #258：吸收能力与饱足耦合——饱足低时新债缓记（防债再生压倒饱足→动作锁死）
            self.消化债 += (1.0 - h) * max(self.饱足, 0.1)
            self._自言("疑问", "【调用】read_line()【观察】『%s…』命中 %.2f——"
                       "预测不了，细读收进教材声道。" % (行[:8], h),
                       ("内生", "注意门控"), [教材号])
        else:
            导.略过行 += 1
            导.文件略过 += 1
        动作 = 动作名[int(rec["a星序号"]) % 4]
        if 动作 in ("提问", "转移话题"):                    # #258 饱足动力学：消化行为
            self.饱足 = self.饱足 * 0.2                     # 执行后吸引力衰减（无地板：
        else:                                               #  0.05 残余仍可锁死，16:43 实证）
            self.饱足 = min(1.0, self.饱足 * 1.6 + 0.05)    # 间隔后恢复
        if 动作 == "转移话题":                              # 联想：新内容×账本旧知识
            hits = self._检索(行, k=2)
            if (len(hits) == 2 and hits[0][1] >= 0.5 and hits[1][1] >= 0.5
                    and hits[0][0] != hits[1][0]):
                self.消化债 = max(0.0, self.消化债 - 0.05)  # 关联命中→内压小降
                self._自言("复盘",
                           "连接顿悟：『%s…』让我想起『%s…』和『%s…』——旧知识连上了。" %
                           (行[:8], str(hits[0][0])[:8], str(hits[1][0])[:8]),
                           ("内生", "联想检索"),
                           [教材号 if 教材号 is not None else len(self.账本) - 1])
        elif 动作 == "提问":                                # 停下想：内部生成+反刍销账
            rng = np.random.default_rng((self.seed, self.拍, 7))
            self.lm.生成(种子="", 温度=1.0, rng=rng, 最大长=16)
            if self.行债 and self.消化债 > 0.1:             # #257 债压下的反刍：
                键 = next(iter(self.行债))                   # 取最早未消化行（字典序=插入序）
                行0, 债额 = self.行债.pop(键)
                h0 = self.lm_m.三元命中率(行0)
                if h0 >= 0.5:                               # 验证通过→销账（内压消退）
                    self.消化债 = max(0.0, self.消化债 - 债额)
                    self._自言("复盘", "【回放消化】『%s…』现在能预测了（%.2f）"
                               "——销债 %.2f，剩 %.2f。" %
                               (行0[:8], h0, 债额, self.消化债),
                               ("内生", "回放消化"), [len(self.账本) - 1])
                else:                                       # 仍预测不了→挂回，继续压着
                    self.行债[键] = (行0, 债额)
                    self._自言("疑问", "【回放】『%s…』仍预测不了（%.2f）——"
                               "压着，继续。" % (行0[:8], h0),
                               ("内生", "回放未过"), [len(self.账本) - 1])
        h1 = self.lm_m.三元命中率(行)
        self.开放上命中 = h1
        if self.文件首遇[key] < 0.5 <= h1:                  # 消化顿悟
            self._自言("复盘", "消化顿悟：『%s…』从命中 %.2f 到 %.2f——现在能预测了。" %
                       (行[:8], self.文件首遇[key], h1),
                       ("内生", "门控命中跃迁"),
                       [教材号 if 教材号 is not None else len(self.账本) - 1])
        if 导.文件读完():                                    # ⑤ 读完复盘
            self._自言("复盘", "『%s』读完了——细读 %d 行、略过 %d 行。消化债 %.2f"
                       "（没嚼透的自己会回头）。" %
                       (os.path.basename(f), 导.文件细读, 导.文件略过, self.消化债),
                       ("内生", "文件读完"), [len(self.账本) - 1])
            导.已读文件 += 1
            导.文件细读 = 导.文件略过 = 0
            导.当前文件 = None

    def _吃(self, rec: dict) -> None:
        super()._吃(rec)
        self.最近候选ê = rec.get("候选ê")     # #272 旁路用：候选自由能快照

    # ---------- 引擎步进（#257：消化内驱——未消化量作为内生偏置项，与私偏 κ 同构） ----------
    def _引擎步(self, u: np.ndarray) -> dict:
        b = self.drv.偏置() if self.拍 >= self.引导拍 else None
        if b is not None:
            私 = self.eng.生成私有()
            wdir = self.W_priv @ np.asarray(私["q8"], dtype=np.float64)
            cand = self.eng.hdc.动作候选
            b = [float(b[i]) + self.私偏κ * float(np.dot(cand[i], wdir))
                 for i in range(len(cand))]
            if self.消化债 > 0:                             # 内压→偏向 反刍/关联行为
                κd = min(self.消化债 * 0.01, 2.0) * self.饱足
                b[1] += κd * 0.6                            # 提问位→停下想+反刍销账
                b[3] += κd * 0.6                            # 转移话题位→联想检索
        else:
            私 = self.eng.生成私有()
        gate, pred = self.drv.学习门(self.i_prev)
        rec = self.eng.通电步进R1(u, a_prev=self.a_prev, 内生偏置=b, efe_mode=True,
                                  学习门=gate if self.拍 >= self.引导拍 else None,
                                  私有上下文=self.prev_c)
        # ★以下状态推进与 v2._引擎步 逐行同构（此前重写漏掉 → 引擎状态冻结 → 熵塌缩）
        self.drv.更新(rec["候选ê"])
        self.a_prev = rec["a星"]
        self.i_prev = int(rec["a星序号"])
        self.prev_c = int(私["c"])
        self.拍 += 1
        if (self.拍 % 20) == 0:
            self.eng.维护R2(q=私["q8"])
        return rec

    def _延答(self, 用户文本: str) -> dict:
        """#270 延答：证据不足→思考态。表达层即时 ack，作答延后（自发，非公式）。"""
        self.账本.append(("接收", 用户文本))
        rec = self._引擎步(self.感知向量(self.codec.编码("思考 " + 用户文本)))
        self._吃(rec)
        self.账本.append(("行动", "思考中：暂无输出"))
        self.思考中 = dict(问题=用户文本, 到达拍=self.拍, 证据=[], 拍数=0,
                          会话=self._当前会话)          # 多窗口：考完回到问它的那个窗口
        接收号 = len(self.账本) - 2
        self._自言("疑问", "【思考】『%s…』我的证据还不够——停下来想，"
                   "想好了再答。" % 用户文本[:10],
                   ("外部", 接收号), [接收号])
        self.历史.append((self.拍, 用户文本,
                          动作名[int(rec["a星序号"]) % 4], "", "（我在想）"))
        return dict(响应="嗯，这个我得琢磨一下，想清楚了再跟你说。", e=self.e, ñ=self.ñ,
                    延答=True)

    def _思考一拍(self) -> None:
        """思考态：每拍自选取证（迭代检索）+重测充分度→作答或如实承认。无固定阶段。"""
        思 = self.思考中
        q = 思["问题"]
        思["拍数"] += 1
        查询 = 思["证据"][-1][0] if 思["证据"] else q
        best = None
        点名舱 = [n for n in self.舱s if n in q]            # #265：问题点名舱→限定取证
        for 词 in (q, 查询):
            for 段, 分 in self._检索(词, k=3):
                if 分 < self.阈值 or (段, "核心") in [(e[0], e[1]) for e in 思["证据"]]:
                    continue
                if best is None or 分 > best[2]:
                    best = (段, "核心", 分)
            舱项 = [(n, self.舱s[n]) for n in 点名舱] if 点名舱 \
                else [(n, c) for n, c in self.舱s.items() if c.挂载]
            for 名字, 舱 in 舱项:
                if not 舱.挂载:
                    continue
                for 段, 分 in 舱.检索(词, k=2):
                    if 分 < 0.4 or (段, 名字) in [(e[0], e[1]) for e in 思["证据"]]:
                        continue
                    if best is None or 分 > best[2]:
                        best = (段, 名字, 分)
        if best is not None:
            段, 源, 分 = best
            思["证据"].append((段, 源, 分))
            self._自言("假设", "【取材】想到『%s…』（%.2f）——和问题有关。"
                       % (段[:10], 分), ("内生", "思考取材"), [len(self.账本) - 1])
        充分 = max([e[2] for e in 思["证据"]], default=0.0)
        账引: list = []
        # #285 深度思考：饱和上限 20→128 拍（更长的思考链；证据足够仍会提前收敛）
        if (len(思["证据"]) >= 2 and 充分 >= self.充分阈值) or 充分 >= 0.6 \
                or 思["拍数"] >= 128:
            if 思["证据"]:
                链 = []
                for 段, 源, 分 in 思["证据"]:
                    归属 = {"用户": "你说过", "自我": "我自己提过",
                            "教材": "教材里说"}.get(源, "《%s》里说" % 源)
                    链.append("%s：『%s』" % (归属, self._清洁块(段)))
                    账引 += [k for k, (t, x) in enumerate(self.账本)
                             if x == 段][:1]
                响应 = ("我想了 %d 拍——" % 思["拍数"]) + "；由此，".join(链) + "。"
            else:
                响应 = ("我想了 %d 拍，还是没想出来——我先如实说：这个我不知道。"
                        % 思["拍数"])
            self.回复台账.append(dict(号=len(self.回复台账) + 1, 问=思["问题"],
                                      响应=响应, 支撑自言=[],
                                      支撑账本=sorted(set(账引)),
                                      延迟拍=思["拍数"]))
            self.待投递 = 响应
            self._自言("复盘", "【作答】%s" % 响应[:36],
                       ("内生", "思考完成"), [len(self.账本) - 1])
            self.思考中 = None

    # ---------- 对外回复（可溯源；#248 自然语言表面） ----------
    def 段源查(self, 段: str):
        """段源查询：全段 或 清洁引用块（#248）。"""
        return self.段源.get(段) or self.清洁引用源.get(段)

    @staticmethod
    def _清洁块(段: str) -> str:
        """取段内最长无 markdown 装饰符的连续块——仍是账本原文的精确子串（审计不变）。"""
        块s = re.split(r"[|`\*_>#\[\]~]+", 段)
        块 = max(块s, key=len).strip()
        return 块 if len(块) >= 8 else 段[:24].strip()

    def _自言检索(self, q: str, 阈值: float = 0.45):
        """#249 自言流检索：用他自己的历史话语回答。有界扫描最近 3000 条，
        字符集重叠（v3 槽位同口径）。返回 (自言号, 原文, 分) 或 None。"""
        q集 = _关键词集(q)
        if not q集:
            return None
        最佳 = None
        for e in self.自言.流[-3000:]:
            t = e["文本"]
            外部应答 = e["触发源"][0] == "外部"
            if 外部应答:
                # 应答记录类自言：只对内嵌问题部分匹配（防泛用应答框架误检）
                m = re.search(r"『([^』]+)』", t)
                t集 = _关键词集(m.group(1)) if m else _关键词集(t)
            else:
                t集 = _关键词集(t)
            if not t集:
                continue
            分 = len(q集 & t集) / max(min(len(q集), len(t集)), 1)
            生效阈值 = 0.7 if 外部应答 else 阈值           # 应答记录须近同问题（防误检）
            if 最佳 is None or 分 > 最佳[2]:
                最佳 = (e["号"], e["文本"], 分, 生效阈值)
        if 最佳 and 最佳[2] >= 最佳[3]:
            return 最佳[:3]
        return None

    # ---------- 检索（精度层·synthesis 顶覆盖；基链 learner/dialogue_v2 不动，守 C14） ----------
    @staticmethod
    def _二元集(s: str) -> set:
        """去标点后的**字 2-gram 集**：比单字集多编码「相邻 / 顺序」，是精度的关键。"""
        t = "".join(c for c in str(s) if c.strip() and c not in 停用符)
        return {t[i:i + 2] for i in range(len(t) - 1)}

    def _语料统计(self):
        """语料自统计 → (段2gram集列表, IDF)。**IDF 由数据自身得出**（不是常数、不是人设
        权重表）——符合「零参数 · 自我调控」：g 出现在越多段里（如『什么』『怎么』）权重越低，
        越是某条记忆专属的字组（如『我叫』『老王』）权重越高。缓存按段数失效，段数不变即复用。"""
        n = len(self.codec._段)
        缓 = getattr(self, "_语料统计_", None)
        if 缓 is not None and 缓[0] == n:
            return 缓[1], 缓[2]
        gs = [self._二元集(t) for t in self.codec._段]
        df: dict = {}
        for g集 in gs:
            for g in g集:
                df[g] = df.get(g, 0) + 1
        idf = {g: float(np.log((n + 1.0) / (c + 0.5))) for g, c in df.items()}
        self._语料统计_ = (n, gs, idf)
        return gs, idf

    def _检索(self, 文本: str, k: int = 2) -> list:
        """检索（精度层）：基链第三通道是**单字集合重叠**（|共享字| / min(字数)），对
        「我 / 什 / 么」这类功能字毫无鉴别力——实测：无关教材段与真答案同为 0.5 平局，
        这正是「输出不精准、并非它想表达的意思」的根因（所有者 2026-10-06）。此处覆盖为
        **IDF 加权的字 2-gram 覆盖率**：
            分 = Σ idf(g)·[g ∈ 段] / Σ idf(g)      （g 遍历查询的 2-gram）
        它度量「这条记忆到底含了多少问题的内容」；只命中普适字组时段分≈0，门限自然把噪声
        挡在外面——于是答不上来时它如实说「没学到」，而不是硬凑一条最像的糊弄。
        逐位保留基链口径：代码标识符整体匹配优先、近因平局裁决、top-k 组装。
        无梯度 / 无 attention / 无外挂模型，IDF 由语料自身统计（非先验权重表）。"""
        if not self.codec._段:
            return []
        段s = self.codec._段
        新g = self._二元集(文本)
        gs, idf = self._语料统计()
        # 分母只计**语料中确实存在**的查询字组：语料一大，连 df=1 的字组 IDF 都很大
        # (log(N/1.5))，若把"语料里根本没出现过的字组"也算进分母，合法问答的分会被这些
        # 够不着的重权项压到 0.2 量级、跌破既有门限（充分阈值/阈值按旧尺度标定）——实测。
        # 这些不存在字组几乎都是**跨词边界的构图碎片**（『王怎』『么了』），不是有效内容，
        # 故诚实地把它们排除在分母之外（并在无任何存在字组命中时让分归零）。
        权重总 = sum(idf.get(g, 0.0) for g in 新g)
        tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", 文本)
        # P4 语义通道（默认关）：查询的**双通道** HDC 码（bigram 编码 + 字袋）——与段向量作余弦证据。
        # 双通道与 `语言编解码器.解码` 同源（改述鲁棒）；字袋通道捕捉「字符语义对齐」，
        # bigram 通道捕捉「顺序结构」——因为精化后的字符向量承载共现语义。
        qv = qbag = qvn = qbn = None
        解码段 = None
        if 语义检索 or 解码通道:
            qv = self.codec.编码(文本)
            qbag = self.codec._字袋(文本)
            qvn = float(np.linalg.norm(qv)) + 1e-9
            qbn = float(np.linalg.norm(qbag)) + 1e-9
        if 解码通道 and 段s:                              # P3：把 解码 接入活动链（计数可判）
            解码段 = self.codec.解码(文本)
            self.诊断_解码调用 += 1
        sc = []
        for i, 旧文本 in enumerate(段s):
            if tokens and any(t in 旧文本 for t in tokens):
                sc.append((1.0, i))                       # 概念名整体匹配（最高通道）
                continue
            分 = 0.0
            if 权重总 > 0.0:
                共 = 新g & gs[i]
                分 = (sum(idf.get(g, 0.0) for g in 共) / 权重总) if 共 else 0.0
            if qv is not None:
                sv = self.codec._段向量[i]
                sb = self.codec._段字袋[i]
                c1 = float(qv @ sv) / (qvn * (float(np.linalg.norm(sv)) + 1e-9))
                c2 = float(qbag @ sb) / (qbn * (float(np.linalg.norm(sb)) + 1e-9))
                if 语义检索:
                    分 = (1.0 - 语义权重) * 分 + 语义权重 * max(0.0, c1, c2)
                if 解码通道 and 旧文本 == 解码段:          # 解码 的向量→文本命中 → 语义证据抬分
                    分 = max(分, 语义权重 * max(0.0, c1, c2))
            sc.append((分, i))
        sc.sort(reverse=True)
        s星 = sc[0][0]
        近因 = [i for s, i in sc if s >= s星 - 0.02]      # 近因平局裁决（同基链）
        最佳i = max(近因)
        最佳分 = next(s for s, i in sc if i == 最佳i)
        其余 = [(s, i) for s, i in sc if i != 最佳i][:max(k - 1, 0)]
        return [(段s[最佳i], 最佳分)] + [(段s[i], s) for s, i in 其余]

    # ---------- 身份模板（#262：显式可审计的角色定义） ----------
    def 加载身份(self, 模板) -> None:
        from identity import 身份模板
        assert isinstance(模板, 身份模板)
        self.身份 = 模板
        # ① 背景事实经自我声道入账（账本 自我 类、段源=自我）
        self._声道 = "自我"
        self.教_v2("\n".join(模板.背景), epochs=2, 记录=False)
        self._声道 = "用户"
        # ② 风格语料学入自声道
        for 句 in 模板.风格语料:
            self.学句(句)
            self.注册去重(句)
            self.段源[句] = "自我"
        # ③ 性格参数注入引擎动态
        if "私偏κ" in 模板.参数:
            self.私偏κ = float(模板.参数["私偏κ"])
        # ④ 教史登记（重放可恢复）+ 账本血统
        self.教史.append(("\n".join(模板.背景), 2, "自我"))
        self.教史.append(("\n".join(模板.风格语料), 1, "自我"))
        self.账本.append(("自我", "身份模板加载：%s（%s）" %
                          (模板.名字, 模板.性格描述[:20])))

    def _自报告(self, q: str) -> str:
        # —— 回忆问句（#路由修正）：「你还记得X吗」=问用户的事 → 走检索作答，
        #    不落入下方「叫什么/名字」自指分支（否则会答成 AI 自己的名字） ——
        if 记忆核心(q) is not None:
            return self._答问(q)
        # —— 身份背景匹配（#262）：2-gram 重叠 ≥0.25 → 用模板设定作答（账本可查） ——
        身 = getattr(self, "身份", None)
        if 身 is not None:
            疑问字 = set("你我什么怎吗哪呢的了吗是")
            q二元 = {q[i:i + 2] for i in range(len(q) - 1)
                    if not (set(q[i:i + 2]) & 疑问字)}
            最佳事实, 最佳分 = None, 0.0
            for 事实 in 身.背景:
                分 = sum(1 for g in q二元 if g in 事实) / max(len(q二元), 1)
                if 分 > 最佳分:
                    最佳事实, 最佳分 = 事实, 分
            if 最佳事实 is not None and 最佳分 >= 0.25:
                return ("按我的设定——『%s』。（这是我的身份设定，账本里查得到。）"
                        % 最佳事实)
        # —— 想什么（自然作答：报当前思考态/节律，不吐内部管线原文） ——
        if any(w in q for w in ("在想什么", "在想啥", "想什么", "在想些什么", "在想")):
            思 = self.思考中
            if 思 is not None:
                return "我正在琢磨『%s』——还没想透，想明白了再跟你说。" % str(思.get("问题"))[:20]
            return ("这会儿没卡着什么事——第 %d 拍，脑子是闲的，你要有话说我就接。"
                    % self.拍)
        # —— 自言流检索（#249）：用他自己的历史话语作答（自主表面） ——
        命中 = self._自言检索(q)
        if 命中 is not None and any(
                k in str(命中[1]) for k in ("【", "证据还不够", "想好了再答",
                                            "调用", "list_dir", "read_file")):
            命中 = None                                        # 内部管线痕迹：不作对答外泄（正常流）
        if 命中 is not None:
            号, 原文, 分 = 命中
            净 = re.sub(r"【[^】]*】", "", str(原文)).strip()   # 去内部通道标记（【思考】…不外泄）
            return ("这事我之前琢磨过——%s（自言自语 #%d；此刻第 %d 拍。）"
                    % (净[:72], 号, self.拍))
        # —— 意图路由（#248，引导件）：口语意图优先，全部回答来自真实遥测/账本 ——
        if any(w in q for w in ("累", "感觉怎么样", "有没有情绪", "会难过",
                                "过得怎么样", "还好吗", "状态怎么样", "怎么样")):
            _, H = self.行为分布()
            return ("还不错——我没有身体，谈不上疲惫或心情，但引擎数据一切正常：能量 "
                    "%.2f（活跃度）、行为熵 %.2f bit。我的状态都写在指标里，不写在"
                    "感觉里。" %
                    (self.e if self.e is not None else -1.0, H))
        if any(w in q for w in ("干嘛", "在做什么", "忙什么", "在干啥", "看什么",
                                "在看", "怎么决定", "哪些文件", "看了什么",
                                "学到", "探索")):
            导 = self.导航器
            if 导 is not None:
                现在 = os.path.basename(导.当前文件) if 导.当前文件 else \
                    (os.path.basename(导.栈[-1][0]) if 导.栈 else "(无)")
                最近 = list(dict.fromkeys(
                    os.path.basename(fn) for _, fn in 导.读迹[-60:]))[:3]
                n教材 = sum(1 for t, _ in self.账本 if t == "教材")
                return ("我在开放域里自主探索（只读）：正在看《%s》；刚看过 %s。已读 %d "
                        "个文件、细读 %d 行，全记进账本了（教材类共 %d 条）。看什么是我"
                        "按通道值和覆盖缺口自己定的，每次决定都在自言流里留了痕——"
                        "你可以查。" %
                        (现在, "、".join("《%s》" % x for x in 最近) or "-",
                         导.已读文件, 导.细读行, n教材))
        if any(w in q for w in ("叫什么", "名字", "是谁")):
            身 = getattr(self, "身份", None)
            if 身 is not None:
                return ("我叫 %s——%s。跑到第 %d 拍了，账本里 %d 条记录。"
                        "我的每句话都由引擎当下状态现场生成。" %
                        (身.名字, 身.性格描述, self.拍, len(self.账本)))
            return ("我叫 AHI——纯 HDC 动力学引擎的对话体（seed %d），已经跑到第 %d 拍。"
                    "我的每句话都由引擎当下状态现场生成，不是写死的。" %
                    (self.seed, self.拍))
        if "聊聊" in q:
            导 = self.导航器
            if 导 is not None and 导.读迹:
                最近 = list(dict.fromkeys(
                    os.path.basename(fn) for _, fn in 导.读迹[-60:]))[:3]
                return ("行呀。我最近在读《%s》，账本里攒了 %d 条记录、记了 %d 段东西。"
                        "你想听我读到的内容，还是聊聊我自己？" %
                        ("》《".join(最近), len(self.账本), len(self.codec._段)))
            return ("行呀。我的账本里有 %d 条记录、记了 %d 段东西。你想听哪块？"
                    % (len(self.账本), len(self.codec._段)))
        r = super()._自报告(q)
        if r.startswith("此刻状态：第"):                    # 遥测模板 → 自然短句（数值仍真）
            return ("现在是第 %d 拍，我记着 %d 段内容，能量 %.2f——一切正常。"
                    "你想聊点什么？" %
                    (self.拍, len(self.codec._段),
                     self.e if self.e is not None else -1.0))
        return r

    # ---------- 推理环（#267：思维链=条件生成+迭代检索；多跳传导至收敛） ----------
    def 推理环(self, q: str, 最大步: int = 6) -> dict:
        q集 = _关键词集(q)
        证据链 = []                                    # (段, 段源, 分)
        已用 = set()
        当前查询 = q
        for 跳 in range(最大步):
            # 条件生成思考句（以上下文为条件——Transformer 式功能，计数实现）
            rng = np.random.default_rng((self.seed, self.拍, 41, 跳))
            思考 = self.lm_m.生成(种子=当前查询[-4:], 温度=0.9, rng=rng, 最大长=20)
            # 迭代检索：问题与思考句 → 最强未用事实
            best = None
            for 词 in (当前查询, 思考):
                for 段, 分 in self._检索(词, k=3):
                    if 段 in 已用 or 分 < self.阈值:
                        continue
                    if best is None or 分 > best[1]:
                        best = (段, 分)
            if best is None:
                break                                  # 进展饱和：无新强证据
            段, 分 = best
            已用.add(段)
            证据链.append((段, self.段源查(段) or self.段源.get(段, "教材"), 分))
            # 多跳：事实中『问题之外的新内容』成为下一跳查询
            新词 = "".join(c for c in 段 if c not in q集).strip()
            if len(新词) < 2:
                break
            当前查询 = 新词[:12]
        return dict(证据链=证据链, 跳数=len(证据链))

    # ---------- 应答决策（决策层·synthesis 顶）：内在状态决定「说什么/说多少/多愿意认」 ----------
    def _应答计划(self, q: str) -> dict:
        """应答计划：**由引擎当下的内在信号决定**，不是写死的检索深度/门限。

        所有者 2026-10-06 指出「引擎核心的决策方式过于草率」——旧的 `_答问` 用常数 k=6 与固定
        阈值 + 一串 `if 关键词 in q` 决定输出，内在状态（消化债/饱足/新异）根本没参与。这里
        把决策交还给生命自身的动力学（沿用 _引擎步 已有的信号，不新增权重表）：
          · 消化债↑（心里攒着没消化的多）→ 检索更深、更愿意把弱证据端出来如实说；
          · 饱足↑（刚说够了）→ 说得更短（体裁=简）；
          · 是明确的回忆问句（记忆核心）→ 接受门更松（它自己也想起来）。
        """
        债 = float(getattr(self, "消化债", 0.0) or 0.0)
        饱 = float(getattr(self, "饱足", 1.0) or 1.0)
        自评 = float(getattr(self, "自评偏差", 0.0) or 0.0)     # P6：自评信号进决策（默认 0）
        回忆问 = 记忆核心(q) is not None
        k = 3 + int(round(min(max(债, 0.0), 1.5) * 2))          # 债多→看更多记忆候选
        k += int(round(max(自评, 0.0) * 2))                     # P6：行为越集中→检索更深（自评驱动）
        门 = float(self.阈值) * (1.0 - 0.3 * min(max(债, 0.0), 1.0))
        门 *= (1.0 - 0.1 * 自评)                                # P6：自评偏差微调接受门（默认 0 无影响）
        if 回忆问:
            门 *= 0.6                                            # 明确回忆→更愿认（自己也想着呢）
        # P7（默认关）：把「线性缩放」升级为「**饱和 + 乘性联合**」，并让**引擎新异 ñ**（旧决策完全不含）
        #   参与——三路内生信号先经 tanh 饱和、再乘性耦合，避免线性叠加/无界放大。默认关 ⇒ 旧口径不变。
        if 非线性决策:
            ñ = float(getattr(self, "ñ", 1.0) or 1.0)
            g债 = float(np.tanh(1.5 * max(债, 0.0)))            # 消化债 → 饱和
            r新 = float(np.tanh(2.0 * (ñ - 0.5)))               # 新异 → 饱和（P7 新增耦合，旧决策无此项）
            门 = float(self.阈值) * (1.0 - 0.35 * g债) * (1.0 - 0.12 * 自评) * (1.0 - 0.15 * max(r新, 0.0))
            k = 3 + int(round(3.0 * g债)) + int(round(2.0 * max(自评, 0.0))) \
                + int(round(3.0 * max(r新, 0.0)))
            if 回忆问:
                门 *= 0.6
            门 = float(min(max(门, 0.02), float(self.阈值)))    # 有界
            k = int(min(max(k, 3), 12))                         # 有界
        体裁 = "简" if 饱 > 0.8 else "详"
        # —— 选路（决策层统一裁决，_答问 只按此分派）：问题意图 × 内在信号 ——
        # 旧实现把"走行动知识/上下文/记忆"散成 _答问 里的一串 if 关键词分支；此处收拢成
        # 一个计划字段 路，使"选路"与"多深/多松/多说"同属决策层，表达层只负责渲染。
        动作s = [int(s) for s in re.findall(r"动作\s*(\d+)", q)]
        动作 = next((i for i in 动作s
                     if any(t == "检验" and x.startswith("动作%d " % i)
                            for t, x in self.账本)), None)     # 账本里真有检验记录才算
        指示 = any(w in q for w in 指示词)
        if 动作 is not None:
            路 = "行动"                                        # 查证式：账本有据才走行动知识
        elif 指示 and any(w in q for w in ("什么", "啥")):
            路 = "上下文内容"                                  # 指示语问内容（『刚才说了什么』）
        elif 指示:
            路 = "上下文话题"                                  # topical 指示语→窗口检索
        else:
            路 = "记忆"                                        # 默认：记忆库检索作答
        return dict(k=k, 门=门, 体裁=体裁, 债=债, 饱=饱, 回忆问=回忆问,
                    路=路, 动作=动作)

    def _凝练(self, 段: str, q: str, 上限: int = 26) -> str:
        """把记忆段凝练成「与问题最相关的那个子句」——**不再整段复述**。

        这是表达层的核心改动：旧路径把记忆段原文塞进引号（『整段』）= 一台引用机，
        输出并非它想表达的意思（所有者 2026-10-06）。改为：按问句关键词挑出最相关的
        子句；子句仍过长时，围绕问题词作窗口截取。字回归：复述率↓。
        """
        s = self._清洁块(段)
        子句 = [x for x in re.split(r"[。！？；，、：:「」『』\n]", s) if x.strip()]
        if not 子句:
            return s[:上限]
        q集 = _关键词集(q)
        选 = max(子句, key=lambda x: len(_关键词集(x) & q集)) if q集 else 子句[0]
        选 = 选.strip().lstrip("，、：: 「」『』")
        for 引导 in 引导词s:                              # 剥引导语：不把"跟你说个八卦"当要点
            if 选.startswith(引导):
                选 = 选[len(引导):].lstrip("，、：: ")
                break
        if len(选) > 上限:
            命中 = [i for i, c in enumerate(选) if c in q集]
            起 = max(0, 命中[0] - 3) if 命中 else 0
            选 = 选[起:起 + 上限]
        return 选

    @staticmethod
    def _共二字(a: str, b: str) -> bool:
        """a、b 是否有共享的 2-gram（用作连接语的"有来由"守卫：碎片与问题无关即弃）。"""
        A = "".join(c for c in str(a) if c.strip())
        B = "".join(c for c in str(b) if c.strip())
        if len(A) < 2 or len(B) < 2:
            return False
        return bool({A[i:i + 2] for i in range(len(A) - 1)}
                    & {B[i:i + 2] for i in range(len(B) - 1)})

    def _回执(self, 净: str, q: str = "") -> str:
        """陈述回执（表达层）：不把用户原话整段塞进引号，而是**点一下要点**再收下——
        既显示真听懂了，又不是引用机。要点=最长的实义子句（截到 22 字自然断）。
        框定语由内在状态（新异 ñ）决定，非写死单一串。"""
        全 = self._清洁块(净)
        子 = [x for x in re.split(r"[。！？；，、：:「」『』\n]", 全) if x.strip()]
        合 = [x for x in 子 if 4 <= len(x) <= 24]
        核 = max(合, key=len) if 合 else (子[0] if 子 else "")
        if len(核) > 22:
            核 = 核[:22] + "…"
        ñ = float(self.ñ if self.ñ is not None else 1.0)
        if not 核:
            return "嗯，我记下了。"
        if ñ >= 0.85:
            return "「%s」——头回听说，记下了。" % 核
        if ñ >= 0.6:
            return "「%s」——嗯，记下了。" % 核
        return "「%s」——这个我心里有数，还是收下了。" % 核

    def _应答表面(self, q: str, 响应: str) -> str:
        """应答表面（表达层·synthesis 顶）：把基链的模板/复述/旁白改写成"它自己的话"。
        · 陈述回执「…记下了：『<全文复述>』（记忆共 N 段…）」→ 短回执（不复述全文）
        · 去掉「（开头这句是我自己组的）」自披露旁白、清掉内部通道标记 【…】。"""
        s = str(响应)
        s = re.sub(r"（开头这句是我自己组的）", "", s)
        s = re.sub(r"【[^】]*】", "", s)
        m = re.search(r"记下了：『(.+?)』（记忆共\s*\d+\s*段[^）]*）。", s)
        if m:
            return self._回执(m.group(1), q)
        return s

    def _答问(self, q: str) -> str:
        _核 = 记忆核心(q)                                    # 回忆问句→用核心问句检索
        if _核:
            q = _核
        self.答问人 = getattr(self, "说话人", "我")
        计 = self._应答计划(q)                               # 决策层：问题意图 × 内在信号 → 统一选路
        路 = 计["路"]
        if 路 == "行动":                                     # 账本里真有该动作的检验记录
            return self._行动知识回答(int(计["动作"]))
        if 路 == "上下文内容":                                # 指示语问内容（『刚才说了什么』）
            from dialogue_v2 import 是问句 as _是问句
            候选 = [e[1] for e in reversed(self.对话流)
                    if e[0] == "你" and e[2] == self._当前会话
                    and e[1] != q and not _是问句(e[1])]
            引 = next((t for t in 候选 if len(t) >= 6), None) or \
                (候选[0] if 候选 else None)
            if 引:
                return ("刚才你说：『%s』——就在咱们刚才的对话里，我记着呢。"
                        % 引[:44])
        if 路 == "上下文话题":                                # topical 指示语→窗口检索
            引用 = self._上下文检索(q)
            if 引用 is not None:
                return ("刚才你说过『%s』——这事我记得，就在咱们刚才的对话里。"
                        "你想往深了聊哪一点？" % 引用[:44])
        hits = self._检索(q, k=计["k"])
        谁 = self.答问人                                      # 多人共用记忆池：本人段优先
        if len(hits) > 1:
            # 「本人」= **明确署了本人名**的段（其它窗口的人说的、以及未署名的自我/教材段
            # 都不算本人）——否则未署名的自我短句会被误当"本人"而压过跨窗口的真记忆
            # （实测：B 窗口问老王，『这一条我记下了。』0.286 压过 0.6 的八卦 → 答成"没学到"）。
            本人 = [h for h in hits if self.段说话人.get(h[0]) == 谁]
            其余 = [h for h in hits if self.段说话人.get(h[0]) != 谁]
            hits = 本人 + 其余
        段1, s1 = (hits[0] if hits else ("", 0.0))
        if s1 < 计["门"]:                                    # 核心未强命中 → 先查挂载舱
            for 名字, 舱 in self.舱s.items():
                if 舱.挂载:
                    hs = 舱.检索(q, k=1)
                    if hs and hs[0][1] >= 0.4:
                        return "《%s》里说：『%s』。" % (名字, self._凝练(hs[0][0], q))
        if not hits:
            推 = self.推理环(q)                             # #267 思维链：多跳检索推理
            if 推["跳数"] >= 2:
                链 = []
                for 段, 源, 分 in 推["证据链"]:
                    归属 = {"用户": "你说过", "自我": "我自己提过",
                            "教材": "教材里说"}.get(源, "账本里")
                    链.append("%s：『%s』" % (归属, self._凝练(段, q)))
                return ("我推理了一下——" + "；由此，".join(链) +
                        "。把这条链连起来就指向你的问题了。（每一跳都在账本里，可核对）")
            return "这个我真不知道——你先教教我，我再答。"
        段1, s1 = hits[0]
        段2 = hits[1] if len(hits) > 1 else None
        if s1 < 计["门"]:
            return ("这个我还没学到，不想瞎猜。要说最接近的一条是『%s』——但差得远。"
                    % self._凝练(段1, q))
        源 = self.段源.get(段1, "用户")
        归属 = {"用户": "你之前跟我说过", "自我": "我自己记着",
                "教材": "我从资料里读到过"}[源]
        说1 = self.段说话人.get(段1)                        # 跨窗口：这条是不是别的窗口的人说的
        if 源 == "用户" and 说1 is not None and 说1 != 谁:
            归属 = "%s那边跟我提过" % 说1
        引1 = self._凝练(段1, q)                             # 表达层：凝练子句，不再整段复述
        self.清洁引用源[引1] = 源
        # 只保留受守卫的条件开场（以问题尾部为条件）；无条件自由采样会复述记忆段
        # 充当“口头禅”（实测出现“我没有职业经历…”等事实回吐）——宁可无连接语，也不复述。
        连接 = self._条件开场(q) or ""
        补 = self._生成应答(段1, q) if 生成路 else None       # P4：HDC 现场生成（默认关；不通过=回退引用）
        if 段2 is not None and 段2[1] >= 计["门"] and 段2[0] != 段1 \
                and self.段源.get(段2[0], "用户") != "自我":   # 自我澄清/连接语不作「相关记忆」
            引2 = self._凝练(段2[0], q)
            self.清洁引用源[引2] = self.段源.get(段2[0], "用户")
            别人 = self.段说话人.get(段2[0])
            if 别人 is not None and 别人 != 谁:                # 交叉带记忆：标明是谁说的
                return "%s%s『%s』。另外，%s那边提过『%s』。%s" % (连接, 归属, 引1, 别人, 引2, 补 or "")
            return "%s%s『%s』。还有一条相关的『%s』。%s" % (连接, 归属, 引1, 引2, 补 or "")
        return "%s%s『%s』。%s" % (连接, 归属, 引1, 补 or "")

    # ---------- P4：HDC 现场生成（默认关；四重守卫，不通过如实回退检索引文） ----------
    def _生成应答(self, 段: str, q: str):
        """以查询尾部为条件，用 HDC 语言模型现场生成一句**非检索**的话（计数条件生成、零梯度）。

        守卫（任一不过 → None，调用方回退到引文）：① 4≤len≤24 且以句号收束；② 三元命中率≥0.5
        （=语料里见过，非乱码）；③ 无经验谓词（不冒充亲历）；④ 无引号/花括号等装饰符；
        ⑤ **新颖**（该句 ∉ 已学句集——即"由已见片段重组"的新句，而非逐字复述记忆）。产物入
        `清洁引用源`（归属=该证据段来源，可溯源）。"""
        if not getattr(self.lm_d, "计数1", None):
            return None
        种子 = q[-4:] if len(q) >= 4 else q
        for 尝试 in range(4):
            rng = np.random.default_rng((self.seed, self.拍, 23, 尝试))
            s = self.lm_d.生成(种子=种子, 温度=1.0, rng=rng, 最大长=20)
            if (4 <= len(s) <= 24 and s.endswith("。")
                    and not any(p in s for p in 经验谓词)
                    and self.lm_d.三元命中率(s) >= 0.5
                    and not any(ch in s for ch in "『』《》【】—…、；：？！?!,（）()")
                    and ('\u4e00' <= s[0] <= '\u9fff')
                    and (s.strip() not in self.lm_d.句集)):     # 生成（重组）而非逐字复述
                净 = s.strip()
                self.清洁引用源[净] = self.段源.get(段, "用户")
                return 净
        return None

    # ---------- 对话流机制（#251：条件生成+上下文记忆，计数实现，零梯度） ----------
    def _条件开场(self, q: str):
        """以问题尾部为条件从对话流模型采样续写（Transformer 式条件生成的计数实现）。
        五重守卫不过→None→回退连接语生成。"""
        if not self.lm_d.计数1:
            return None                                    # 冷对话流：无条件统计
        种子 = q[-6:] if len(q) >= 6 else q
        τ = float(0.7 + 0.5 * (self.ñ if self.ñ is not None else 1.0))
        for 尝试 in range(3):
            rng = np.random.default_rng((self.seed, self.拍, 11, 尝试))
            s = self.lm_d.生成(种子=种子, 温度=τ, rng=rng, 最大长=18)
            if (4 <= len(s) <= 20 and s.endswith("。")
                    and not any(p in s for p in 经验谓词)
                    and self.lm_d.三元命中率(s) >= 0.5
                    and not any(ch in s for ch in "『』《》【】—…、；：？！?!,（）()")
                    and ('\u4e00' <= s[0] <= '\u9fff')
                    and self._共二字(s, q)):          # ★ 必须与问题有 2-gram 重合（否则是采样碎片）
                return s
        return None

    def _上下文检索(self, q: str):
        """上下文记忆窗口检索：返回**本窗口**最近 6 轮内与问题最相关的用户原话或 None。
        （多窗口：只看 self._当前会话 的上下文——A 窗口的"刚才"不会串到 B 窗口。）"""
        q集 = _关键词集(q)
        best = None
        for e in reversed(self.对话流):
            if e[0] != "你" or e[2] != self._当前会话:
                continue
            文本 = e[1]
            t集 = _关键词集(文本)
            if not t集:
                continue
            分 = len(q集 & t集) / max(min(len(q集), len(t集)), 1)
            if best is None or 分 > best[1]:
                best = (文本, 分)
        return best if (best and best[1] >= 0.35) else None

    def _行动知识回答(self, i: int) -> str:
        依据 = [k for k, (t, x) in enumerate(self.账本)
                if t == "检验" and x.startswith("动作%d " % i)]
        if i not in self.束 or i >= self.动作数 or not 依据:
            self._暂存依据 = {len(self.账本) - 1}
            return "动作 %d 我还没试过——账本里没有它的检验记录，我不想瞎猜。" % i
        束 = np.sign(self.束[i])
        序 = np.argsort(-np.abs(束))[:3]
        方向 = "、".join("第%d维往%s" % (d + 1, "正" if 束[d] > 0 else "负") for d in 序)
        self._暂存依据 = set(依据[-3:])
        return ("我自己试出来的——动作 %d 之后世界往这些方向：%s。这不是谁教我的。" %
                (i, 方向))

    def _溯源报告(self, rep: dict) -> str:
        自言s = [e for e in self.自言.流 if e["号"] in rep["支撑自言"]][:4]
        段s = ["自言#%d（%s，第%d拍）：%s" % (e["号"], e["类型"], e["拍"], e["文本"][:28])
               for e in 自言s]
        段s += ["账本#%d（%s）：%s" % (i, self.账本[i][0], self.账本[i][1][:28])
                for i in rep["支撑账本"][-6:]]          # 末 6 条含本轮接收条目（溯源主据）
        return ("回复 %d 是这么来的——%s。这些条目我逐条核对过，账本里都在。" %
                (rep["号"], "；".join(段s)))

    # ---------- 通用输入：向量直驱（Phase2；任一模态统一到 D 维向量空间） ----------
    def 步_向量(self, 向量, 原文: str = None, 谁: str = None, 会话: str = None) -> dict:
        """以一组 D 维向量直接驱引擎（不经 codec.编码(文本)）。

        用途：全接口预处理的下游——文本/音频特征/任意特征经统一输入层转成同一 D 维向量后，
        都可走此入口驱对话。原文(可选)只作「语词标签」：给标签则按语词路由（问句→作答/
        陈述→注册），无标签则如实报告「收到一段没有语词的感觉」。确定性：同 seed + 同向量
        = 逐位一致；事件史记 "步向量" 供升维/载入重放。
        多窗口：会话= 本次输入所属窗口（默认沿用上一拍）。记忆共享、上下文按窗口隔离。"""
        if 谁 is not None:
            self.说话人 = str(谁)
        if 会话 is not None:
            self._当前会话 = str(会话)
        v = np.asarray(向量, dtype=np.float32).ravel()
        # 维度适配（升维/换维后事件重放：旧维向量按当前 D 补零/截断 → 确定性、可复现）
        D = int(self.探针.shape[1])
        if v.shape[0] < D:
            v = np.concatenate([v, np.zeros(D - v.shape[0], dtype=np.float32)])
        elif v.shape[0] > D:
            v = v[:D]
        self.事件史.append(("步向量", 原文, self.说话人,
                            [float(x) for x in v], self._当前会话))
        u = self.感知向量(v)
        rec = self._引擎步(u)
        self._吃(rec)
        标签 = 原文 if 原文 else "（一段没有语词的感觉：%d 维）" % v.shape[0]
        if 原文 and 是问句(原文):
            响应 = self._自报告(原文) if 是自指(原文) else self._答问(原文)
        elif 原文:
            if self.注册去重(原文):
                self.段源[原文] = "用户"
            self.段说话人.setdefault(原文, self.说话人)
            if 是请求(原文):
                响应 = self._自报告(原文)                       # 祈使/请求句→自报告（不落"记下了"）
            else:
                响应 = self._应答表面(原文, "记下了：『%s』（记忆共 %d 段，能量 %.2f）。" %
                                      (原文, len(self.codec._段), self.e))
        else:
            响应 = ("（我收到一段没有语词的感觉——%d 维，能量 %.2f，新异 %.2f。"
                    "它进了我的记忆，但我还叫不出它的名字。）" %
                    (v.shape[0], self.e if self.e is not None else -1.0,
                     self.ñ if self.ñ is not None else 0.0))
        self.lm_d.学句(标签 + "。" + 响应)
        self.对话流.append(("你", 标签, self._当前会话))
        self.对话流.append(("我", 响应, self._当前会话))
        self.回复台账.append(dict(号=len(self.回复台账) + 1, 问=标签, 响应=响应,
                                  支撑自言=[], 支撑账本=[]))
        self.自审承载()          # Phase3·自发：收到一段感觉后，同样由它自己掂量承载（向量入口同律）
        return dict(响应=响应, e=self.e, ñ=self.ñ, 模态="向量",
                    回复号=len(self.回复台账))

    def 步(self, 用户文本: str, 谁: str = None, 会话: str = None) -> dict:
        if 谁 is not None:
            self.说话人 = str(谁)
        if 会话 is not None:
            self._当前会话 = str(会话)                         # 多窗口：本拍所属窗口
        self.事件史.append(("步", 用户文本, self.说话人, self._当前会话))
        # —— #270 延答内驱：证据不足的问题→思考态（自发：他自己的不确定信号） ——
        即时词 = ("累", "名字", "是谁", "聊聊", "干嘛", "看什么", "怎么决定",
                 "学到", "探索", "怎么样", "喜欢", "讨厌", "想什么")
        if (self._声道 == "用户" and 是问句(用户文本)
                and "为什么这么回答" not in 用户文本
                and not any(w in 用户文本 for w in 即时词)
                and 记忆核心(用户文本) is None            # 回忆问句→检索作答，不进延答
                and self.思考中 is None):
            h = self._检索(用户文本, k=1)
            s1 = h[0][1] if h else 0.0
            if s1 < self.充分阈值:
                return self._延答(用户文本)
        元问题 = (("为什么这么回答" in 用户文本) or
                  ("为什么" in 用户文本 and "这么" in 用户文本)) \
            and bool(self.回复台账) and self._声道 == "用户"
        r = super().步(用户文本)
        # —— 多账号共用记忆池：记录本条用户陈述的说话人（供检索隔离） ——
        if not 是问句(用户文本) and 用户文本 in self.codec._段:
            self.段说话人[用户文本] = self.说话人
        if not 元问题:                                          # 表达层：模板/复述→它自己的话
            if 是请求(用户文本) and "记下了：『" in str(r["响应"]):
                r["响应"] = self._自报告(用户文本)               # 祈使/请求句→自报告（不落"记下了"）
            else:
                r["响应"] = self._应答表面(用户文本, r["响应"])
        if 元问题:                                              # 元问题接口：溯源上一回复
            rep = self.回复台账[-1]
            r["响应"] = self._溯源报告(rep)
            self.账本[-1] = ("行动", _引号外文本(r["响应"]))     # 行动记录与实际输出一致
        接收号 = len(self.账本) - 2                             # v6.步 先记 接收 后记 行动
        # —— 外部触发自言（复盘）+ 支撑捕获 ——
        自言号 = self._自言("复盘",
                            "你刚说了『%s…』。我是这么答的：%s（回复 %d；第 %d 拍。）" %
                            (用户文本[:10], r["响应"][:60], len(self.回复台账) + 1,
                             self.拍),
                            ("外部", 接收号), [接收号])
        支撑账本 = {接收号} | set(self._暂存依据)
        self._暂存依据 = set()
        for 段 in re.findall(r"『([^』]+)』", r["响应"]):        # 引用段 → 账本条目
            if len(段) < 2:                                     # 单字查证引文（v6 口径）
                continue                                        # 不作支撑（防噪声泛匹配）
            for k, (t, x) in enumerate(self.账本):
                if x == 段 or (段 and 段 in x):
                    支撑账本.add(k)
        支撑自言 = [自言号]
        for e in self.自言.流[:-1]:                             # 相关旧自言（引用同一账本条目）
            if any(号 in 支撑账本 for (号, _) in e["引用"]):
                支撑自言.append(e["号"])
                if len(支撑自言) >= 5:
                    break
        # —— 对话流机制（#251）：跨轮条件学习 + 上下文窗口 ——
        # （本拍生成发生在 super().步 内、学习在之后——条件只来自历史轮，不自引）
        self.lm_d.学句(用户文本 + "。" + r["响应"])
        self.对话流.append(("你", 用户文本, self._当前会话))
        self.对话流.append(("我", r["响应"], self._当前会话))
        self.回复台账.append(dict(号=len(self.回复台账) + 1, 问=用户文本,
                                  响应=r["响应"], 支撑自言=支撑自言,
                                  支撑账本=sorted(支撑账本)))
        r["回复号"] = len(self.回复台账)
        self.自审承载()          # Phase3·自发：这一句学完，自己掂量「装得下吗」（自愿，非外部强推）
        return r

    def 自主拍(self, n: int = 5) -> int:
        self.事件史.append(("自主", int(n)))
        return super().自主拍(n)

    # ---------- 看门狗心跳（V6：status.json 单向数据通道；与读方并发→原子替换重试） ----------
    def 写心跳(self, 路径: str) -> None:
        import json
        import time as _t
        tmp = 路径 + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(拍号=self.拍, 引导拍=self.引导拍, κ=self.私偏κ,
                           最近动作=list(self.动作史[-100:])), f, ensure_ascii=False)
        for _ in range(5):                                   # 读方偶发占用句柄→短重试
            try:
                os.replace(tmp, 路径)
                return
            except PermissionError:
                _t.sleep(0.02)
        try:
            os.replace(tmp, 路径)
        except PermissionError:
            pass                                             # 下一拍重写，心跳不断流

    # ---------- 持久化（事件重放式；同 seed+同事件序列=逐位再生） ----------
    def 保存(self, 路径: str) -> None:
        import pickle as _p
        import lang_codec as _lc                       # Phase3：维度入档（生长后的 D 可跨重启恢复）
        导 = self.导航器
        with open(路径, "wb") as f:
            _p.dump(dict(seed=self.seed, 私偏κ=self.私偏κ,
                         D=int(_lc.D), 教史=list(self.教史), 事件=list(self.事件史),
                         世界seed=self.世界seed, 动作数=self.动作数,
                         书=self.书, 文件界=self.文件界,
                         开放根=(list(导.根s) if 导 is not None else None),
                         开放屏蔽=(list(导.屏蔽) if 导 is not None else None)), f)

    @classmethod
    def 载入(cls, 路径: str, 维度=None) -> "综合体":
        import pickle as _p
        import lang_codec as _lc
        with open(路径, "rb") as f:
            d = _p.load(f)
        # Phase3：恢复维度——显式 维度 优先（升维用），否则取档内 D（缺省=现值，向后兼容）。
        _lc.D = int(维度) if 维度 is not None else int(d.get("D", _lc.D))
        桥 = cls(seed=d["seed"], 私偏κ=d.get("私偏κ", 10.0))
        桥.动作数 = d.get("动作数", 8)
        from dialogue_v4 import 连接语料
        from dialogue_v5 import 自我澄清语料
        教seq = [tuple(x) for x in d.get("教史") or []]
        for e in 教seq:
            if e[0] in (连接语料, 自我澄清语料):
                continue                               # __init__ 链已教，跳过防 ×2（v5 同律）
            桥._声道 = e[2] if len(e) > 2 else "用户"
            桥.教_v2(e[0], epochs=int(e[1]), 记录=False)
        桥._声道 = "用户"
        if d.get("世界seed") is not None:
            桥.设世界(隐藏世界(seed=d["世界seed"], 动作数=d["动作数"]))
        if d.get("书"):
            桥.设教材(d["书"], d.get("文件界"))
        if d.get("开放根"):                                    # 开放域根入档 → 重放断供不炸
            桥.设开放域(list(d["开放根"]), list(d.get("开放屏蔽") or []))
        # Phase3·自发：重放=按原事件序列重建记忆/账本/自言，**不是**新的学习——
        # 期间关掉承载自审（否则会在重建途中又抛扩容请求，违背「决定只发生在其真实节律里」）。
        桥._重放中 = True
        for ev in d["事件"]:
            if ev[0] == "步":
                # 旧存档 ("步",文,谁) / 多窗口 ("步",文,谁,会话)——向后兼容（缺会话=None）
                桥.步(ev[1], ev[2] if len(ev) > 2 else None,
                     ev[3] if len(ev) > 3 else None)
            elif ev[0] == "步向量":                        # Phase2：向量直驱事件重放
                桥.步_向量(np.asarray(ev[3], dtype=np.float32),
                          ev[1], ev[2] if len(ev) > 2 else None,
                          ev[4] if len(ev) > 4 else None)
            elif ev[0] == "自主":
                桥.自主拍(int(ev[1]))
            elif ev[0] == "断供":
                桥.断供一拍()
        桥._重放中 = False
        return 桥
