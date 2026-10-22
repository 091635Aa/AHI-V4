# -*- coding: utf-8 -*-
"""meeting38/器官_脑.py —— AHI 脑：连续思维循环（边干边想，不等手）

━━━ 红线一（架构）━━━
本器官只读写共享工作区，不调用、不等待任何其他器官：
  · 每 ~50ms 一个"思维拍"，读视觉/听觉/动作/语义的**最新值**；
  · 手在动的时候它照样在想（**它不检查"手是否忙"来决定自己能不能思考**）——
    这与上一版"一个动作没结算完整条思维线程 return"的单向流水线正相反；
  · 它只在"要不要写下一个意图"这一步看手的状态，而"想"这一步永不阻塞。

━━━ 红线二（非 Transformer）━━━
决策不用任何神经网络/attention/梯度。用的是：
  · HDC 双极码（借 `态.编码`）做元素新异性/相似度；
  · 计数经验（尝试/成败）做选择——"乱动→看结果→记账→按经验选"。
这是全自主的、与 Transformer 无关的决策机制。

职责：
  1. 归因：手点完一下，比较"点击前/后"的元素流，把世界差分交给语义层学 affordance；
  2. 注意：按新异性/变化挑焦点元素（写回视觉槽的焦点）；含躯体级安全闸
     （角区回避＋钱包反射；禁 OCR 后危词事实源改系统自报，见 安全事实.py）；
  3. 出招：手闲时把"下一个意图"写进意图槽（探索未碰过的元素，轮换左/右/中键/滚轮）；
  4. 发声：周期性把自身状态写进表达槽（嘴去说）——想与说同时进行；
  5. 思考链（步3.6，所有者 2026-10-05）：每拍把实义思考（看/想/招/预/果/路）写进
     共享工作区「思考槽」——"看得见的思考过程"由脑自己的循环原生给出（非工具调用）。
"""
from __future__ import annotations

import threading
import time

import numpy as np


def _句柄签名(元素s):
    """元素流的轻量签名（用于差分）——只用句柄+字+心，全整数，可逐位复现。"""
    out = []
    for e in 元素s:
        心 = e.get("心") or (0, 0)
        out.append((str(e.get("句柄")), str(e.get("字", ""))[:16],
                    int(心[0]) // 8, int(心[1]) // 8))
    out.sort()
    return out


class 脑器:
    def __init__(self, 态, 语义图=None, 台账=None, 种子: int = 3802,
                 允许键=("左", "左", "左", "右", "中"), 允许滚=(0, 0, 0, 0, -3, 3),
                 安全探=None, 允许打字=(), 允许拖=False, 允许双=False, 允许回报=False,
                 档案=None):
        self.态 = 态
        self.语义 = 语义图
        self.账 = 台账
        # V3 模块二：功能档案（操作干净化）。默认 None ⇒ _出招 路径零改动（C14）。
        self.档案 = 档案
        self.rng = np.random.default_rng(int(种子))
        self.允许键 = tuple(允许键)
        self.允许滚 = tuple(允许滚)
        # 步5 教会用电脑：键入/拖拽 也进探索字母表（像婴儿挥四肢——不是写死任务流程）。
        # 默认 ()/False：_出招 路径**零额外 rng 抽取** → 既有行为逐位不变（C14/C13）。
        self.允许打字 = tuple(允许打字)      # 非空时偶发「键入」招（内容随机取自本元组）
        self.允许拖 = bool(允许拖)           # True 时偶发「拖拽」招（把元素拖去邻近处）
        # 步7 目的性（默认全关 → 既有行为逐位不变）；步8：颜色桶退役，身份=图标簿（形状）：
        self.允许双 = bool(允许双)           # True 时 识当前/目标命中 → 「双击=运行它」
        self.允许回报 = bool(允许回报)       # True 时 多巴胺（RPE 式）+ 图标簿登记
        self.簿 = {}                         # 图标簿：编号 → dict(束,心,名称,成功,大变)
        self._凹视史 = {}                    # 步8：句柄 → (形状束, t)——眼看过谁就记谁
        self.多巴胺总 = 0.0                  # 累计分泌量（遥测）
        self.达成数 = 0                      # 目标达成次数（找到→+3 的那次）
        # 步8 形状描述子（色无关结构统计；识别门：同图标(含抖动)≥0.85、异图标≤0.65 之间）
        try:
            from 形状束 import 形状描述子
            self.形状描 = 形状描述子(态.编码)
        except Exception:
            self.形状描 = None
        self._已试 = {}                     # 句柄 → 尝试次数
        self._待归因 = None                 # dict(句柄, 键, 前签, 前元素s, t, 版本)
        self._待发意图 = None               # 已预置、待手消费的下一招（连招队列）
        self._连招预备 = 0
        self._上次动作计数 = 0
        self._上次放弃计数 = 0               # 手「放弃」单调计数（与动作计数同律：差值检测，免竞态）
        self._上次说 = 0.0
        self._上次看 = 0.0                   # 「看」叙述节流（步3.6 思考链）
        self._上次安全 = 0.0                 # 安全闸叙述节流
        self._最近结果 = []
        self.思维拍 = 0
        self._安全拒拦 = False               # 本拍是否被安全闸拦下（防误报"眼前没东西"）
        self._安全统计 = dict(整窗回避=0, 控件危词=0)
        self._安全探 = 安全探               # 注入接口（测试）；默认=系统自报事实源
        if self._安全探 is None:
            try:
                import 安全事实
                self._安全探 = 安全事实.默认源()
            except Exception:
                self._安全探 = None          # 事实源不可用 → 安全闸退化（与旧版 OCR 缺失同界）

    def _记台账(self, 文, **kw):
        if self.账 is not None:
            self.账.记("脑", 文=文, **kw)

    # ── 0. 思考链（原生；所有者 2026-10-05「要看得见思考过程」）──
    # 每拍在自己的循环里把**实义思考**写进共享工作区「思考槽」：看（所见）→ 想（判断）
    # → 招（动作）→ 预（预测）→ 果（世界答复）→ 路（下一步路径）。不是工具调用、
    # 不是外挂模型、不含 Transformer——与本器官其余通信同一范式（只写槽位，不调用任何器官）。
    #
    # ★ 分层（所有者 2026-10-06「深度思考我们是看不到的，只能看『我应该点哪里』这种」）：
    #   「看」（感知汇总）是**内部回响** → 层="内"（只留档，不外显）；
    #   「想/招/预/果/路」（我要干什么/该点哪里/点了会怎样/下一步）是**朝外正文** → 层="外"。
    #   演播台、文本流取 取思考(只要外=True)（默认）→ 内层不外显。
    def _思(self, 类: str, 文: str, 层: str = "外"):
        try:
            self.态.写思考(类=类, 文=文, 层=层)
        except Exception:
            pass

    def _叙述看(self, 视, 动):
        """节流 0.8s 的「看」：把当前所见/注视/意外/手状态讲成一句正文（**内层回响，不外显**）。"""
        t = time.time()
        if t - self._上次看 < 0.8:
            return
        self._上次看 = t
        try:
            场 = self.态.取视场()
        except Exception:
            场 = {}
        注 = 场.get("注视") or (0, 0, 0)
        停 = int(注[2]) if len(注) > 2 else 0
        元s = 视.get("元素s", [])
        未碰 = sum(1 for e in 元s if e.get("句柄") not in self._已试)
        self._思("看", "眼前 %d 件东西；注视格(%d,%d) 停留 %d 拍；意外度 %.2f；手 %s；其中没碰过的 %d 件。"
                 % (len(元s), int(注[0]), int(注[1]), 停,
                    float(场.get("意外度") or 0.0), str(动.get("状态", "闲")), 未碰),
                 层="内")

    def _预测(self, 句柄) -> str:
        """预：基于自身经验（计数/上回效果）给出对结果的预期——经验说话，不编造。"""
        近 = [r for r in self._最近结果 if r.get("句柄") == 句柄]
        if 近:
            r = 近[-1]
            return "预想：上回点它结果「%s」（变化率 %.2f），这回若还是同类控件应会再变。" \
                   % (r.get("效果", "?"), float(r.get("率", 0.0)))
        return "预想：还没碰过它——若它是有功能的控件，画面会变；若毫无反应，说明它只是块背景，我记下来。"

    # ── 1. 归因：手点完 → 世界差分 → 语义层学 affordance ──
    def _归因(self, 视, 动):
        # 放弃信号（目标消失/异常复位）→ 立刻释放连招位与待归因（防死锁）
        # 注：动作槽「反馈」是**滚动列表**（共存态.写动作 用 append 累积），旧版
        #     str(反馈).startswith("放弃") 因把列表转字符串而恒假（潜伏 bug，步3 修）。
        #     改用手的「放弃计数」单调计数做差值检测——与 动作计数 同律，免竞态、免陈旧误判。
        弃数 = int(动.get("放弃计数", 0))
        if 弃数 != self._上次放弃计数:
            self._上次放弃计数 = 弃数
            if self._待发意图 is not None:
                self._待发意图 = None
                if self._待归因 is not None:
                    self._记台账("招数被放弃", 目标=self._待归因.get("句柄"))
                    self._思("果", "招数被放弃：目标在飞行中消失了，我重新挑。")
                    self._待归因 = None
        次数 = int(动.get("动作计数", 0))
        if 次数 <= self._上次动作计数:
            # 超时看门狗：出招后 8s 手仍闲且无结果 → 视为招数丢失，释放（旧版死锁点）
            if (self._待归因 is not None and 动.get("状态") == "闲"
                    and time.time() - self._待归因["t"] > 8.0):
                self._记台账("招数超时", 目标=self._待归因.get("句柄"))
                self._思("果", "招数超时：手 8 秒没动静，我重出下一招。")
                self._待归因 = None
                self._待发意图 = None
            return
        self._上次动作计数 = 次数
        self._待发意图 = None                               # 手已消费上招，腾出连招位
        if self._待归因 is None:
            return
        待 = self._待归因
        self._待归因 = None
        后签 = _句柄签名(视.get("元素s", []))
        前签 = 待["前签"]
        变化 = self.变化率(前签, 后签)
        效果 = ("大变" if 变化 >= 0.20 else "微变" if 变化 >= 0.05 else "无感")
        # 步5 配套：归因 → 事件槽 → 我器入库（meeting38 五专家 T5 的既定契约，
        # 生产侧一直未接线 → 事件槽饥饿、长时库 0 增）。纯追加写，不改任何决策。
        try:
            self.态.投事件(dict(类="归因",
                                文本="%s %s→%s(变化%.2f)" % (待.get("键", "?"), 待["句柄"], 效果, 变化),
                                拍=self.思维拍, 源="脑"))
        except Exception:
            pass
        # 步8 多巴胺 v2（RPE 预测误差式，允许回报 开时）+ 图标簿登记：
        #   预期按**被动作元素**判定（凹视史按句柄取该元素的形状束，与图标簿比对）——
        #   不是"眼睛正看着什么"（目光被目标锁住时，手在别处探索会被误判预期落空）。
        #   实际大变而预期 → +0.4（确认）；大变而意外 → +1.0（发现）；
        #   预期大变却无感 → −0.5（预测落空）；达成（在被指认处成功运行被命名的东西）→ +3。
        #   同形状成功运行 ≥2 次获「直达资格」（教两三次就会）。关时零改动。
        史 = self._凹视史.get(str(待.get("句柄")))
        束被 = 史[0] if (史 and time.time() - 史[1] < 3.0) else None
        条号待 = 待.get("条号")
        if 条号待 is not None and self.簿.get(条号待) is not None:
            条 = self.簿[条号待]                            # 直达招：出招时已认出条目，按号直记
        else:
            条 = self._簿识束(束被)
        预期 = bool(条 is not None and 条.get("成功", 0) >= 1)
        心元 = None
        for e in 待.get("前元素s", []):
            if e.get("句柄") == 待.get("句柄"):
                心元 = e.get("心")
                主 = e.get("主色")
                break
        if 心元 is None and 待.get("心") is not None:
            心元 = tuple(待.get("心"))                      # 直达招：位置=直达落点
        达成 = False
        try:
            目标 = self.态.取目标()
        except Exception:
            目标 = {}
        名 = str(目标.get("名称") or "")
        位锚 = (目标.get("心") is not None and 心元 is not None
                and float(np.hypot(float(心元[0]) - float(目标["心"][0]),
                                   float(心元[1]) - float(目标["心"][1]))) < 120.0)
        if self.允许回报:
            r = 0.0
            if 效果 == "大变":
                r = 0.4 if 预期 else 1.0
            elif 效果 == "无感" and 预期:
                r = -0.5
            if 效果 == "大变" and 心元 is not None:
                if 条 is not None:
                    条["大变"] = 条.get("大变", 0) + 1
                    if 位锚:
                        条["心"] = tuple(心元)               # 心只随**被指认处**的实证更新（防漂移）
                    if 名 and 条.get("名称") == 名 and 位锚:
                        条["成功"] = 条.get("成功", 0) + 1
                        达成 = True
                    elif 名 and not 条.get("名称") and 位锚:
                        条["名称"] = 名                     # 已认得的形状获得命名（须在被指认处）
                elif 名 and 束被 is not None and 位锚:      # 只立教学命名的档且须在被指认处
                    self.簿[len(self.簿)] = dict(束=束被, 心=tuple(心元), 名称=名,
                                                 成功=1, 大变=1)
                    达成 = True                             # 首次成功运行了被命名的东西
                if 达成:
                    r += 3.0                                # 找到了！多巴胺大分泌
                    self.达成数 += 1
            self.多巴胺总 += r
            try:
                self.态.投事件(dict(类="多巴胺",
                                    文本="%+.1f %s %s→%s%s" % (r, 待.get("键", "?"),
                                                              待.get("句柄"), 效果,
                                                              "（目标达成）" if 达成 else ""),
                                    拍=self.思维拍, 源="脑"))
            except Exception:
                pass
            if 达成 and time.time() - self._上次说 > 5.0:
                self._上次说 = time.time()
                self.态.排队说("找到了！就是它——这个形状我记得住。")
        self._思("果", "结果：点在 %s 上，世界「%s」（变化率 %.2f）。" % (待["句柄"], 效果, 变化))
        self._思("路", "下一步：%s"
                 % ("先回看新冒出来的东西，再挑没碰过的。" if 效果 == "大变"
                    else "继续按「没碰过的优先」挑下一个。" if 效果 == "微变"
                    else "它大概率不是控件，换个没碰过的继续。"))
        # 世界差分交给语义层：功能签名 = HDC 束(新增码 ⊕ 消失码 ⊕ 文字变码)
        try:
            if self.语义 is not None:
                self.语义.记点击(待["句柄"], 待["前元素s"], 视.get("元素s", []),
                                 真实命中句柄=待["句柄"])
        except Exception:
            pass
        # ── V3 模块二 真接线（审计 P1-2 修复）：亲历点击 → 功能档案 learns 标签 ──
        #   原先档案的学习入口 `记点击()` 在 meeting38 内**只有 test_模块二.py 自己调用**，
        #   器官_脑 亲历点击时只写既有 `语义.记点击` ⇒ 档案里元素永远是「未知」，`挑()` 只能返回「看」。
        #   `危险词` 交给档案自己的 `_危险词` 规则去判（不把「效果」冒充危险词）。
        #   整块带 try + 档案存在判断：`档案=None`（默认）时**零开销、行为逐位不变**（C14）。
        try:
            if self.档案 is not None:
                self.档案.记点击(待["句柄"], 待.get("前元素s"), 视.get("元素s", []),
                                 真实命中=待["句柄"])
        except Exception:
            pass
        self._已试[待["句柄"]] = self._已试.get(待["句柄"], 0) + 1
        self._最近结果.append(dict(句柄=待["句柄"], 键=待["键"], 效果=效果,
                                  率=round(变化, 3)))
        self._记台账("归因", 目标=待["句柄"], 键=待["键"], 效果=效果,
                     率=round(变化, 3))

    @staticmethod
    def 变化率(前签, 后签) -> float:
        a, b = set(前签), set(后签)
        if not a and not b:
            return 0.0
        return len(a ^ b) / max(len(a | b), 1)

    # ── 2. 注意：挑焦点（新异性最高的元素）──
    # 实机安全闸（躯体级脊髓反射，非语义）——步3 配套修复（安全回归，专家02独有发现）：
    #   禁 OCR（C9）后 元素s[*].字 恒空，危词表不能再读字面；改由**系统自报事实源**驱动：
    #   ① 整窗回避——前景窗口标题含危词 → 本拍不出任何招（钱包/破坏语境整屏不碰）；
    #   ② 钱包反射——候选落点处的 UIA 控件名含危词 → 跳过该候选（最多核实 K 个候选，控成本）。
    #   事实源：安全事实.py（win32gui 窗口标题 + comtypes UIA 控件名；缓存/熔断/优雅退化）。
    #   角区回避——屏幕右上角（最大化窗的关闭钮一带）不落点（原样保留）。
    危词 = ("支付", "付款", "购买", "充值", "续费", "会员", "开通", "订单", "交易",
            "余额", "钱包", "转账", "花呗", "借贷", "￥", "¥", "价格", "付费",
            "删除", "卸载", "格式化", "关机", "重启", "注销", "清空", "销毁")
    角区 = 220          # 屏幕右上角横带（关闭/最小化钮区）
    角区高 = 80
    安全探量 = 3        # 每次出招最多用事实源核实几个候选（保成本有界）

    @classmethod
    def _危词命中(cls, 文) -> str:
        s = str(文 or "")
        for z in cls.危词:
            if z in s:
                return z
        return ""

    def _探窗口(self) -> str:
        if self._安全探 is None:
            return ""
        try:
            return str(self._安全探.窗口标题() or "")
        except Exception:
            return ""

    def _探控件(self, 点) -> str:
        if self._安全探 is None:
            return ""
        try:
            return str(self._安全探.点控件名((int(点[0]), int(点[1]))) or "")
        except Exception:
            return ""

    # ── 步7 目标命中判据（色锚 + 位锚双通道）：教学指认给 主色+心 两个锚 ──
    @staticmethod
    def _目标接近(目标, 主, 心):
        """色近（Σ|Δrgb|<90）或 位近（心距<90px）。真机上 120px 粒度下图标主色会被壁纸
        混色——位锚（移动性导入：「我要看那里」）保证指认仍然可命中。"""
        if not isinstance(目标, dict):
            return False
        目色, 目心 = 目标.get("主色"), 目标.get("心")
        if 目色 is not None and isinstance(主, (tuple, list)) and len(主) == 3:
            if sum(abs(float(a) - float(b)) for a, b in zip(主, 目色)) < 90.0:
                return True
        if 目心 is not None and 心 is not None:
            if float(np.hypot(float(心[0]) - float(目心[0]),
                              float(心[1]) - float(目心[1]))) < 90.0:
                return True
        return False

    def _注意(self, 视):
        元素s = 视.get("元素s", [])
        self._安全拒拦 = False
        if not 元素s:
            return None
        屏 = 视.get("屏幕") or (1920, 1080)
        # ① 整窗回避（快通道：win32gui 窗口标题）
        题 = self._探窗口()
        z = self._危词命中(题)
        if z:
            self._安全拒拦 = True
            self._安全统计["整窗回避"] += 1
            self._记台账("安全闸-整窗回避", 窗口=题[:40], 危词=z)
            return None
        # 步7：目标槽（移动性偏置的目的端）——空槽=无目标，零影响（既有行为不变）
        try:
            目标 = self.态.取目标()
        except Exception:
            目标 = {}
        目标色 = 目标.get("主色") if isinstance(目标, dict) else None
        # ② 打分（角区/尺寸过滤；新异＋有字——字恒空后"有字"项自然为 0，保留原式；
        #    步7 回报偏置=多巴胺账本按主色桶回看（允许回报 开时才计入）；目标色近→强吸引压过新异）
        候选 = []
        for i, e in enumerate(元素s):
            框 = e.get("框") or (0, 0, 0, 0)
            w, h = 框[2], 框[3]
            if w < 14 or h < 12 or w * h > 0.6 * max(1, 屏[0] * 屏[1]):
                continue
            心 = e.get("心") or (0, 0)
            if 心[0] > 屏[0] - self.角区 and 心[1] < self.角区高:
                continue                                # 角区回避
            字 = str(e.get("字", ""))
            尝 = self._已试.get(e.get("句柄"), 0)
            新异 = 1.0 / (1.0 + 尝)                       # 没碰过的更值得碰
            有字 = 0.15 if 字 else 0.0                     # 有字的更像按钮
            分 = 新异 + 有字
            主 = e.get("主色")
            # （步8：颜色桶回报偏置退役——图标身份=图标簿的形状束，见 _识当前/直达）
            if 目标色 is not None and isinstance(主, (tuple, list)) and len(主) == 3:
                色距 = sum(abs(float(a) - float(b)) for a, b in zip(主, 目标色))
                if 色距 < 90.0:
                    分 += 3.0 * (1.0 - 色距 / 90.0)      # 我在找这个颜色（看→被吸引→靠近）
            if 目标.get("心") is not None and 心 is not None:
                位距 = float(np.hypot(float(心[0]) - float(目标["心"][0]),
                                      float(心[1]) - float(目标["心"][1])))
                if 位距 < 150.0:
                    分 += 2.0 * (1.0 - 位距 / 150.0)     # 移动性导入：我要看那里（位置锚）
            候选.append((分, i, e))
        候选.sort(key=lambda t: (-t[0], t[1]))          # 确定性：分降序，同分按元素原序
        # ③ 逐候选核实（≤K 个）：落点控件名含危词 → 跳过；全被否 → 本拍不出招
        for 分, i, e in 候选[:self.安全探量]:
            心 = e.get("心") or (0, 0)
            名 = self._探控件(心)
            z = self._危词命中(名)
            if z:
                self._安全统计["控件危词"] += 1
                self._记台账("安全闸-控件危词", 名=名[:40], 危词=z,
                             句柄=e.get("句柄"))
                continue
            self.态.置焦点(e.get("句柄"))
            return e
        if 候选:
            self._安全拒拦 = True                        # 候选全被安全闸否掉 → 不出招
        return None

    # ── 3. 出招：手闲 → 写下一个意图（不等手，但也不抢手）──
    识别门 = 0.75           # 形状识别门（组内最大相似）：同图标(抖动)0.85+ / 异图标 0.65− 之间

    def _识当前(self):
        """中央凹形状束组 vs 图标簿：返回 (条号, 条目, 相似)。束组内取最大（容忍框拟合抖动）；
        全部低于门 → (None, None, 0)。"""
        if self.形状描 is None or not self.簿:
            return None, None, 0.0
        try:
            场 = self.态.取视场()
        except Exception:
            return None, None, 0.0
        束s = [b for b in (场.get("中央凹束s") or []) if b is not None]
        if not 束s and 场.get("中央凹束") is not None:
            束s = [场.get("中央凹束")]
        if not 束s:
            return None, None, 0.0
        bestk, best, bests = None, None, self.识别门
        for k, 条 in self.簿.items():
            try:
                s = max(self.形状描.相似(b, 条["束"]) for b in 束s)
            except Exception:
                continue
            if s >= bests:
                bestk, best, bests = k, 条, s
        return (bestk, best, bests) if best is not None else (None, None, 0.0)

    def _簿识束(self, 束):
        """形状束 → 图标簿条目（组外单束匹配；低于门 → None）。"""
        if self.形状描 is None or 束 is None or not self.簿:
            return None
        best, bests = None, self.识别门
        for 条 in self.簿.values():
            try:
                s = self.形状描.相似(束, 条["束"])
            except Exception:
                continue
            if s >= bests:
                best, bests = 条, s
        return best

    def _凹视采样(self):
        """每拍刷 凹视史（步8）：眼此刻的中央凹 → 记 句柄→(束, t)。失败安全。"""
        if self.形状描 is None:
            return
        try:
            场 = self.态.取视场()
            句 = 场.get("中央凹句柄")
            束0 = next((b for b in (场.get("中央凹束s") or []) if b is not None), None)
            if 句 and 束0 is not None:
                self._凹视史[str(句)] = (束0, time.time())
        except Exception:
            pass

    def _出招(self, 视, 动, 意):
        """手闲 → 出下一招；手忙 → 仍可**预置**下一招（连招：前招未收势，后招已列队）。
        两种情形都不阻塞"想"这一步——本器官从不因手忙而停止思考。"""
        if self._待归因 is not None:
            return None                                    # 上一招还没结算，先等世界给答复
        if self._待发意图 is not None:
            return None                                    # 已预置一招，等手消费（不刷屏）
        # ── 步8 直达：图标簿认识的东西 → **直接知道在哪里**（首招即中，无漫游）──
        #   a) 眼前正是簿里成功运行 ≥2 次的形状 → 直奔其位置双击运行；
        #   b) 教学命名的目标（目标槽.名称）在簿里 → 直奔该条目位置。
        # 危词闸照常兜底（直达落点也要过控件名核查）。
        if self.允许双:
            条号, 条, _相 = self._识当前()
            直 = None
            条k = None
            if 条 is not None and 条.get("成功", 0) >= 2:
                直, 条k = 条, 条号
            else:
                try:
                    名 = str(self.态.取目标().get("名称") or "")
                except Exception:
                    名 = ""
                if 名:
                    for k, t in self.簿.items():
                        if t.get("名称") == 名:
                            直, 条k = t, k
                            break
            if 直 is None:
                # 无聊巡访（步8）：眼前没有没碰过的东西、或世界安静（无意外）时，
                # 回访簿里最熟的那件——「下一次我直接知道在哪里」的日常形态。
                未试 = [e for e in (视.get("元素s") or [])
                        if e.get("句柄") not in self._已试]
                try:
                    _意外 = float(self.态.取视场().get("意外度") or 0.0)
                except Exception:
                    _意外 = 1.0
                if not 未试 or _意外 < 0.05:
                    候 = [(t.get("成功", 0), k, t) for k, t in self.簿.items()
                          if t.get("成功", 0) >= 2]
                    if 候:
                        候.sort(key=lambda x: (-x[0], x[1]))
                        条k, 直 = 候[0][1], 候[0][2]
            if 直 is not None:
                心 = tuple(直["心"])
                if not self._危词命中(self._探控件(心)):
                    意图 = dict(目标点=心, 键="双", 兜底=心)
                    self.态.写意图(当前=意图, 来源="直达", 强度=1.0)
                    self._思("招", "我认得——%s就在 (%d,%d)，直接过去双击运行。"
                             % (("「%s」" % 直["名称"]) if 直.get("名称") else "这家伙",
                                心[0], 心[1]))
                    self._待发意图 = 意图
                    self._待归因 = dict(句柄="直达", 键="双", 条号=条k, 心=心,
                                        前签=_句柄签名(视.get("元素s", [])),
                                        前元素s=list(视.get("元素s", [])), t=time.time())
                    self._记台账("直达", 位置=心, 名称=str(直.get("名称") or ""))
                    return 意图
        焦点 = self._注意(视)
        话 = 视.get("屏幕") or (1920, 1080)
        if 焦点 is None:
            if self._安全拒拦:
                self._安全拒拦 = False                   # 安全闸拦下：不误报"眼前没数出东西"
                if time.time() - self._上次安全 > 5.0:
                    self._上次安全 = time.time()
                    self._思("想", "安全闸拦下：这屏/这个落点带危险词（支付、删除一类），本拍不出招。")
                return None
            # 没东西可碰：发一句"空"的自我陈述，保持思维与嘴的活性
            if time.time() - self._上次说 > 6.0:
                self.态.排队说("眼前没数出可碰的东西，我先看着。")
                self._思("想", "眼前没数出可碰的（元素表空），我先看着，等世界给点变化。")
                self._上次说 = time.time()
            return None
        # ── V3 模块二 接线：功能档案三层过滤（默认 None ⇒ 本块不执行，既有行为逐位不变）──
        if self.档案 is not None:
            _挑 = self.档案.挑(焦点, 视.get("元素s") or [])
            _动 = _挑.get("动作")
            if _动 is None:
                if time.time() - self._上次说 > 6.0:
                    self.态.排队说("这些我都碰过、也都指到死角上了，我换个地方看看。")
                    self._上次说 = time.time()
                return None
            if _动 == "看":
                self._思("想", "这件我没见过它的功能，先看一眼，不急着点。")
                return None
            _新焦 = _挑.get("目标")
            if isinstance(_新焦, dict) and _新焦 is not 焦点:
                焦点 = _新焦
                self._思("招", "档案说这件是「可用」的，先点它，别去碰那些干扰项。")
        句柄 = 焦点.get("句柄")
        尝 = self._已试.get(句柄, 0)
        框 = 焦点.get("框") or (0, 0, 0, 0)
        self._思("想", "盯上 %s：%s %dx%d（在注视窗里稳住的那个），已碰过 %d 次——%s"
                 % (句柄, str(焦点.get("形状", "方块")), int(框[2]), int(框[3]), 尝,
                    "新目标，先试它" if 尝 == 0 else "老对象，按经验处理"))
        if 尝 >= 3 and len([h for h, n in self._已试.items() if n == 0]) == 0:
            # 碰够了的界面：翻页试试（滚轮），制造新元素
            意图 = dict(目标点=tuple(焦点.get("心")),
                        滚动=int(self.rng.choice(self.允许滚)),
                        兜底=tuple(焦点.get("心")))
        else:
            心 = 焦点.get("心") or (0, 0)
            # （步8：颜色命中出双击退役——图标身份=图标簿形状束，双击路由归 _出招 顶部「直达」）
            键 = str(self.rng.choice(self.允许键))
            意图 = dict(句柄=句柄, 目标点=tuple(心), 键=键,
                        兜底=tuple(心))
            # 步8 教学期双击：目标槽指认在场（焦点位近被指认处）→ 用**教过的动作**（双击）试它。
            # 仅目标槽非空时进（默认零扰动）；rng 流与既有路径一致（choice 已抽，不再多抽）。
            if self.允许双 and not 意图.get("键入") and not 意图.get("拖到"):
                try:
                    _教心 = self.态.取目标().get("心")
                except Exception:
                    _教心 = None
                if _教心 is not None and 心 is not None:
                    if float(np.hypot(float(心[0]) - float(_教心[0]),
                                      float(心[1]) - float(_教心[1]))) < 90.0:
                        意图 = dict(句柄=句柄, 目标点=tuple(心), 键="双", 兜底=tuple(心))
            # 步5 教会用电脑：键入/拖 也进探索字母表（偶发替换「按键」招——探索，非写死流程）。
            # 默认全关：不进这两个分支 → 不多抽 rng → 既有行为逐位不变。
            if self.允许打字 and float(self.rng.random()) < 0.18:
                意图 = dict(句柄=句柄, 目标点=tuple(心),
                            键入=str(self.rng.choice(self.允许打字)), 兜底=tuple(心))
            elif self.允许拖 and float(self.rng.random()) < 0.18:
                dx = int(self.rng.integers(-240, 241))
                dy = int(self.rng.integers(-160, 161))
                意图 = dict(句柄=句柄, 目标点=tuple(心), 键="左",
                            拖到=(max(0, int(心[0]) + dx), max(0, int(心[1]) + dy)),
                            兜底=tuple(心))
        前签 = _句柄签名(视.get("元素s", []))
        点 = tuple(意图.get("目标点") or (0, 0))
        if 意图.get("滚动") is not None:
            self._思("招", "把光标移到 (%d,%d)，滚轮 %d 格（这块碰够了，翻页找新东西）。"
                     % (int(点[0]), int(点[1]), int(意图["滚动"])))
        elif 意图.get("键入") is not None:
            self._思("招", "把光标移到 (%d,%d)，键入「%s」（看它接不接收）。"
                     % (int(点[0]), int(点[1]), str(意图["键入"])[:12]))
        elif 意图.get("拖到") is not None:
            self._思("招", "抓住 (%d,%d) 这件，拖去 (%d,%d)（看它跟不跟着走）。"
                     % (int(点[0]), int(点[1]), int(意图["拖到"][0]), int(意图["拖到"][1])))
        elif 意图.get("键") == "双":
            self._思("招", "就是它——移过去 (%d,%d)，停一下，双击运行。" % (int(点[0]), int(点[1])))
        else:
            self._思("招", "把光标移到 (%d,%d)，按%s键。" % (int(点[0]), int(点[1]), 意图.get("键", "左")))
        self._思("预", self._预测(句柄))
        self._思("路", "路线：等手走完比世界差分——大变=它有功能，微变=可能只是被选中，无感=换下一个。")
        self.态.写意图(当前=意图, 来源="探索", 强度=1.0)
        self._待发意图 = 意图                               # 列入连招位（手忙时也算预置）
        if 动.get("状态") != "闲":
            self._连招预备 += 1
        self._待归因 = dict(句柄=句柄,
                            键=("打" if 意图.get("键入") is not None else
                                "拖" if 意图.get("拖到") is not None else
                                意图.get("键", "滚")),
                            前签=前签, 前元素s=list(视.get("元素s", [])),
                            t=time.time())
        self._记台账("出招", 目标=句柄, 字=str(焦点.get("字", ""))[:16],
                     键=意图.get("键", "滚"), 尝=尝)
        return 意图

    # ── 4. 发声：把自身状态写进表达槽（嘴去说）——想与说并行 ──
    def _发声(self, 视, 听, 动, 语):
        t = time.time()
        if t - self._上次说 < 8.0:
            return
        self._上次说 = t
        元n = len(视.get("元素s", []))
        核 = len(语.get("按钮s", {}))
        词 = len(语.get("词表", []))
        近听 = 听.get("最近主频", 0.0)
        声 = ("耳朵刚听着 %.0fHz。" % 近听) if 近听 > 40 else "耳朵静。"
        self.态.排队说("我在想：眼前 %d 件东西，碰熟了 %d 个，认识了 %d 类功能。%s"
                    % (元n, 核, 词, 声))

    # ── 一个思维拍 ──
    def 一拍(self):
        self.思维拍 += 1
        视 = self.态.取视觉()
        听 = self.态.取听觉()
        动 = self.态.取动作()
        语 = self.态.取语义()
        self._归因(视, 动)                                 # 先消化世界对上一招的答复
        self._凹视采样()                                   # 步8：刷凹视史（句柄→形状束）
        self._叙述看(视, 动)                               # 把「所见」讲成正文（节流 0.8s）
        self._出招(视, 动, self.态.取意图())                # 再出下一招（想/招/预/路入思考槽）
        self._发声(视, 听, 动, 语)                          # 想的同时说话

    # ── 连续循环（不检查手忙不忙决定自己能不能想）──
    def 循环(self, to_stop: threading.Event):
        while not to_stop.is_set():
            t0 = time.time()
            try:
                self.一拍()
            except Exception as ex:
                self._记台账("脑异常:%r" % ex)
            time.sleep(max(0.0, 0.05 - (time.time() - t0)))

    # ── 结算 ──
    def 统计(self) -> dict:
        return dict(思维拍=self.思维拍, 已碰元素=len(self._已试),
                    总尝试=sum(self._已试.values()), 连招预备=self._连招预备,
                    安全闸=dict(self._安全统计),
                    多巴胺=dict(总量=round(self.多巴胺总, 1), 达成数=self.达成数,
                                图标簿=[dict(名称=t.get("名称", ""),
                                             心=list(t.get("心") or ()),
                                             成功=t.get("成功", 0), 大变=t.get("大变", 0))
                                         for t in self.簿.values()]),
                    最近结果=list(self._最近结果[-12:]))
