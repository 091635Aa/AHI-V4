# -*- coding: utf-8 -*-
"""V3主程序.py —— AHI V3 总装（模块一~四合体）：能启动 · 能看网页 · 能说话 · 能操作。

装配（都是 2.1 已跑通的底座，V3 只加四层新模块 + meeting39 五内核，不重写引擎）：
  共存态（共享工作区）· 综合体（meeting31 synthesis，想/回忆/自言）· 嘴（纯文本降级）
  容量桥（FSW 长时记忆底图 M；★语义检索真通路见 容量桥.py 文件头）
  ＋meeting39 五内核（★ 2026-10-07 接线；见 内核宿主.py）
      自模型 ĝ（ε_self 自监督）→ 相空间算子 T（「想」= 相空间迭代，非检索）
      → 真学习（ε_self 经既有内生偏置 b[] 反哺决策）→ 自我回路（自指束 À_t + provenance 门控）
      → 顿悟涌现（相变共现取代 `if cosv>=0.8`）
  ＋V3 模块一 输出层 / 探针_思考核
  ＋V3 模块二 功能档案
  ＋V3 模块三 视觉_看网页
  ＋V3 模块四 学习_三阶段

用法：
  python V3主程序.py             # 演示：看网页 → 学 → 操作规划 → 说话（四模块 + 五内核一条龙）
  python V3主程序.py --看网页 URL # 只看一屏网页并打印控件/正文
  python V3主程序.py --刷网页 [URL...] # ★真刷真实网页：看→学→消化→说话（默认搜狐首页）
  python V3主程序.py --交互       # 纯文本对话（先想后说）
  python V3主程序.py --自主 秒数  # 自主运行（想 / 睡）

铁律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性｜C14 不破既有。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))              # .../meeting38/实施
M38 = os.path.dirname(HERE)                                    # .../meeting38
AHI3 = os.path.dirname(M38)                                    # .../AHI3
for _p in (HERE, M38, AHI3, os.path.join(AHI3, "meeting31")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(AHI3, "测试记录")


def _行(s=""):
    """安全打印（出口编码自愈）。

    ★修复（审计 P0-3 同类）：默认 GBK 控制台下，浏览器窗口标题里的**零宽字符**
    （如 Edge 标题里的 `\\u200b`）会让 `print` 抛 UnicodeEncodeError 并让进程 exit=1
    —— 判据/演示全过却被判失败。此处①重配 stdout/stderr 为 utf-8（errors=replace）；
    ②剥掉零宽与方向控制字符（它们对内容无意义，只破坏输出）。
    """
    try:
        import sys as _sys
        for _s in (_sys.stdout, _sys.stderr):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    except Exception:
        pass
    t = str(s)
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e"):
        t = t.replace(ch, "")
    try:
        print(t, flush=True)
    except Exception:
        print(t.encode("utf-8", "replace").decode("utf-8", "replace"), flush=True)


def _置前景(子串: str) -> bool:
    """把标题含 子串 的可见窗口置前景（界面识别/UIA 都需要它在前台）。"""
    try:
        import win32gui
        hits = []
        def cb(h, _):
            if win32gui.IsWindowVisible(h) and 子串 in win32gui.GetWindowText(h):
                hits.append(h)
        win32gui.EnumWindows(cb, None)
        if not hits:
            return False
        h = hits[0]
        try:
            win32gui.ShowWindow(h, 3)
        except Exception:
            win32gui.ShowWindow(h, 9)
        win32gui.SetForegroundWindow(h)
        return True
    except Exception:
        return False


def 造测试页() -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "v3_demo_page.html")
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>AHI-V3-自指引擎</title></head><body>
<nav><a href="#1">首页</a> <a href="#2">文档</a> <a href="#3">关于</a></nav>
<main>
<h1>阿希的 V3</h1>
<p>这是给自指引擎看的正文：HDC/VSA 双极超向量，不用 Transformer，不 OCR。</p>
<p>模块四把这段正文摄入 → 关联记忆 → 复盘成概念快照。</p>
</main></body></html>"""
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return p


class V3生命:
    """V3 总装：四模块 + 2.1 底座。"""

    def __init__(self, 种子: int = 310):
        self.种子 = int(种子)
        from 共存态 import 共存态
        from 容量桥 import 容量桥
        from 输出层 import 输出层
        from 探针_思考核 import 思考核探针
        from 功能档案 import 功能档案
        from 视觉_看网页 import 网页眼
        from 学习_三阶段 import 三阶段消化

        self.态 = 共存态(种子=3800, 用gpu=False)               # 共享工作区（编码器）
        self.容量 = 容量桥(种子=3800)                          # 记忆底图 M（FSW）
        self.输出 = 输出层(种子=self.种子)                     # 模块一
        self.探针 = 思考核探针(os.path.join(OUT, "探针_思考核_v3.ndjson"),
                             维=4, 种子=3800)                  # 模块一
        self.档案 = 功能档案(种子=3800, 记忆桥=self.容量)      # 模块二
        self.眼 = 网页眼(种子=3800, 编码器=self.态.编码, 记忆=self.容量)   # 模块三
        self.消化 = 三阶段消化(编码器=self.态.编码, 记忆桥=self.容量)      # 模块四
        self.桥 = None                                          # 综合体（惰性起，重）
        self._内核宿主 = None                                   # meeting39 五内核宿主
        self.记录 = {}

    def 起综合体(self, 五内核: bool = True) -> bool:
        """起综合体。`五内核=True`（默认）→ 挂 meeting39 五内核（自我意识配置）。

        这是审计 P0-2 的修复点：原先这里 `from synthesis import 综合体`，挂的是**基类**，
        五内核从未装配（`meeting38/` 对五内核引用数=0）。
        """
        try:
            if 五内核:
                import 内核宿主
                self.桥 = 内核宿主.造桥(seed=self.种子, 档案seed=self.种子)
                self._内核宿主 = 内核宿主
            else:
                from synthesis import 综合体
                self.桥 = 综合体(seed=self.种子)
                self._内核宿主 = None
            根 = [AHI3]
            try:
                self.桥.设开放域(根, 屏蔽s=("测试记录",))
            except Exception:
                pass
            return True
        except Exception as ex:
            _行("  [综合体启动失败] %r" % ex)
            self.桥 = None
            self._内核宿主 = None
            return False

    # ── 五内核只读态（给说话/证据落盘用）──
    def 内核态(self) -> dict:
        hz = getattr(self, "_内核宿主", None)
        if hz is None or self.桥 is None:
            return dict(挂载=False)
        try:
            return hz.内核态(self.桥)
        except Exception as ex:
            return dict(挂载=False, 错误=repr(ex))

    def 自省文(self) -> list:
        hz = getattr(self, "_内核宿主", None)
        if hz is None or self.桥 is None:
            return []
        try:
            return hz.自省文(self.桥)
        except Exception:
            return []

    # ── 取 S（自我状态）：真引擎遥测 + 情感因子层粗值 ──
    def 取S(self) -> dict:
        e = 新 = 债 = 饱 = None
        if self.桥 is not None:
            e = getattr(self.桥, "e", None)
            新 = getattr(self.桥, "ñ", 0.0)
            债 = getattr(self.桥, "消化债", 0.0)
            饱 = getattr(self.桥, "饱足", 1.0)
        try:
            _E, 点, 区 = self.容量.遥测情感态(能量=e, 新异=新 or 0.0,
                                            消化债=债 or 0.0, 饱足=饱 if 饱 is not None else 1.0)
        except Exception:
            点, 区 = {}, ""
        return dict(拍=getattr(self.桥, "拍", 0), e=e, ñ=新, 消化债=债, 饱足=饱, 点=点, 区=区)

    # ── 模块三：看网页 ──
    def 看网页(self, url: str = None, 等s: float = 5.0, 用本地页: bool = True) -> dict:
        """看一屏。`url=None` → 造本地演示页；给了 url → 直接开那个网址（真站点）。"""
        from 视觉_看网页 import 打开网页
        if url is None:
            页 = 造测试页()
            url = "file:///" + 页.replace("\\", "/")
            用本地页 = True
        else:
            用本地页 = bool(用本地页 and str(url).lower().endswith((".html", ".htm")))
        try:
            打开网页(url, 用edge=True, 参数=[
                "--user-data-dir=" + os.path.join(OUT, "v3_profile"),
                "--no-first-run", "--new-window",
                "--force-renderer-accessibility",
                "--window-size=1600,1000", "--window-position=20,20"])
        except Exception:
            pass
        time.sleep(float(等s))
        # 置前景：本地页用唯一标题精确命中；真站点标题未知 → 退到浏览器关键词
        标 = "AHI-V3-自指引擎" if 用本地页 else str(url).split("//")[-1].split("/")[0]
        _置前景(标) or _置前景("Edge") or _置前景("Chrome") or _置前景("浏览器")
        time.sleep(0.6)
        r = self.眼.看(存记忆=True)
        for _ in range(4):
            if len((r.get("布局") or {}).get("正文预览") or "") > 20:
                break
            time.sleep(1.5)
            r = self.眼.看(存记忆=True)
        return r

    # ── 模块四：学（对本屏正文做三阶段消化）──
    def 学(self, 正文: str) -> dict:
        句s = [s.strip() for s in 正文.replace("\n", "。").split("。") if len(s.strip()) >= 6]
        if not 句s:
            return dict(摄入=0)
        return self.消化.消化一环(句s, 来源="网页")

    # ── 模块二：操作规划（对当下控件做三层过滤，给出该点/回避/先看）──
    def 操作规划(self, 标题: str, 控件s, 记录: bool = False) -> dict:
        """`记录=True` ⇒ 把这次「点」的裁决当真写进档案（走 `记点击` 的学习入口）。

        ★审计 P1-2：原先演示里 `挑()` 被调一次就丢弃，档案里元素永远是「未知」，
        所以永远只会「裁决：看」。`记录=True` 时按裁决真的走一次「点击→世界差分→建档」，
        档案才会长出「可用/干扰项」标签 —— 这是**它自己试出来的**，不是外部灌的规则。
        """
        界 = self.档案.识界面(标题)
        # 把 UIA 控件转成档案可判的元素
        元素s = [dict(句柄="%s@%s" % (c.get("角色", "?"), (c.get("名") or "")[:16]),
                      框=c["框"], 心=(c["框"][0] + c["框"][2] // 2, c["框"][1] + c["框"][3] // 2),
                      字=(c.get("名") or ""), 主色=(120, 120, 120))
               for c in (控件s or []) if c.get("框")]
        裁 = self.档案.挑(None, 元素s)
        避 = [e for e in 元素s if self.档案.判(e) in ("干扰项", "回避")]
        节奏 = self.档案.节奏(180.0)
        记 = None
        if 记录 and 裁.get("动作") == "点" and isinstance(裁.get("目标"), dict):
            目标 = 裁["目标"]
            键 = str(目标.get("句柄") or "?")
            # 差分近似：这一件点下去后它自己从候选集合退场（其余不动）。
            # 真机点击由 器官_手 负责（V3 演示不开手，避免干扰现场）；此处只走**建档**口径。
            后 = [e for e in 元素s if e.get("句柄") != 键]
            记 = self.档案.记点击(键, 元素s, 后, 真实命中=键)
        return dict(界面=界["类型"], 界面经验=界["经验"], 元素数=len(元素s),
                    裁决=dict(动作=裁.get("动作"), 理由=裁.get("理由"),
                            目标=(裁.get("目标") or {}).get("字", "")[:20] if 裁.get("目标") else None),
                    已回避=len(避), 节奏=节奏, 建档=记,
                    档案=self.档案.快照())

    # ── 模块一：说话（只读 S / 意图 / M / 五内核内态 → 生成一句带个性的话）──
    def 说话(self, 话题: str = "", 意图: dict = None) -> dict:
        S = self.取S()
        意图 = dict(意图 or {"类别": "说"})
        M = None
        if 话题:
            try:
                M = self.容量.查经历(self.态.编码.文本码(话题))
            except Exception:
                M = None
        内 = self.内核态()
        句 = self.输出.生成正文(意图, S, M, 内态=内)
        try:
            self.态.写它说(句)
            self.态.排队说(句)
        except Exception:
            pass
        return dict(句=句, S=dict(拍=S.get("拍"), 区=S.get("区"), 效价=(S.get("点") or {}).get("V"),
                                 唤醒=(S.get("点") or {}).get("A"), 消化债=S.get("消化债")),
                    M命中=(M or {}).get("文本") if isinstance(M, dict) else None,
                    M通路=(M or {}).get("通路") if isinstance(M, dict) else None,
                    内态=内, 自省=self.自省文())

    # ── 喂一拍给综合体（五内核就在这条链上跑：自模型/思核/真学习/自我回路/顿悟）──
    def 喂(self, 文本: str, 来源: str = "自源") -> dict:
        """把一段输入喂进引擎链一拍。

        `来源="自源"` ⇒ 它自己看的（网页正文/自主），provenance 允许塑造「我」；
        `来源="外源"` ⇒ 用户说的，只进内容记忆，W_self 逐位不变（专家03 J5）。
        """
        if self.桥 is None:
            return dict(拍=None, 喂=False)
        hz = getattr(self, "_内核宿主", None)
        if hz is not None:
            hz.设来源(self.桥, 自源=(来源 == "自源"))
        try:
            r = self.桥.步_向量(self._感知向量(str(文本)), 原文=str(文本) if 来源 == "外源" else None)
            return dict(喂=True, 拍=getattr(self.桥, "拍", None),
                        响应=(r or {}).get("响应") if isinstance(r, dict) else None)
        except Exception as ex:
            return dict(喂=False, 错误=repr(ex))

    def _感知向量(self, 文本: str):
        """文本 → 16 维感觉向量（口径与 synthesis.感知向量 一致：探针投影 ×2）。

        ★不要覆盖基类的 `感知向量`（那会改掉引擎既有的语义→感觉变换，属破坏 C14：
        基类既有 `def 感知向量`，子类同名会静默覆盖、改变引擎输入标度）。
        """
        v = self.态.编码.文本码(str(文本)[:512])
        try:
            return self.桥.感知向量(v) if self.桥 is not None else v
        except Exception:
            return v

    # ── 喂一串：把一段长文本切成多片，逐片喂引擎（**连贯多拍** → 五内核对它自己看的东西留下轨迹）──
    def 喂一串(self, 文本: str, 来源: str = "自源", 最多片: int = 24,
               入记忆: bool = True) -> dict:
        """把网页正文切成句/片，**逐片连续喂**引擎链。

        为什么必须多拍：五内核对「我自己」的预测学习建立在**连续自态**上——
        只喂 1 拍则 ε_self 无上一拍可比（迹条数=1、ε=None）、思核无轨迹、顿悟核无窗口。
        多拍连喂后，自模型才有 ε_self 序列（真学习的原料），思核才有 z 轨迹，
        顿悟核才有前窗/后窗可比。

        `入记忆=True`：把这段内容作为**经历**写进记忆底图 M（带语义向量）。
        ★粒度（第二轮实测纠正）：**逐句入记忆**，而不是把整页当一条。
          实测证据：整页向量被导航词主导——「搜狐首页 vs 新闻首页」整页相似 **0.816**（全是导航），
          而「一条新闻标题 vs 整页」只有 **0.01~0.04**。学习侧关联是**句对句**比较，
          整页入库会让关联永远绑不上（真刷三站时 `新绑定=0` 的根因）。
          另外整页也留一条（话题级记忆，供「我自己看来的『搜狐』」这类引用）。
        ★注意「记忆 ≠ 学习」（所有者硬命题 / 裁定 H5）：入 M 的是**它看来的内容**（记着），
          复盘产出的**概念快照**只进 `学习迹`（变会）——两者物理分离。
        """
        if self.桥 is None:
            return dict(喂=False, 片数=0)
        片s = self._切片(文本, 最多片=最多片)
        if not 片s:
            return dict(喂=False, 片数=0)
        记 = None
        句记 = 0
        if 入记忆 and self.容量 is not None:
            try:
                记 = self.容量.记经历(拍=int(getattr(self.桥, "拍", 0) or 0), 类型="网页",
                                      文本=str(文本)[:400], 来源="网页")
            except Exception:
                记 = None
        hz = getattr(self, "_内核宿主", None)
        if hz is not None:
            hz.设来源(self.桥, 自源=(来源 == "自源"))
        n = 0
        错 = None
        for p in 片s:
            try:
                self.桥.步_向量(self._感知向量(p), 原文=None)
                n += 1
            except Exception as ex:
                错 = repr(ex)
                break
            if 入记忆 and self.容量 is not None and len(p) >= 4:
                try:
                    self.容量.记经历(拍=int(getattr(self.桥, "拍", 0) or 0), 类型="网页句",
                                      文本=p[:400], 来源="网页")
                    句记 += 1
                except Exception:
                    pass
        # ★每看完一段，让它**真的想一拍**（相空间算子 T 推进 → 思核轨迹留痕）。
        #   思考轨迹是「停下来想」的产物，没有思考态就没有轨迹（见 内核宿主.想一拍）。
        想 = False
        if n > 0 and hz is not None:
            try:
                想 = bool(hz.想一拍(self.桥, 问题=str(文本)[:80]))
                if 想:
                    self.桥.步_向量(self._感知向量(str(文本)[:200]), 原文=None)
                    n += 1
            except Exception:
                想 = False
        return dict(喂=True, 片数=n, 切出=len(片s), 想一拍=想, 记忆记录=记, 句子入记忆=句记,
                    拍=getattr(self.桥, "拍", None), 错误=错)

    @staticmethod
    def _切片(文本: str, 最多片: int = 24) -> list:
        """正文 → 句子片（标点切分，去空白/零宽，保序、确定性）。"""
        t = str(文本 or "")
        for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            t = t.replace(ch, "")
        原 = [x.strip() for x in t.replace("\r", "\n").split("\n")]
        片s = []
        for 行 in 原:
            if not 行:
                continue
            缓 = ""
            for 字 in 行:
                缓 += 字
                if 字 in "。！？!?；;":
                    if len(缓.strip()) >= 4:
                        片s.append(缓.strip())
                    缓 = ""
            if len(缓.strip()) >= 4:
                片s.append(缓.strip())
        return 片s[:max(1, int(最多片))]

    # ── ★第六层：画面（视觉想象）把它**看见的**接上，让它能"从记忆里想出画面" ──
    def 接画面(self) -> bool:
        """把「真帧来源」接给画面核（`网页眼.截图`；本文件不 import 器官，只传函数）。

        内核层（`内核_画面.py`）不 import 任何器官，所以"眼睛在哪"由这里接入——
        这是接线，不是内嵌。返回 False 表示画面层未挂载（例如五内核=False 起的桥）。
        """
        if self.桥 is None:
            return False
        hz = getattr(self, "_内核宿主", None)
        if hz is None:
            return False
        try:
            return bool(hz.设帧源(self.桥, self.眼))
        except Exception:
            return False

    def 画面一拍(self) -> dict:
        """显式看一屏 → 从记忆里读出画面（回放）→ 组合想象（把记忆里的物件搬到新位置）。"""
        hz = getattr(self, "_内核宿主", None)
        if hz is None or self.桥 is None:
            return dict(画面=False, 原因="未挂载")
        f = None
        try:
            f = self.眼.截图()                      # 真帧：真屏；没有（无 mss/无桌面）则 None
        except Exception:
            f = None
        return hz.画面一拍(self.桥, 帧=f)

    def 想象一拍(self, 位移=(3, 2)) -> dict:
        """★「脑中想出画面」：从记忆的场景里把物件整体搬家到**它没看过的新位置**，
        再按同一读出路径解出来；并把这幅想象中的画面喂给它自己的自模型（更新=False）。"""
        hz = getattr(self, "_内核宿主", None)
        if hz is None or self.桥 is None:
            return dict(想象=False, 原因="未挂载")
        return hz.想象一拍(self.桥, 位移=位移)

    def 存画面png(self, 名字: str = "画面_回放") -> dict:
        """把"它想出来的画面"落成 PNG（粗格 16×9，供人眼核对）。"""
        hm = getattr(self.桥, "画面核", None) if self.桥 is not None else None
        if hm is None or not getattr(hm, "库", None):
            return dict(存=False, 原因="画面库为空")
        os.makedirs(OUT, exist_ok=True)
        回 = hm.造png(路径=os.path.join(OUT, 名字 + "_回放.png"))
        想 = hm.造想象png(路径=os.path.join(OUT, 名字 + "_想象.png"))
        return dict(存=True, 回放=回, 想象=想, 库大小=len(hm.库))

    # ── ★真自主运行：五内核全程在链上，不需要外部输入（「独立想、独立学」）──
    def 自主运行(self, 拍数: int = 600, 休息s: float = 0.0,
                 记录间隔: int = 20, 打印: bool = True,
                 想象间隔: int = 0, 画面目录: str = None) -> dict:
        """自维持地跑 N 拍自主拍（`断供一拍`）。

        ★为什么必须走这条：V3 原来的 `--自主` 直接转发给老的 `文本态`（2.1 链），
        **五内核根本没参与**。自主拍才是「它自己驱动自己」的形态：
          每拍 → `断供一拍` → 行动/教材/文件域通道 → `_引擎步`（自模型 ε_self / 算子T /
          真学习反偏置 / 自我回路 À_t / 顿悟相变）+ **周期自评与计划自言**（它自己的话）。
        这就是「独立操作、独立做角色」的机制底座：不是外部脚本喂它，是它自己出招。
        """
        if self.桥 is None:
            return dict(自主=False, 原因="综合体未挂载")
        前 = self.内核态()
        ε轨迹, 站变化, 自言块 = [], [], []
        想象s = []
        t0 = time.time()
        for i in range(int(拍数)):
            try:
                self.桥.断供一拍()
            except Exception as ex:
                return dict(自主=False, 已跑=i, 错误=repr(ex), 前=前)
            zm = getattr(self.桥, "自模型", None)
            if zm is not None and zm.迹:
                e = zm.迹[-1].get("ε_self")
                if e is not None:
                    ε轨迹.append(round(float(e), 4))
            sx = getattr(self.桥, "思核", None)
            if sx is not None and sx.迹:
                站变化.append(self.思核站(sx))
            if 记录间隔 and (i + 1) % int(记录间隔) == 0:
                流 = getattr(getattr(self.桥, "自言", None), "流", []) or []
                for x in 流[-3:]:
                    文本 = str(getattr(x, "文本", ""))
                    if 文本.startswith(("顿悟涌现", "自我回路")):
                        自言块.append(dict(拍=getattr(x, "拍", None), 文本=文本[:110]))
                if 打印 and i and (i + 1) % (int(记录间隔) * 5) == 0:
                    _行("      自主 %d/%d 拍：ε=%s 顿悟=%s 自源塑造=%s" % (
                        i + 1, 拍数,
                        (ε轨迹[-1] if ε轨迹 else None),
                        (getattr(getattr(self.桥, "顿悟核", None), "顿悟账本", None) or []) and
                        len(self.桥.顿悟核.顿悟账本),
                        (getattr(getattr(self.桥, "自我核", None), "自源更新次数", None))))
            if 休息s:
                time.sleep(float(休息s))
            # ★画面：每「想象间隔」拍——从记忆里**想出一幅新画面**（整体搬家式组合），
            #   并把它喂给自模型（更新=False：想象不算经历）。落 PNG 供人眼核对。
            if 想象间隔 and (i + 1) % int(想象间隔) == 0:
                try:
                    im = self.想象一拍(位移=(3, 2))
                    if im.get("想象"):
                        想象s.append(dict(拍=int(getattr(self.桥, "拍", 0)),
                                        新组合格数=im.get("新组合格数"),
                                        ε_self=(None if im.get("ε_self") is None
                                                else round(float(im["ε_self"]), 4))))
                        if 画面目录:
                            self.存画面png(名字="画面_第%d拍" % int(getattr(self.桥, "拍", 0)))
                            self.存画面png(名字="画面_最新")
                except Exception:
                    pass
        后 = self.内核态()
        自省 = self.自省文()
        return dict(自主=True, 拍数=int(拍数), 用时s=round(time.time() - t0, 2),
                    ε条数=len(ε轨迹),
                    ε首=(ε轨迹[0] if ε轨迹 else None),
                    ε末=(ε轨迹[-1] if ε轨迹 else None),
                    ε均=(round(sum(ε轨迹) / len(ε轨迹), 4) if ε轨迹 else None),
                    站轨迹去重=sorted(set(站变化)),
                    想象s=想象s,
                    自省条数=len(自省), 自省样本=自省,
                    脑内=self._近期自语(),
                    内核前=前, 内核后=后)

    @staticmethod
    def 思核站(思核) -> str:
        try:
            return str(思核.站点(思核.迹[-1]["z"]))
        except Exception:
            return "?"

    def _近期自语(self, 上限: int = 5) -> list:
        """它自己的最近几句话（自言流：疑问/预测/复盘/自评/计划）——「脑子里在想什么」。

        ⚠ 自言流条目是 **dict**（`inner_speech.自言空间.追加`：号/拍/类型/文本/触发源/引用），
        不是对象——首版我按对象读属性，结果全空（已修）。
        """
        if self.桥 is None:
            return []
        流 = getattr(getattr(self.桥, "自言", None), "流", []) or []
        出 = []
        for x in 流[-int(上限):]:
            if isinstance(x, dict):
                出.append(dict(拍=x.get("拍"), 类型=str(x.get("类型", "")),
                               文本=str(x.get("文本", ""))[:130]))
            else:
                出.append(dict(拍=getattr(x, "拍", None), 类型=str(getattr(x, "类型", "")),
                               文本=str(getattr(x, "文本", ""))[:130]))
        return 出

    def 探针一拍(self) -> int:
        try:
            self.态.写思考("想", "V3 演示：内部回响（不外显，仅探针可见）", 层="内")
        except Exception:
            pass
        self.探针.扫(self.态, 上限=10)
        return self.探针.计数


def 演示(种子: int = 310):
    _行("=" * 68)
    _行("AHI V3 演示 —— 能启动 · 能看网页 · 能学 · 能操作 · 能说话")
    _行("=" * 68)
    t0 = time.time()
    命 = V3生命(种子=种子)
    _行("[1/5] 启动生命（共存态 + 记忆底图 M + 四模块）… OK")
    命.起综合体()
    _行("      综合体：%s" % ("已挂载（想/回忆/自言）" if 命.桥 else "未挂载（降级）"))

    _行("[2/5] 看网页（模块三：UIA 结构 + 视觉块交叉验证）…")
    r = 命.看网页()
    布 = r.get("布局") or {}
    _行("      窗口：%s" % (r.get("窗口") or {}).get("标题", ""))
    _行("      UIA 控件数=%s，视觉块=%s，未覆盖=%s，不一致=%s" % (
        r.get("控件数"), (r.get("视觉通道") or {}).get("块数"),
        (r.get("交叉验证") or {}).get("未覆盖数"), (r.get("交叉验证") or {}).get("不一致数")))
    _行("      正文区=%s 导航区=%s 正文段数=%s" % (布.get("正文区"), 布.get("导航区"), 布.get("正文段数")))
    _行("      正文预览：%s" % (布.get("正文预览") or "")[:120].replace("\n", " / "))

    _行("[3/5] 学（模块四：摄入 → 关联 → 复盘）…")
    xr = 命.学(布.get("正文预览") or "")
    复 = xr.get("复盘") or {}
    _行("      摄入 %s 句；新绑定 %s 条；复盘关键词=%s；快照入库=%s" % (
        xr.get("摄入"), xr.get("新绑定"), 复.get("关键词"), 复.get("入库")))
    _行("      概念快照：%s" % (复.get("摘要") or "—"))

    _行("[4/5] 操作规划（模块二：功能档案三层过滤 + 节奏感）…")
    规 = 命.操作规划((r.get("窗口") or {}).get("标题", ""), r.get("控件样本") or [])
    _行("      界面类型=%s（%s）" % (规["界面"], 规["界面经验"]))
    _行("      裁决：%s —— %s" % (规["裁决"]["动作"], 规["裁决"]["理由"]))
    _行("      节奏（距目标 180px）：%s" % (规["节奏"],))

    _行("[5/5] 说话（模块一：只读 S / 意图 / M / 五内核内态 → 带个性的话）…")
    # ★先把正文喂进引擎链一拍（五内核对**它自己看的东西**留下痕迹；provenance=自源）
    喂r = 命.喂(布.get("正文预览") or "", 来源="自源")
    _行("      喂引擎一拍：%s（拍=%s，内态 provenance=自源）" % (
        喂r.get("喂"), 喂r.get("拍")))
    说 = 命.说话(话题=布.get("正文预览") or "")
    _行("      S：拍=%s 心境=%s 效价=%s 唤醒=%s 消化债=%s" % (
        说["S"]["拍"], 说["S"]["区"], 说["S"]["效价"], 说["S"]["唤醒"], 说["S"]["消化债"]))
    _行("      它说：%s" % 说["句"])
    _行("      五内核内态：%s" % json.dumps(说["内态"], ensure_ascii=False, default=str)[:400])
    for z in (说.get("自省") or []):
        _行("      脑内（不外显）：%s" % z["文本"])
    n探针 = 命.探针一拍()
    _行("      探针（只读，不进输出流）已记 %d 条 → 测试记录\\探针_思考核_v3.ndjson" % n探针)

    报 = dict(种子=种子, 用时s=round(time.time() - t0, 2),
              看网页=dict(控件数=r.get("控件数"), 正文段数=布.get("正文段数"),
                        导航数=布.get("导航数"), 外壳剔除数=布.get("外壳剔除数"),
                        交叉验证=r.get("交叉验证"),
                        存入记忆=r.get("存入记忆")),
              学=dict(摄入=xr.get("摄入"), 新绑定=xr.get("新绑定"),
                     关键词=复.get("关键词"), 摘要=复.get("摘要"), 入库=复.get("入库"),
                     学习迹=复.get("学习迹")),
              操作规划=规, 说话=说, 喂引擎=喂r, 探针条数=n探针,
              五内核=命.内核态(),
              模块一测试=命.输出 is not None, 模块二测试=命.档案 is not None)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "V3演示_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    _行("=" * 68)
    _行("完成。结果落盘 → 测试记录\\V3演示_结果.json（用时 %.1fs）" % (time.time() - t0))
    return 报


# ══════════════════ ★真刷真实网页：看 → 学 → 消化 → 说话 ══════════════════
默认网址s = ("https://www.sohu.com/", "https://news.sohu.com/", "https://www.163.com/")


def 刷网页(网址s=None, 种子: int = 310, 每站等s: float = 6.0):
    """真去刷真实网页（默认搜狐）：逐站 打开 → 看（UIA 结构）→ 学（三阶段消化）→ 喂引擎。

    这是「先教它东西、让它去刷正经网页」的可执行形态：
      · 看：模块三 UIA 结构通道（不 OCR），拿标题/正文/导航；
      · 学：模块四 摄入→关联（**语义通路**）→复盘 → 写入**独立学习迹**；
      · 喂：把正文喂进五内核链一拍（自模型/思核/真学习/自我回路/顿悟都在这一拍上跑）。
    全程只读网页结构 + 纯 numpy 统计，**不 OCR、不外挂模型**。
    """
    网址s = list(网址s or 默认网址s)
    _行("=" * 68)
    _行("AHI V3 · 真刷真实网页 —— 看 → 学 → 消化 → 喂引擎（%d 站）" % len(网址s))
    _行("=" * 68)
    t0 = time.time()
    命 = V3生命(种子=种子)
    if not 命.起综合体():
        _行("  [警告] 五内核未挂载，继续以四模块跑")
    else:
        _行("  五内核已挂载：%s" % "、".join(
            k for k, v in (命.内核态() or {}).items() if v and k != "挂载"))

    报s = []
    for url in 网址s:
        _行("\n[站] %s" % url)
        st = time.time()
        try:
            r = 命.看网页(url=url, 等s=每站等s)
        except Exception as ex:
            _行("      看失败：%r" % ex)
            r = {}
        布 = r.get("布局") or {}
        标题 = (r.get("窗口") or {}).get("标题", "")
        正文 = 布.get("正文预览") or ""
        _行("      窗口：%s" % 标题)
        _行("      控件=%s 正文段=%s 导航=%s 外壳剔除=%s" % (
            r.get("控件数"), 布.get("正文段数"), 布.get("导航数"), 布.get("外壳剔除数")))
        _行("      正文（前140字）：%s" % 正文[:140].replace("\n", " / "))
        if not 正文:
            _行("      [跳过] 本屏无正文（可访问性树未暴露或仍在加载）")
            continue
        xr = 命.学(正文)
        复 = xr.get("复盘") or {}
        _行("      学：摄入 %s 句 / 新绑定 %s 条 / 关键词 %s" % (
            xr.get("摄入"), xr.get("新绑定"), 复.get("关键词")))
        _行("      复盘：%s（学习迹第 %s 条；记忆底图未增=%s）" % (
            复.get("摘要"), 复.get("学习迹"), 复.get("记忆底图未增")))
        前 = 命.内核态()
        # ★最多片=60：真站点的 UIA 遍历把导航与正文混排，真新闻句多在**前 24 句之后**
        #   （实测：整页正文前 24 句几乎全是导航项）。多喂几片，语义内容才进得来。
        喂r = 命.喂一串(正文, 来源="自源", 最多片=60)
        后 = 命.内核态()
        _行("      喂引擎：%s（连续 %s 拍 → 拍号 %s；句子入记忆 %s 条）" % (
            喂r.get("喂"), 喂r.get("片数"), 喂r.get("拍"), 喂r.get("句子入记忆")))
        _行("      五内核前后对照：ε%s→%s ｜ 自模型更新 %s→%s ｜ 思核迹 %s→%s ｜ 顿悟 %s→%s" % (
            (前.get("自模型") or {}).get("ε均值"), (后.get("自模型") or {}).get("ε均值"),
            (前.get("自模型") or {}).get("更新次数"), (后.get("自模型") or {}).get("更新次数"),
            (前.get("思核") or {}).get("迹条数"), (后.get("思核") or {}).get("迹条数"),
            (前.get("顿悟涌现") or {}).get("顿悟条数"), (后.get("顿悟涌现") or {}).get("顿悟条数")))
        # ★第二趟「关联」：此时**逐句已在记忆里**，句对句才有可比的粒度。
        #   实测口径：整页 vs 整页 0.82（全是导航词）；单句 vs 整页 0.01~0.04（绑不上）。
        绑2 = 命.消化.关联()
        例 = ("，例：" + str(绑2[0])[:70]) if 绑2 else "（本屏句子与库内旧句相似度不足——不硬绑，这正是相似下限的作用）"
        _行("      再关联（句级记忆就绪后）：新绑定 %s 条%s" % (len(绑2), 例))
        说 = 命.说话(话题=正文)
        _行("      它说：%s" % 说["句"])
        for z in (说.get("自省") or []):
            _行("      脑内（不外显）：%s" % z["文本"])
        报s.append(dict(网址=url, 标题=标题, 控件数=r.get("控件数"),
                       正文段数=布.get("正文段数"), 导航数=布.get("导航数"),
                       外壳剔除数=布.get("外壳剔除数"), 正文=正文[:1200],
                       学=dict(摄入=xr.get("摄入"), 新绑定=xr.get("新绑定"),
                              关键词=复.get("关键词"), 摘要=复.get("摘要"),
                              学习迹=复.get("学习迹"), 记忆底图未增=复.get("记忆底图未增")),
                       再关联=dict(绑定数=len(绑2), 样本=绑2[:3]),
                       喂=喂r, 内核前=前, 内核后=后, 说话=说["句"], 内态=说.get("内态"),
                       用时s=round(time.time() - st, 2)))
    os.makedirs(OUT, exist_ok=True)
    报 = dict(种子=种子, 用时s=round(time.time() - t0, 2), 站s=报s,
              五内核=命.内核态(),
              学习迹快照=(命.消化.快照().get("学习迹") if 命.消化 is not None else None),
              记忆库大小=命.容量.库大小(),
              容量桥统计=dict(命.容量.统计))
    with open(os.path.join(OUT, "刷网页_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    _行("\n" + "=" * 68)
    _行("刷完 %d 站，用时 %.1fs → 测试记录\\刷网页_结果.json" % (len(报s), time.time() - t0))
    return 报


def 自主(拍数: int = 600, 种子: int = 310, 网页: bool = False, 网址s=None,
        读文件: bool = False, 读根: str = None, 读间隔拍: int = 40,
        刷一次s: float = 60.0, 休息s: float = 0.0,
        想象: bool = False, 想象间隔拍: int = 300) -> dict:
    """★真自主：五内核全程在链上，自维持跑 N 拍自主拍（`--自主` 的新实现）。

    `网页=True` 时，每「刷一次s」秒去刷一屏真实网页（看→学→喂），其余时间自主拍。
    `读文件=True` 时，每「读间隔拍」拍读一段自己工作区里的源码喂进引擎——
    这是「**给它东西学**」的自主形态（它自己去读，provenance=自源）。

    ★为什么要「给输入」（第三轮实测）：纯自主 1200 拍，自模型 ε 首窗 0.4706 → 末窗 **0.5025（不降）**，
    因为**自态几乎不动**（相邻帧变化极小）⇒ **没有新信息可学**（不是学习坏掉：
    `步2` 实测有输入时 训首 0.6256→训末 0.0103、AUC=1.0）。
    加"每 40 拍读一段文件"后：ε 首窗 0.5458 → 末窗 **0.4709（下降）**——同 seed 同拍数对照。
    """
    _行("=" * 68)
    _行("AHI V3 · 自主运行（五内核全程在链上）：%d 拍%s%s" % (
        拍数, "＋周期性真刷网页" if 网页 else "", "＋周期性读源码" if 读文件 else ""))
    _行("=" * 68)
    t0 = time.time()
    命 = V3生命(种子=种子)
    if not 命.起综合体():
        _行("  [警告] 五内核未挂载")
    else:
        _行("  五内核已挂载：%s" % "、".join(
            k for k, v in (命.内核态() or {}).items() if v and k != "挂载"))
    if 想象:
        接 = 命.接画面()
        _行("  画面层（视觉想象）：%s；每 %d 拍想一幅新画面" % ("已接帧源" if 接 else "未挂载", 想象间隔拍))
    文件s = _列源码(读根) if 读文件 else []
    if 读文件:
        _行("  可读源码：%d 个文件（每 %d 拍读一段）" % (len(文件s), 读间隔拍))
    段s = []
    剩 = int(拍数)
    每次 = 60
    站 = list(网址s or 默认网址s)
    轮 = 0
    读序 = 0
    ε段 = []
    while 剩 > 0:
        本 = min(每次, 剩)
        # 段内穿插"读源码"：以拍为单位，每 读间隔拍 读一段
        if 读文件 and 文件s:
            for i in range(0, 本, max(1, int(读间隔拍))):
                小 = min(int(读间隔拍), 本 - i)
                命.自主运行(拍数=小, 休息s=休息s,
                          想象间隔=(想象间隔拍 if 想象 else 0),
                          画面目录=(OUT if 想象 else None))
                try:
                    t = open(文件s[读序 % len(文件s)], encoding="utf-8", errors="replace").read()
                    读序 += 1
                    命.喂一串(t[:1500], 来源="自源", 最多片=25)
                except Exception:
                    pass
            r = 命.自主运行(拍数=0) if False else _末段(命)
        else:
            r = 命.自主运行(拍数=本, 休息s=休息s,
                          想象间隔=(想象间隔拍 if 想象 else 0),
                          画面目录=(OUT if 想象 else None))
        段s.append(r)
        if r.get("ε末") is not None:
            ε段.append(r["ε末"])
        _行("  [段%d] %d 拍：ε %s→%s（均 %s）｜思核站 %s｜顿悟 %s" % (
            轮 + 1, 本, r.get("ε首"), r.get("ε末"), r.get("ε均"),
            r.get("站轨迹去重"),
            (r.get("内核后") or {}).get("顿悟涌现", {}).get("顿悟条数")
            if isinstance((r.get("内核后") or {}).get("顿悟涌现"), dict) else None))
        剩 -= 本
        轮 += 1
        if 网页 and 站:
            url = 站[轮 % len(站)]
            _行("        （自主间隔：去刷一屏 %s）" % url)
            try:
                pg = 命.看网页(url=url, 等s=6.0)
                布 = pg.get("布局") or {}
                正文 = 布.get("正文预览") or ""
                if 正文:
                    命.学(正文)
                    命.喂一串(正文, 来源="自源", 最多片=60)
                    _行("          看/学/喂完成：正文段 %s，控件 %s" % (
                        布.get("正文段数"), pg.get("控件数")))
            except Exception as ex:
                _行("          刷网页失败：%r" % ex)
    后 = 命.内核态()
    报 = dict(种子=种子, 拍数=int(拍数), 用时s=round(time.time() - t0, 2),
              读文件=bool(读文件), 读文件数=len(文件s),
              想象=bool(想象), 想象间隔拍=int(想象间隔拍),
              段s=[{k: v for k, v in x.items() if k not in ("内核前", "内核后")} for x in 段s],
              内核末=后,
              自省=命.自省文(),
              脑内=命._近期自语(上限=8),
              学习迹=(命.消化.快照().get("学习迹") if 命.消化 is not None else None),
              记忆库大小=命.容量.库大小())
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "自主运行_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    _行("\n" + "=" * 68)
    _行("自主运行完成（%d 拍，%.1fs）→ 测试记录\\自主运行_结果.json" % (拍数, time.time() - t0))
    for z in (报.get("自省") or []):
        _行("  脑内（不外显）：%s" % z["文本"])
    return 报


def _末段(命) -> dict:
    """读文件穿插模式下，取一次"零拍"汇总（只读当前内核态，不推进）。"""
    return 命.自主运行(拍数=1, 记录间隔=0, 打印=False)


def 画面演示(种子: int = 310) -> dict:
    """★画面（视觉想象）演示：看一屏 → **从记忆里读出画面** → 把记忆里的物件搬到新位置。

    诚实口径：读出来的是**粗结构画面**（16×9 格里每件 2×2×3 的粗块），不是照片；
    但它确实是从那份 ±1 码**解**出来的（不是把原帧存下来重放）——PNG 落
    `测试记录/画面_演示_回放.png` 与 `画面_演示_想象.png` 供人眼核对。
    """
    _行("=" * 68)
    _行("AHI V3 · 画面演示（视觉想象：从自己的画面码里解出一屏，再组合出新画面）")
    _行("=" * 68)
    命 = V3生命(种子=种子)
    if not 命.起综合体():
        _行("  [警告] 综合体未挂载")
        return dict(画面=False)
    接 = 命.接画面()
    _行("  画面层：%s（帧源＝真屏截图；无桌面/无 mss 时退回合成屏）" % ("已接" if 接 else "未接"))
    r = 命.画面一拍()
    if not r.get("画面"):
        _行("  真屏不可用（%s）→ 用合成屏演示" % r.get("原因"))
        帧 = _合成屏()
        hz = 命._内核宿主
        r = hz.画面一拍(命.桥, 帧=帧) if hz is not None else dict(画面=False)
    核 = getattr(命.桥, "画面核", None)
    _行("  看一屏：结构件 %s 个；回放命中 %s 个" % (
        (r.get("记") or {}).get("物体数"), (r.get("回放") or {}).get("命中数")))
    im = 命.想象一拍(位移=(3, 2))
    # ★"还认不认得出来"是**量**出来的（不能拿"想象动作发生了"冒充"认得出来"——那是两件事）：
    #   逐件读出 → 与库外观比码域相似 → 数过判据的件数（由 内核宿主.想象一拍 给出）。
    _行("  想象：把记忆里的物件整体搬到 +(%d,%d) 格；新格位 %s 个；**仍认得出来 %s/%s**"
        "（码似最低 %s）；喂自模型 ε_self=%s" % (
            3, 2, im.get("新组合格数"), im.get("认得件数"), im.get("总件数"),
            (None if im.get("码似最低") is None else round(float(im["码似最低"]), 3)),
            (None if im.get("ε_self") is None else round(float(im["ε_self"]), 4))))
    核 = getattr(命.桥, "画面核", None)
    if 核 is not None:
        _行("  画面核摘要：%s" % 核.摘要())
    存 = 命.存画面png(名字="画面_演示")
    _行("  落盘：%s" % 存)
    报 = dict(画面=True, 记=r.get("记"), 回放=r.get("回放"), 想象=im,
              存=存, 核=(核.摘要() if 核 is not None else None),
              接帧源=bool(接), 五内核=命.内核态())
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "画面演示_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    return 报


def _合成屏(H: int = 144, W: int = 256):
    """合成一屏（7 件分离结构）——用于没有桌面/没有 mss 的场合，保证演示可复现。"""
    import numpy as _np
    rng = _np.random.default_rng(7)
    图 = _np.full((H, W, 3), 40, _np.float32)
    for x, y, w, h, c in [(16, 16, 32, 32, (230, 60, 60)), (80, 16, 32, 32, (60, 200, 90)),
                          (144, 16, 32, 32, (80, 120, 240)), (208, 16, 32, 32, (240, 220, 40)),
                          (16, 96, 64, 16, (250, 250, 250)), (112, 96, 32, 48, (20, 20, 20)),
                          (192, 96, 48, 32, (150, 90, 200))]:
        图[y:y + h, x:x + w] = _np.asarray(c, _np.float32)
    return 图 + rng.normal(0, 3, 图.shape).astype(_np.float32)


def _列源码(根: str = None, 上限: int = 400) -> list:
    """列出可读的源码文件（确定性排序；跳过备份与缓存）。`根=None` → AHI3 自身。"""
    根 = 根 or AHI3
    出 = []
    for 目, 目录s, 文件s in os.walk(根):
        目录s[:] = [d for d in 目录s
                   if not d.startswith(("backup_", "__pycache__", "测试记录", ".git"))]
        for f in sorted(文件s):
            if f.endswith((".py", ".md")):
                出.append(os.path.join(目, f))
            if len(出) >= int(上限):
                return sorted(出)
    return sorted(出)


def main():
    ap = argparse.ArgumentParser(description="AHI V3 主程序")
    ap.add_argument("--看网页", metavar="URL", default=None, help="只看一屏网页并打印")
    ap.add_argument("--刷网页", metavar="URL", nargs="*", default=None,
                    help="真刷真实网页（默认搜狐首页）：看→学→消化→说话")
    ap.add_argument("--交互", action="store_true", help="纯文本对话（走 2.1 文本态）")
    ap.add_argument("--自主", type=int, default=None, metavar="拍数",
                    help="★真自主：五内核链上跑 N 拍自主拍（原「秒数」语义已改为拍数）")
    ap.add_argument("--自主网页", action="store_true", help="自主运行期间周期性真刷网页")
    ap.add_argument("--自主读", action="store_true",
                    help="★自主运行期间周期性读自己工作区的源码（给它东西学；实测 ε 会降）")
    ap.add_argument("--自主想象", action="store_true",
                    help="★自主运行期间周期性「从记忆里想出一幅新画面」（视觉想象；落 PNG）")
    ap.add_argument("--想象间隔", type=int, default=300, help="每多少拍想一幅（默认 300）")
    ap.add_argument("--画面", action="store_true",
                    help="★画面演示：看一屏 → 从记忆里读出画面 → 组合想象（落 PNG）")
    ap.add_argument("--读间隔", type=int, default=40, help="每多少拍读一段（默认 40）")
    ap.add_argument("--旧自主", type=float, default=None, metavar="秒数",
                    help="走 2.1 文本态的旧自主模式（兼容保留）")
    ap.add_argument("--种子", type=int, default=310)
    a = ap.parse_args()
    if a.交互 or a.旧自主 is not None:
        # 复用已跑通的纯文本对话态（文本态.py）
        from 文本态 import 文本态
        t = 文本态(种子=a.种子).建(根s=[AHI3])
        if a.交互:
            t.运行_交互()
        else:
            t.运行_脚本([], 每轮自主拍=0, 结尾自主拍=int(a.旧自主))
        return
    if a.自主 is not None:
        自主(拍数=int(a.自主), 种子=a.种子, 网页=bool(a.自主网页),
             读文件=bool(a.自主读), 读间隔拍=int(a.读间隔),
             想象=bool(a.自主想象), 想象间隔拍=int(a.想象间隔))
        return
    if a.画面:
        画面演示(种子=a.种子)
        return
    if a.刷网页 is not None:
        刷网页(网址s=(a.刷网页 or None), 种子=a.种子)
        return
    if a.看网页:
        命 = V3生命(种子=a.种子)
        r = 命.看网页(url=a.看网页) if a.看网页 != "本地" else 命.看网页()
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
        return
    演示(种子=a.种子)


if __name__ == "__main__":
    main()