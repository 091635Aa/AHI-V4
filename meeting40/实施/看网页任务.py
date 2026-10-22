# -*- coding: utf-8 -*-
"""meeting40/实施/看网页任务.py —— 基础任务③：**看网页**（读 → 学进记忆 → 产出文档 → 说一句）。

口径（所有者：「看网页这些基础的完成」）：
  ① **读**：`视觉_看网页.打开网页()` 开真站（或本地页）→ `网页眼.看()`（**UIA 结构通道，不 OCR**）
     拿标题/正文/导航/外壳剔除；
  ② **学**：正文（真语义向量）入 `容量桥`（`类型="网页正文"`，`来源="网页"`）⇒ 之后可被 `查经历` **语义检索命中**；
  ③ **产出**：复用 [文档任务.py](文档任务.py) 把读到的正文合成一份**网页摘要文档**（`.docx`+`.md`）；
  ④ **说**：复用 `输出层.生成正文`（**只读** S/意图/M）说**一句**带个性的话（思考不公布，模块一契约）。

**通路如实标注**（不假装看过）：
  · `真浏览器UIA` —— 真开了浏览器且 UIA 读到了正文（`正文段数>0`）；
  · `本地页(无UIA)` —— 没开浏览器 / 可访问性树没暴露正文 ⇒ 用给定本地页正文，**并在证据里标清楚**；
  · `失败` —— 连本地正文都没有 ⇒ **不写记忆、不产文档**，如实返回失败（sham 判据就查这条）。

纪律：纯新增文件；不 import 器官；不改既有件；真开浏览器只在 `允许开浏览器=True` 时发生（默认 False，避免夺前台）。
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

import 文档任务

产物目录默认 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "产物")


def _切句(正文: str, 最少: int = 6) -> list:
    句s = [s.strip() for s in str(正文).replace("\r", "\n").replace("\n", "。").split("。")]
    return [s for s in 句s if len(s) >= 最少]


def 写本地页(段落s: list, 目录: str, 标题: str = "AHI 本地演示页") -> str:
    """把一个本地 HTML 页写盘（确定性）。**只用于无浏览器时的可复现通路**。"""
    os.makedirs(目录, exist_ok=True)
    体 = "".join("<p>%s</p>" % str(x) for x in 段落s)
    码 = ("<!DOCTYPE html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
         "<title>%s</title></head><body><h1>%s</h1>%s</body></html>"
         % (标题, 标题, 体))
    名 = "网页_%s.html" % hashlib.sha256(码.encode("utf-8")).hexdigest()[:8]
    路径 = os.path.join(目录, 名)
    with open(路径, "w", encoding="utf-8") as f:
        f.write(码)
    return 路径


def 判通路(标题前: str, 标题后: str, 正文段数: int, 关键词: str = "") -> tuple:
    """**纯函数**通路裁决（可判据化，不碰浏览器）。

    返回 `(通路, 切到目标)`：
      · `真浏览器UIA`        —— 正文段数>0 且前台**确实切到了目标页**（标题变了，或标题含目标关键词）；
      · `前台未切到目标页`   —— 读到正文了，但前台窗口没换（读到的是**别的窗口**）⇒ 必须如实标注，
                                绝不许把别的窗口的内容当成目标站；
      · `本地页(无UIA)`      —— 没读到 UIA 正文（无正文段）；调用方改用本地页正文；
      · `失败`               —— 连本地正文也没有。
    ★这条闸是**真站实测逼出来的**：首跑 `sohu.com` 时前台仍是 DSH 窗口（标题「DeepSeek Harness」），
      旧实现却报 `真浏览器UIA` —— 等于把别的窗口当成搜狐。
    """
    前 = str(标题前 or "").strip()
    后 = str(标题后 or "").strip()
    段 = int(正文段数 or 0)
    切 = bool(后 and (后 != 前 or (关键词 and str(关键词).lower() in 后.lower())))
    if 段 > 0 and 切:
        return "真浏览器UIA", True
    if 段 > 0:
        return "前台未切到目标页", False
    return "本地页(无UIA)", False


def _关键词(url: str) -> str:
    """从 URL 取域名主干做标题比对关键词（如 www.sohu.com → sohu）。"""
    s = str(url or "")
    s = s.split("//")[-1].split("/")[0]
    部 = [x for x in s.split(".") if x and x not in ("www", "com", "cn", "net", "org", "html")]
    return 部[0] if 部 else ""


def 看网页(url: str = None, 段落s: list = None, 允许开浏览器: bool = False, 等s: float = 6.0,
          桥=None, 产物目录: str = None, 主题: str = None, 打开方式: str = "edge") -> dict:
    """看一屏网页 → 学 → 产文档 → 说一句。返回证据 dict（含 `通路`）。

    `打开方式`：`"edge"`＝复用 `视觉_看网页.打开网页`（Chromium + 可访问性树旗标）；
    `"默认"`＝`os.startfile(url)`（默认浏览器，通常更容易拿到前台）。
    """
    目录 = 产物目录 or 产物目录默认
    os.makedirs(目录, exist_ok=True)
    段落s = [str(x) for x in (段落s or [])]
    t0 = time.time()
    通路, 标题, 正文 = None, "", ""
    眼报, 标题前, 标题后, 切到目标 = None, "", "", None
    本地页 = 写本地页(段落s, 目录) if 段落s else None

    # ── ① 读 ──
    if 允许开浏览器 and (url or 本地页):
        try:
            from 视觉_看网页 import 打开网页, 网页眼
            眼 = 网页眼(种子=3800, 编码器=None, 记忆=None)
            标题前 = str(眼.窗口标题() or "")                     # ★开浏览器**前**的前台标题
            if str(打开方式) == "默认":
                os.startfile(str(url or 本地页))
            else:
                打开网页(url or 本地页, 用edge=True)
            time.sleep(max(1.0, float(等s)))
            眼报 = 眼.看(存记忆=False)
            布局 = (眼报 or {}).get("布局") or {}
            标题后 = str(((眼报 or {}).get("窗口") or {}).get("标题") or "")
            正文 = str(布局.get("正文预览") or "")
            通路, 切到目标 = 判通路(标题前, 标题后, int(布局.get("正文段数") or 0),
                                 _关键词(url))
            标题 = 标题后 or 标题前
        except Exception as ex:
            眼报 = dict(异常=repr(ex))
    if 通路 in (None, "本地页(无UIA)", "失败") or not 正文:
        正文 = "\n".join(段落s)
        标题 = 标题 or (str(眼报.get("窗口", {}).get("标题") or "") if isinstance(眼报, dict)
                      and isinstance(眼报.get("窗口"), dict) and 眼报.get("窗口") else "")
        标题 = 标题 or ("AHI 本地演示页" if 段落s else "")
        通路 = "本地页(无UIA)" if 段落s else "失败"

    # ── 失败：如实返回，不写记忆、不产文档 ──
    if 通路 == "失败" or len(str(正文).strip()) < 6:
        证据 = dict(任务="看网页", 通路="失败", 原因="没读到正文（未开浏览器且未给本地页正文）",
                    URL=url, 本地页=本地页, 入记忆=None, 文档=None, 说话=None,
                    眼报=眼报, 耗时s=round(time.time() - t0, 3))
        _存证据(目录, "看网页_失败", 证据)
        return 证据

    正文 = str(正文).strip()

    # ── ② 学：正文（带语义向量）入记忆 ──
    入记忆 = None
    if 桥 is not None:
        try:
            向量 = None
            编 = getattr(桥, "编码器", None)
            if 编 is not None:
                向量 = 编.文本码(正文[:512])
            rid = 桥.记经历(拍=0, 类型="网页正文", 文本=正文[:400], 来源="网页", 向量=向量)
            入记忆 = dict(rid=rid, 向量维=(None if 向量 is None else int(np.asarray(向量).shape[0])),
                        字数=len(正文))
        except Exception as ex:
            入记忆 = dict(异常=repr(ex))

    # ── 检索回证（学进去没有：用**同一向量**查，相似必须高）──
    检索 = None
    if 桥 is not None and 入记忆 and "异常" not in 入记忆:
        try:
            编 = getattr(桥, "编码器", None)
            q = 编.文本码(正文[:512]) if 编 is not None else None
            if q is not None:
                命 = 桥.查经历(np.asarray(q, dtype=np.float32), 通路="语义")
                检索 = None if 命 is None else dict(文本=str(命.get("文本", ""))[:60],
                                                相似=round(float(命.get("相似") or 0.0), 4),
                                                通路=str(命.get("通路", "")))
        except Exception as ex:
            检索 = dict(异常=repr(ex))

    # ── ③ 产出：网页摘要文档（复用任务①）──
    文档 = None
    主题默认 = "网页摘要：%s" % (标题 or url or "本地页")
    if 通路 == "前台未切到目标页":
        主题默认 = "（注意：前台未切到目标页，实际读到的是另一个窗口）" + 主题默认
    try:
        素材 = []
        for s in _切句(正文)[:12]:
            素材.append(dict(文本=s, 来源="网页", 向量=None))
        文档 = 文档任务.做文档(主题=(主题 or 主题默认), 桥=桥, 素材s=素材, 产物目录=目录,
                          文件名="网页摘要_%s" % hashlib.sha256(正文[:200].encode("utf-8")
                                                              ).hexdigest()[:8])
    except Exception as ex:
        文档 = dict(异常=repr(ex))

    # ── ④ 说：输出层只读生成一句（思考不公布）──
    说话 = None
    try:
        from 输出层 import 输出层 as _输出层
        层 = _输出层(种子=310)
        S = dict(点=dict(V=0.2, A=0.5), 拍=0)
        意图 = dict(类别="说", 目标=(标题 or "这一屏"))
        M = (dict(文本=(检索 or {}).get("文本", ""), 相似=(检索 or {}).get("相似", 0.0),
                  来源="网页") if 检索 else None)
        说话 = 层.生成正文(意图, S, M)
    except Exception as ex:
        说话 = "【说话失败】%r" % (ex,)

    证据 = dict(任务="看网页", 通路=通路, URL=url, 本地页=本地页, 标题=标题,
                前台标题_开浏览器前=标题前, 前台标题_读完后=标题后,
                目标关键词=_关键词(url), 切到目标页=切到目标,
                正文段数=len(_切句(正文)), 正文字数=len(正文), 正文前200=正文[:200],
                控件数=(None if not isinstance(眼报, dict) else 眼报.get("控件数")),
                外壳剔除数=(None if not isinstance(眼报, dict)
                        else ((眼报.get("布局") or {}).get("外壳剔除数"))),
                入记忆=入记忆, 检索回证=检索,
                文档=(None if 文档 is None else dict(docx=文档.get("产物", {}).get("docx"),
                                                 md=文档.get("产物", {}).get("md"),
                                                 sha256=(文档.get("docx") or {}).get("sha256"),
                                                 条目数=文档.get("条目数"))),
                说话=说话, 耗时s=round(time.time() - t0, 3),
                口径="真语义向量入记忆（UIA 不 OCR）；产出复用文档任务；说话复用输出层（思考不公布）")
    _存证据(目录, "看网页_%s" % hashlib.sha256((url or 正文[:80]).encode("utf-8")).hexdigest()[:8], 证据)
    return 证据


def _存证据(目录: str, 名: str, 证据: dict) -> str:
    路径 = os.path.join(目录, 名 + "_证据.json")
    with open(路径, "w", encoding="utf-8") as f:
        json.dump(证据, f, ensure_ascii=False, indent=1, default=str)
    证据["证据文件"] = 路径
    return 路径


# ─────────────────────────── 自测 ───────────────────────────
def 自测() -> dict:
    import shutil
    import sys
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _AHI3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _p in (os.path.join(_AHI3, "meeting38"), os.path.join(_AHI3, "meeting38", "实施")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from 共存态 import 编码器
    from 容量桥 import 容量桥
    目录 = tempfile.mkdtemp(prefix="meeting40_看网页_")
    案 = {}
    try:
        桥 = 容量桥(种子=3800, 编码器=编码器(种子=3800, 用gpu=False))
        段 = ["模块三用 UIA 结构通道看网页，不做 OCR。",
              "正文以语义向量存进容量桥，之后能按语义检索回来。",
              "看网页任务的产物是一份网页摘要文档，并说一句自己的话。"]
        库0 = 桥.库大小()
        果 = 看网页(段落s=段, 允许开浏览器=False, 桥=桥, 产物目录=目录, 主题="看网页自测")
        库1 = 桥.库大小()
        案["通路标注"] = (果["通路"] == "本地页(无UIA)")
        案["读到正文"] = (果["正文段数"] >= 3 and 果["正文字数"] > 20)
        案["入记忆"] = (库1 == 库0 + 1 and bool(果["入记忆"]))
        案["检索回证"] = bool(果["检索回证"] and 果["检索回证"].get("相似", 0) >= 0.9
                        and 果["检索回证"].get("通路") == "语义")
        案["产出文档"] = bool(果["文档"] and os.path.exists(果["文档"]["docx"])
                        and 文档任务._zip合法(果["文档"]["docx"]))
        案["说了话"] = bool(果["说话"] and len(str(果["说话"])) >= 4)
        案["无元话语"] = ("内部状态" not in str(果["说话"]) and "向量" not in str(果["说话"]))
        案["证据落盘"] = os.path.exists(果["证据文件"])
        # ★诚实闸（真站实测逼出来的）：前台没切到目标页 ⇒ 必须标「前台未切到目标页」，不许报「真浏览器UIA」
        案["闸_没切页不冒充"] = (判通路("DeepSeek Harness", "DeepSeek Harness", 18, "sohu")
                          == ("前台未切到目标页", False))
        案["闸_切页才算真"] = (判通路("旧窗口", "搜狐首页", 18, "sohu")
                         == ("真浏览器UIA", True))
        案["闸_中文标题靠换页判定"] = (判通路("旧窗口", "搜狐首页", 5, "sohu")
                             == ("真浏览器UIA", True))
        案["闸_标题没变但含关键词也算"] = (判通路("Sohu Homepage", "Sohu Homepage", 5, "sohu")
                              == ("真浏览器UIA", True))
        案["闸_没换页又无关键词则如实降级"] = (判通路("搜狐首页", "搜狐首页", 5, "sohu")
                                == ("前台未切到目标页", False))
        案["闸_无正文走本地"] = (判通路("旧", "新", 0, "sohu") == ("本地页(无UIA)", False))
        # sham：什么都不给 ⇒ 必须如实判失败，且**不写记忆**
        库2 = 桥.库大小()
        坏 = 看网页(允许开浏览器=False, 桥=桥, 产物目录=目录)
        案["sham如实失败"] = (坏["通路"] == "失败" and 桥.库大小() == 库2
                        and 坏["文档"] is None and 坏["入记忆"] is None)
        return dict(通过=bool(all(案.values())), 逐项=案, 证据=果, 目录=目录)
    finally:
        shutil.rmtree(目录, ignore_errors=True)


if __name__ == "__main__":
    import sys
    果 = 自测()
    print(json.dumps({k: v for k, v in 果.items() if k != "证据"}, ensure_ascii=False, indent=1,
                     default=str))
    print("[看网页任务] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
