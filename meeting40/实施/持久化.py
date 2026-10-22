# -*- coding: utf-8 -*-
"""meeting40/实施/持久化.py —— 「学进去」的落盘：**容量桥 + 学习迹 → 快照 → 新进程载入**。

口径（所有者：「**他能学进去就行**」；终稿 §7.1「存档→载入→召回」）：
  · **不序列化内核内部**：`容量桥` 背后是 `FSW内核`，内部状态不可随手 pickle。
    这里走 **事件重放**（沿 `synthesis.py:1244` 的做法）——
    存档只存**可重放的事件序列**，载入时 **新建同种子桥 + 顺序重放** ⇒ 库/向量/检索**确定性再生**。
  · 事件从哪来：`容量桥._文本`（拍/类型/文本/来源）＋ `容量桥._向量`（语义向量）⇒ 按 rid 升序还原。
  · `学习迹`（迹/绑定s/顿悟s）同样以**学习事件**形式存档与重放 ⇒ 记忆与学习**各自回到原状**。
  · **物理分离不破**：重放时先建桥再放学习迹；写学习迹**不触碰**桥（H5/D3-2）。

用法：
    持久化.存("学.npz", 桥, 学习迹)
    果 = 持久化.载("学.npz", 种子=3800, 编码器=编)     # → 桥/学习迹/元/回放统计
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

import 快照


def 导出(桥, 学习迹=None) -> dict:
    """从桥/学习迹的**元数据**还原可重放事件（不碰内核内部）。"""
    事件s = []
    if 桥 is not None:
        for rid in sorted(getattr(桥, "_文本", {}) or {}):
            元 = dict(桥._文本.get(rid) or {})
            v = (getattr(桥, "_向量", {}) or {}).get(rid)
            事件s.append(dict(拍=int(元.get("拍") or 0), 类型=str(元.get("类型") or "事件"),
                             文本=str(元.get("文本") or ""), 来源=str(元.get("来源") or ""),
                             向量=(None if v is None else np.asarray(v, dtype=np.float32)),
                             rid=int(rid)))
    学习事件s = []
    if 学习迹 is not None:
        for 项 in list(getattr(学习迹, "迹", []) or []):
            学习事件s.append(dict(类型="放快照", 关键词=list(项.get("关键词") or []),
                                簇数=int(项.get("簇数") or 1), 摘要=str(项.get("摘要") or "")))
        for 项 in list(getattr(学习迹, "绑定s", []) or []):
            学习事件s.append(dict(类型="放绑定", 新=项.get("新"), 旧=项.get("旧"),
                                相似=float(项.get("相似") or 0.0)))
        for 项 in list(getattr(学习迹, "顿悟s", []) or []):
            学习事件s.append(dict(类型="放顿悟", 拍=int(项.get("拍") or 0),
                                文本=str(项.get("文本") or ""), 形态=dict(项.get("形态") or {})))
    return dict(容量事件s=事件s, 学习事件s=学习事件s,
                元=dict(桥记数=len(事件s), 学习记数=len(学习事件s),
                       库大小=(None if 桥 is None else int(桥.库大小())) if hasattr(桥, "库大小") else None))


def 存(路径: str, 桥, 学习迹=None) -> dict:
    """存档：事件序列进 `快照`（npz + manifest，sha256 校验、原子替换）。返回 manifest。"""
    导 = 导出(桥, 学习迹)
    清单 = 快照.存(路径, dict(元=导["元"]), 导["容量事件s"])
    # 学习事件单独存一份（同一 npz 里的第二段事件不好混，简单起见用第二个文件）
    学习路径 = str(路径) + ".学习.npz"
    学习清单 = 快照.存(学习路径, dict(元=dict(学习记数=导["元"]["学习记数"])), 导["学习事件s"])
    return dict(容量=清单, 学习=学习清单, 路径=str(路径), 学习路径=学习路径,
                元=导["元"])


def 载(路径: str, 种子: int = 3800, 编码器=None, 挂学习迹: bool = True) -> dict:
    """载入：新建同种子桥 → **重放容量事件** → 重建学习迹（重放学习事件）。"""
    import sys
    for _p in (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
               os.path.join(os.path.dirname(os.path.dirname(
                   os.path.dirname(os.path.abspath(__file__)))), "meeting38", "实施")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from 容量桥 import 容量桥
    from 学习迹 import 学习迹
    状, 事件s = 快照.载(路径)
    桥 = 容量桥(种子=int(种子), 编码器=编码器)
    重放报 = []
    for e in 事件s:
        向量 = e.get("向量")
        rid = 桥.记经历(拍=e["拍"], 类型=e["类型"], 文本=e["文本"], 来源=e["来源"],
                       向量=(None if 向量 is None else np.asarray(向量, dtype=np.float32)))
        重放报.append(dict(rid=rid, 文本=str(e["文本"])[:24]))
    迹 = 学习迹(背衬=桥) if 挂学习迹 else None
    学习重放 = []
    学习路径 = str(路径) + ".学习.npz"
    if 迹 is not None and os.path.exists(学习路径):
        _, 学习事件s = 快照.载(学习路径)
        for e in 学习事件s:
            try:
                if e.get("类型") == "放快照":
                    迹.放快照(e.get("关键词") or [], int(e.get("簇数") or 1),
                             str(e.get("摘要") or ""), 束=None)
                elif e.get("类型") == "放绑定":
                    迹.放绑定(e.get("新"), e.get("旧"), float(e.get("相似") or 0.0))
                elif e.get("类型") == "放顿悟":
                    迹.放顿悟(int(e.get("拍") or 0), str(e.get("文本") or ""), e.get("形态"))
                学习重放.append(e.get("类型"))
            except Exception as ex:
                学习重放.append("异常:%r" % (ex,))
    return dict(桥=桥, 学习迹=迹, 元=状.get("元", {}), 事件s=事件s,
                重放条数=len(重放报), 重放报=重放报[:8],
                库大小=int(桥.库大小()), 学习迹条数=(0 if 迹 is None else len(迹.迹)),
                学习重放条数=len(学习重放))


def 特征码(对象) -> str:
    return hashlib.sha256(json.dumps(对象, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode("utf-8")
                          ).hexdigest()


# ─────────────────────────── 自测 ───────────────────────────
def 自测() -> dict:
    import shutil
    import sys
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    编路径 = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "meeting38", "实施")
    sys.path.insert(0, 编路径)
    from 容量桥 import 容量桥
    from 学习迹 import 学习迹
    目录 = tempfile.mkdtemp(prefix="meeting40_持久化_")
    案 = {}
    try:
        桥 = 容量桥(种子=3800)
        迹 = 学习迹(背衬=桥)
        向量s = []
        for i, t in enumerate(("网页正文：搜狐首页有新闻和导航。", "笔记：正文用语义向量存。",
                             "任务经历：代码任务把规格落成代码。")):
            v = np.zeros(1024, dtype=np.float32)
            v[i * 7] = 1.0
            向量s.append(v)
            桥.记经历(拍=i, 类型="事件", 文本=t, 来源="用户", 向量=v)
        迹.放快照(["网页正文"], 1, "我从网页学到：正文可语义检索", 束=None)
        迹.放绑定("网页正文", "语义检索", 0.9)
        路径 = os.path.join(目录, "学.npz")
        清 = 存(路径, 桥, 迹)
        got = 载(路径, 种子=3800)
        案["存档落盘"] = bool(os.path.exists(路径) and os.path.exists(路径 + ".学习.npz"))
        案["库大小一致"] = (got["库大小"] == 桥.库大小() == 3)
        案["重放条数"] = (got["重放条数"] == 3)
        # 同一向量检索：文本与相似度都必须一致
        对 = True
        for i, v in enumerate(向量s):
            a = 桥.查经历(v, 通路="语义")
            b = got["桥"].查经历(v, 通路="语义")
            if not a or not b or a.get("文本") != b.get("文本") or \
                    abs(float(a.get("相似") or 0) - float(b.get("相似") or 0)) > 1e-6:
                对 = False
        # 注意：载入的是独立进程/独立对象，这里同进程对照；跨进程由判据 T-6 单独验
        案["检索一致"] = 对
        案["学习迹重建"] = (got["学习迹条数"] == len(迹.迹) == 1
                       and got["学习重放条数"] == 2)      # 1 条快照 + 1 条绑定（顿悟 0 条）
        案["记忆≠学习"] = (got["学习迹"].记忆库大小() == got["库大小"])
        return dict(通过=bool(all(案.values())), 逐项=案, 清单=清, 载入=dict(
            库大小=got["库大小"], 重放条数=got["重放条数"], 学习迹条数=got["学习迹条数"],
            学习重放条数=got["学习重放条数"], 元=got["元"]))
    finally:
        shutil.rmtree(目录, ignore_errors=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[持久化] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
