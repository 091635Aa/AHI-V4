# -*- coding: utf-8 -*-
"""meeting38/自检.py —— AHI 并发架构自检（不碰真屏即可证架构）

上一版死于"串行 + 工具调用"：动作没结算完，思维线程 return。本自检的核心就是
**证伪那个病**：手在忙的时候，脑必须仍在推进；轨迹必须连续（无瞬移、能中途改落点）。

自检项：
  A1 并发无阻塞   五路并发锤击共享工作区：无死锁、单次写不超时预算
  A2 轨迹连续     最小急动度轨迹无瞬移；相邻点跳变 ≤ 阈值
  A3 中途纠偏     元素移动 → 剩余段终点被改写（在动中改落点），纠偏计数+1
  A4 手忙脑不停   手状态=移动中时，脑一拍仍能出招/推进（旧版会 return）
  A5 HDC 语义     同名异效分词、异名同效归词、最小包含层级、叠加据裁决收敛
  A6 非 Transformer 审计  全目录无 attention/QKV/softmax/transformers 结构
  A7 可复现       同种子编码器两次输出逐位相同
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                       # noqa: E402

from 共存态 import 共存态, 编码器                          # noqa: E402
from 器官_手 import 手器                                  # noqa: E402
from 器官_脑 import 脑器                                  # noqa: E402
from 语义 import 语义图                                   # noqa: E402


def _合成立体(态, n=8):
    """合成一屏元素（不碰真屏），句柄/字/框/心齐备。"""
    元s = []
    for i in range(n):
        x, y, w, h = 60 + (i % 4) * 220, 80 + (i // 4) * 160, 120, 44
        元s.append(dict(句柄="E%d" % i, 框=(x, y, w, h), 心=(x + w // 2, y + h // 2),
                        字="钮%d" % i, 主色=(200, 200, 200), 形状="横条", 置信=0.9))
    态.写视觉(元s, 帧号=1, 变化率=0.0, 屏幕=(1920, 1080))
    return 元s


# ══════════════ A1 并发无阻塞 ══════════════
def A1():
    态 = 共存态(用gpu=False)
    错 = []
    停 = threading.Event()

    def 读():
        while not 停.is_set():
            try:
                态.取视觉(); 态.取动作(); 态.取意图(); 态.取听觉()
                态.取语义(); 态.取表达()
            except Exception as ex:
                错.append(repr(ex))

    def 写():
        i = 0
        while not 停.is_set():
            i += 1
            try:
                态.写视觉(_合成立体(态, 6), 帧号=i, 变化率=0.01)
                态.写动作(进度=0.5, 状态="移动中")
                态.写意图(当前=dict(句柄="E1"), 来源="自检")
                态.写听觉(主频=440.0, 能量=0.1, 起音=(i % 20 == 0))
            except Exception as ex:
                错.append(repr(ex))

    线s = [threading.Thread(target=读) for _ in range(4)] + \
          [threading.Thread(target=写) for _ in range(2)]
    for t in 线s:
        t.start()
    time.sleep(1.5)
    停.set()
    for t in 线s:
        t.join(timeout=2.0)
    挂 = [t.name for t in 线s if t.is_alive()]
    度量 = 态.取度量()
    return dict(通过=bool(not 错 and not 挂 and 度量["最长写ms"] < 50),
                错=错[:3], 未退出线程=挂, 最长写ms=度量["最长写ms"],
                读写次数=度量["读写次数"])


# ══════════════ A2 轨迹连续 ══════════════
def A2():
    态 = 共存态(用gpu=False)
    _合成立体(态, 8)
    手 = 手器(态, 演练=True)
    手._移光标((100, 100))                                # 演练模式：虚拟光标，不碰真屏
    手.投喂意图(dict(目标点=(900, 700), 来源="自检A2"))   # 经意图槽驱动 360° 自由控制器
    手._一拍()                                           # 闲→起手（读意图、进入「移动中」）
    for _ in range(600):
        手._推进()
        if 手._目标 is None:                             # 到达→收尾，控制器复位为「闲」
            break
    点s = [(float(x), float(y)) for x, y in 手._迹]       # 实际逐拍路径（非预采样点列）
    跳 = [float(np.hypot(点s[i + 1][0] - 点s[i][0], 点s[i + 1][1] - 点s[i][1]))
          for i in range(len(点s) - 1)]
    最大跳 = max(跳) if 跳 else 0.0
    终误差 = float(np.hypot(点s[-1][0] - 900, 点s[-1][1] - 700)) if 点s else 1e9
    return dict(通过=bool(len(点s) >= 6 and 最大跳 <= 120 and 终误差 <= 3),
                点数=len(点s), 首点=点s[0] if 点s else None,
                最大相邻跳变=round(最大跳, 1), 终点误差=round(终误差, 1))


# ══════════════ A3 中途纠偏 ══════════════
def A3():
    态 = 共存态(用gpu=False)
    _合成立体(态, 8)
    手 = 手器(态, 演练=True)
    手._移光标((100, 100))                                # 演练模式：虚拟光标，不碰真屏
    手.投喂意图(dict(句柄="E2", 目标点=(500, 300), 兜底=(500, 300), 来源="自检A3"))
    手._一拍()                                           # 闲→起手（跟踪 E2）
    旧目标 = tuple(float(v) for v in 手._目标) if 手._目标 is not None else None
    for _ in range(6):                                   # 飞行若干拍（尚未到达）
        手._推进()
    # 元素 E2 在动：把视觉槽里的心挪走（手每拍 _纠偏 读它最新位置）
    元s = 态.取视觉()["元素s"]
    for i, e in enumerate(元s):
        if e.get("句柄") == "E2":
            元s[i] = dict(e, 心=(1000, 800), 框=(1000, 800, 120, 44))
    态.写视觉(元s, 帧号=2)
    for _ in range(8):                                   # 纠偏后继续推进（途中改向）
        手._推进()
        if 手._目标 is not None and float(np.hypot(手._目标[0] - 1000.0,
                                                   手._目标[1] - 800.0)) < 2.5:
            break
    动 = 态.取动作()
    新目标 = tuple(float(v) for v in 手._目标) if 手._目标 is not None else None
    改 = bool(新目标 is not None and 旧目标 is not None and
              (abs(新目标[0] - 旧目标[0]) > 2.5 or abs(新目标[1] - 旧目标[1]) > 2.5))
    return dict(通过=bool(改 and int(动.get("纠偏次数", 0)) >= 1),
                旧终点=旧目标, 新终点=新目标, 纠偏次数=int(动.get("纠偏次数", 0)))


# ══════════════ A4 手忙脑不停 ══════════════
def A4():
    态 = 共存态(用gpu=False)
    _合成立体(态, 8)
    语 = 语义图(态)
    脑 = 脑器(态, 语)
    # 手在忙
    态.写动作(状态="移动中", 进度=0.4, 句柄="E3")
    脑.一拍()
    a = 脑.思维拍
    态.写动作(状态="移动中", 进度=0.7)
    脑.一拍()
    b = 脑.思维拍
    意 = 态.取意图()
    return dict(通过=bool(b > a and 意.get("当前") is not None),
                手状态=态.取动作().get("状态"), 思维拍=(a, b),
                已出意图=bool(意.get("当前") is not None))


# ══════════════ A5 HDC 语义 ══════════════
def A5():
    态 = 共存态(用gpu=False)
    语 = 语义图(态)
    c = 态.编码

    def 元素(h, 字, x, y, w=100, hh=40):
        return dict(句柄=h, 框=(x, y, w, hh), 心=(x + w // 2, y + hh // 2),
                    字=字, 主色=(10, 10, 10), 形状="横条", 置信=0.9)

    # 同名异效：两个都叫"确定"，但点击后产生的效果不同
    前 = [元素("A", "确定", 10, 10), 元素("B", "确定", 300, 10)]
    后A = [元素("A", "确定", 10, 10), 元素("B", "确定", 300, 10), 元素("X", "弹窗", 100, 200)]
    后B = [元素("A", "确定", 10, 10), 元素("B", "确定", 300, 10), 元素("Y", "菜单", 100, 200)]
    语.记点击("A", 前, 后A, 真实命中句柄="A")
    语.记点击("B", 前, 后B, 真实命中句柄="B")
    sA, sB = 语.取按钮("A"), 语.取按钮("B")
    同名异效分开 = bool(sA and sB and 语.取上层 is not None and
                        c.相似(sA["功能签名"], sB["功能签名"]) < 0.5)

    # 异名同效：两个不同名字，但效果一样
    前2 = [元素("C", "绿键", 10, 10), 元素("D", "蓝键", 300, 10)]
    后2 = 前2 + [元素("Z", "面板", 100, 200)]
    语.记点击("C", 前2, 后2, 真实命中句柄="C")
    语.记点击("D", 前2, 后2, 真实命中句柄="D")
    sC, sD = 语.取按钮("C"), 语.取按钮("D")
    异名同效归并 = bool(sC and sD and c.相似(sC["功能签名"], sD["功能签名"]) >= 0.8)

    # 层级：最小包含者
    层 = [元素("W", "", 0, 0, 900, 600), 元素("P", "", 60, 60, 300, 200),
          元素("Q", "小钮", 100, 100, 80, 30)]
    树 = 语.建容器树(层)
    q = next((it for it in 树 if it.get("句柄") == "Q"), {})
    父Q = q.get("父")
    层级正确 = bool(父Q == "P")

    # 叠加：据真实裁决收敛（初始未知）
    叠前 = [元素("U", "", 100, 100, 200, 120), 元素("V", "", 140, 130, 200, 120)]
    初 = 语.取上层("U", "V")
    for _ in range(6):
        语.记点击("U", 叠前, 叠前, 真实命中句柄="V")       # 世界裁决：V 在上
    末 = 语.取上层("U", "V")
    叠加收敛 = bool(str(初) != str(末) and 末 == "V")
    判 = 语.判据()
    return dict(通过=bool(同名异效分开 and 异名同效归并 and 层级正确 and 叠加收敛),
                同名异效分开=同名异效分开, 异名同效归并=异名同效归并,
                层级正确=层级正确, 叠加初=初, 叠加末=末, 叠加收敛=叠加收敛,
                判据=判)


# ══════════════ A6 非 Transformer 审计 ══════════════
def _剥注释字符串(源码: str) -> str:
    """用 tokenize 剥掉注释与字符串，只留可执行代码 token——避免把 docstring 里的
    『禁止 attention』这种声明文字误判成真的用了 attention。"""
    import io
    import tokenize
    out = []
    try:
        for tk in tokenize.generate_tokens(io.StringIO(源码).readline):
            if tk.type in (tokenize.COMMENT, tokenize.STRING):
                out.append(" ")
            else:
                out.append(tk.string)
    except Exception:
        return 源码
    return " ".join(out)


def A6():
    # 只认「代码结构」型用法：真导入、真调用、QKV 投影层——注释/docstring 不算
    禁式 = (r"import\s+transformers", r"from\s+transformers",
            r"\.softmax\s*\(", r"F\.softmax", r"nn\.MultiheadAttention",
            r"MultiheadAttention", r"scaled_dot_product_attention",
            r"\battention\s*\(", r"self_attn", r"q_proj", r"k_proj", r"v_proj",
            r"class\s+\w*Attention", r"nn\.Transformer", r"nn\.Linear")
    命中 = []
    _跳过 = ("__pycache__", "backup_", "备份_")
    _扫 = []
    for _根, _目录s, _文件s in os.walk(HERE):
        _目录s[:] = [d for d in _目录s if not d.startswith(_跳过)]
        for f in sorted(_文件s):
            if f.endswith(".py"):
                _扫.append(os.path.join(_根, f))
    for _p in sorted(_扫):
        f = os.path.relpath(_p, HERE)
        with open(_p, encoding="utf-8") as fh:
            码 = _剥注释字符串(fh.read())
        for pat in 禁式:
            if re.search(pat, 码):
                命中.append((f, pat))
    return dict(通过=bool(not 命中), 命中=命中,
                审计口径="tokenize 剥离注释/字符串后扫代码结构（递归含 实施/ 等子目录；跳过 backup_/__pycache__）",
                扫描文件=len(_扫))


# ══════════════ A7 可复现 ══════════════
def A7():
    a = 编码器(种子=3800, 用gpu=False)
    b = 编码器(种子=3800, 用gpu=False)
    x = np.arange(32, dtype=np.float32)
    ok = bool(np.array_equal(a.特征码(x), b.特征码(x)) and
              np.array_equal(a.文本码("保存新建撤销"), b.文本码("保存新建撤销")))
    return dict(通过=ok)


def 主():
    项s = [("A1并发无阻塞", A1), ("A2轨迹连续", A2), ("A3中途纠偏", A3),
           ("A4手忙脑不停", A4), ("A5HDC语义", A5), ("A6非Transformer审计", A6),
           ("A7可复现", A7)]
    果 = {}
    for 名, fn in 项s:
        t0 = time.time()
        try:
            果[名] = fn()
        except Exception as ex:
            果[名] = dict(通过=False, 异常=repr(ex))
        果[名]["耗时s"] = round(time.time() - t0, 2)
        print("[%s] %s  %s" % ("过" if 果[名].get("通过") else "未过", 名,
                              json.dumps({k: v for k, v in 果[名].items()
                                          if k != "通过"}, ensure_ascii=False,
                                         default=str)[:200]), flush=True)
    总 = all(v.get("通过") for v in 果.values())
    print("\n==== 自检总判：%s ====" % ("全过" if 总 else "有未过项"))
    with open(os.path.join(HERE, "自检结果.json"), "w", encoding="utf-8") as f:
        json.dump(dict(总过=总, 项=果), f, ensure_ascii=False, indent=1, default=str)
    return 总, 果


if __name__ == "__main__":
    主()
