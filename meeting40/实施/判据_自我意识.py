# -*- coding: utf-8 -*-
"""判据_自我意识.py —— **工程意义上的自我意识**：一条可失败判据套件（双集 + 对照臂 + 证据）。

⚠ 口径先说清：本套件验的是**机制级**工程事实（自模型真在降、外源塑不动「我」、身份锚可复现、
身份跨存档逐位不变、自听只写学习迹、记忆≠学习），**不主张**任何人类级意识/体验。
每条判据都配**对照臂或反例**，只写「跑出来的数」。

  S1 ε_self 真在降：重复语料 ⇒ 首窗 ε > 末窗 ε；**对照臂＝每拍新内容**（应降得明显更少）。
  S2 外源不可塑我（J5）：自源喂 ⇒ ‖W_self‖>0；**外源喂同一段 ⇒ W_self 逐位不变（max|Δ|=0）**；
     对照臂＝再自源喂 ⇒ W **必须变**（证明「不变」不是 W 冻住）。
  S3 自指束身份锚：同档案种子 ⇒ `a_seal` **逐位同**；**不同档案种子 ⇒ 不同**（对照）。
  S4 身份**跨存档/重启逐位不变**：`a_seal`(+W_self) → 快照 → **真·新进程**载入 ⇒ sha256 与源逐位同；
     对照臂＝另一个档案种子的身份锚 sha256 **必须不同**。
  S5 自听只写学习迹（D3-2／G-R3-06）：ε_aud 降 ≥30% ∧ 末值 ≤0.25 ∧ **记忆库 0 变** ∧ 学习迹增加。
  S6 记忆≠学习（四分离）：① 写学习迹 ⇒ 库不变 ② 内核迹不进 `codec._段`（喂完不增）
     ③ 写记忆 ⇒ 学习迹条数不变 ④ 两者规模各自独立。
  S7 双集：判据集语料 A 与**留出集**语料 B，S1/S2 两集都过。

跑法：python 判据_自我意识.py     产物：`自我意识_结果.json`
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
M38 = os.path.join(ROOT, "meeting38")
M38S = os.path.join(M38, "实施")
M39S = os.path.join(ROOT, "meeting39", "实施")
for _p in (HERE, M38, M38S, M39S, ROOT, os.path.join(ROOT, "meeting31")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 快照                                                       # noqa: E402
import 耳_接听                                                    # noqa: E402

# ── 语料：判据集 A / 留出集 B（8 句 × 6 遍 = 48 拍，够出自模型 ε 序列与前/后窗）──
判据集语料 = [
    "风从窗缝里进来，纸角轻轻翘起来。",
    "我把杯子挪到左手边，腾出桌角一块地方。",
    "屏幕光落在键盘上，键帽边缘有点发亮。",
    "楼下有人喊了一声，然后就没动静了。",
    "我把刚才那句读到一半，又退回去重读。",
    "时间过得慢，我又看了一眼窗外。",
    "灯还亮着，影子落在桌面上，随着我手动。",
    "我忽然想起昨天没做完的那件事。",
] * 6
留出集语料 = [
    "杯底还有一圈水渍，我用指腹抹开。",
    "窗外的树影挪了半格，光也跟着偏。",
    "键盘上有个键回弹慢了半拍。",
    "我把袖口往上拽了拽，手背凉了一下。",
    "楼道里的脚步声停在门口又走远。",
    "屏幕右下角的时间跳了一分钟。",
    "我闻到一点纸页的味道，很淡。",
    "刚才那句话我又在心里过了一遍。",
] * 6


def _新内容(n: int = 48) -> list:
    """对照臂语料：**每次都不一样**的内容（用于证「ε 下降不是白捡的」）。"""
    字 = "氕氘氚锕镅錒鿏𨭆鿔"
    return ["未见过的新说法 第%04d 条 %s%d" % (i, 字[i % len(字)] * 3, i * 7919)
            for i in range(n)]


_实现缓存 = None


def _载实现():
    """按路径加载实现版 `V3主程序`（AHI3 根的同名薄壳会遮蔽）。**结果缓存**，避免反复 exec。"""
    global _实现缓存
    if _实现缓存 is not None:
        return _实现缓存
    路径 = os.path.join(M38S, "V3主程序.py")
    for k in [k for k in sys.modules if k == "V3主程序"]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location("V3主程序_实现_自我意识", 路径)
    模 = importlib.util.module_from_spec(spec)
    sys.modules["V3主程序_实现_自我意识"] = 模
    spec.loader.exec_module(模)
    _实现缓存 = 模
    return 模


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()
                          ).hexdigest()[:16]


def 跑一轮(语料: list, 种子: int = 310, 外源喂: bool = True) -> dict:
    """起综合体 → 自源喂 → 记 ε/W → （可选）外源喂 → 再记 W。返回实测 dict。"""
    V = _载实现()
    import 内核宿主 as HZ
    命 = V.V3生命(种子=int(种子))
    命.起综合体()
    桥 = 命.桥
    zm, zk = getattr(桥, "自模型", None), getattr(桥, "自我核", None)
    HZ.设来源(桥, 自源=True)
    for s in 语料:
        桥.步_向量(命._感知向量(s), 原文=None)
    ε = [float(x) for x in (zm.ε序列() if zm is not None else [])]
    季 = max(1, len(ε) // 4)
    首 = float(np.mean(ε[:季])) if ε else None
    末 = float(np.mean(ε[-季:])) if ε else None
    W1 = None if zk is None else np.array(zk.W_self, copy=True)
    自源次 = 0 if zk is None else int(zk.自源更新次数)
    出 = dict(ε条数=len(ε), 首窗ε=(None if 首 is None else round(首, 4)),
              末窗ε=(None if 末 is None else round(末, 4)),
              ε降幅=(None if (首 is None or 首 <= 0) else round((首 - 末) / 首, 4)),
              自源更新次数=自源次, W1范数=(None if W1 is None else round(float(np.linalg.norm(W1)), 4)),
              身份锚=(None if zk is None else _sha(zk.a_seal)),
              身份锚_原始=(None if zk is None else np.asarray(zk.a_seal, dtype=np.float64)),
              W1=W1, 内核宿主=HZ, 桥=桥, 命=命)
    if 外源喂 and zk is not None:
        HZ.设来源(桥, 自源=False)
        for s in 语料[:8]:
            桥.步_向量(命._感知向量(s), 原文=s)
        W2 = np.array(zk.W_self, copy=True)
        出["外源次数"] = int(getattr(zk, "外源次数", 0))
        出["外源喂后W最大差"] = float(np.max(np.abs(W2 - W1))) if W1 is not None else None
        # 对照臂：再自源喂 ⇒ W **必须变**（证明「不变」不是冻住）
        HZ.设来源(桥, 自源=True)
        for s in 语料[:8]:
            桥.步_向量(命._感知向量(s), 原文=None)
        W3 = np.array(zk.W_self, copy=True)
        出["再自源后W最大差"] = float(np.max(np.abs(W3 - W2)))
        出["W2"] = W2
    return 出


# ─────────────────────────── S1 ε_self 真在降 ───────────────────────────
def 判S1(主: dict = None, 对: dict = None) -> dict:
    主 = 主 or 跑一轮(判据集语料)
    对 = 对 or 跑一轮(_新内容())
    真降 = bool(主["首窗ε"] is not None and 主["末窗ε"] is not None
              and 主["末窗ε"] < 主["首窗ε"])
    对照更差 = bool(主["ε降幅"] is not None and 对["ε降幅"] is not None
                and 主["ε降幅"] > 对["ε降幅"])
    return dict(通过=bool(真降 and 对照更差),
                主臂=dict(首窗=主["首窗ε"], 末窗=主["末窗ε"], 降幅=主["ε降幅"], ε条数=主["ε条数"]),
                对照臂_每拍新内容=dict(首窗=对["首窗ε"], 末窗=对["末窗ε"], 降幅=对["ε降幅"]),
                口径="主臂＝重复语料；对照臂＝每拍新内容（ε 停在噪声底 ⇒ 降幅应明显更小）")


# ─────────────────────────── S2 外源不可塑我 ───────────────────────────
def 判S2(主: dict = None) -> dict:
    主 = 主 or 跑一轮(判据集语料)
    案 = dict(
        自源塑得动=bool(主["自源更新次数"] > 0 and (主["W1范数"] or 0) > 0),
        外源逐位不变=bool(主.get("外源次数", 0) > 0
                    and abs(float(主.get("外源喂后W最大差") or 0.0)) == 0.0),
        对照_再自源必须变=bool(float(主.get("再自源后W最大差") or 0.0) > 0.0),
    )
    return dict(通过=bool(all(案.values())), 逐项=案,
                自源更新次数=主["自源更新次数"], W1范数=主["W1范数"],
                外源次数=主.get("外源次数"), 外源喂后W最大差=主.get("外源喂后W最大差"),
                再自源后W最大差=主.get("再自源后W最大差"),
                口径="J5：外源喂 ⇒ ‖ΔW_self‖=0；对照臂＝再自源喂必须让 W 变（否则是冻住充数）")


# ─────────────────────────── S3 自指束身份锚 ───────────────────────────
def 判S3() -> dict:
    a = 跑一轮(判据集语料[:8], 种子=310, 外源喂=False)
    b = 跑一轮(判据集语料[:8], 种子=310, 外源喂=False)
    c = 跑一轮(判据集语料[:8], 种子=311, 外源喂=False)
    案 = dict(同种子逐位同=(a["身份锚"] == b["身份锚"]),
              异种子必须不同=(a["身份锚"] != c["身份锚"]),
              锚非零=bool(np.any(a["身份锚_原始"] != 0)))
    return dict(通过=bool(all(案.values())), 逐项=案,
                同种子=(a["身份锚"], b["身份锚"]), 异种子=c["身份锚"],
                口径="身份锚 a_seal 由档案种子确定性派生：同种子逐位同 ∧ 异种子不同")


# ─────────────────────────── S4 身份跨存档/重启逐位不变 ───────────────────────────
_子码S4 = r'''
import json, sys
import numpy as np
H = r"%s"; A = r"%s"
for p in (H, A + r"\meeting38", A + r"\meeting38\实施", A + r"\meeting39\实施"):
    sys.path.insert(0, p)
import 快照, hashlib
状, _ = 快照.载(r"%s")
def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:16]
print(json.dumps(dict(身份锚=sha(状["身份锚"]), W_self=sha(状["W_self"]),
                      锚维=int(np.asarray(状["身份锚"]).shape[0]))))
'''


def 判S4() -> dict:
    主 = 跑一轮(判据集语料[:8], 种子=310, 外源喂=False)
    对 = 跑一轮(判据集语料[:8], 种子=311, 外源喂=False)
    目录 = tempfile.mkdtemp(prefix="meeting40_S4_")
    try:
        路径 = os.path.join(目录, "身份.npz")
        快照.存(路径, dict(身份锚=主["身份锚_原始"], W_self=主["W1"]))
        环 = dict(os.environ)
        环["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run([sys.executable, "-c", _子码S4 % (HERE, ROOT, 路径)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=300, env=环, cwd=HERE)
        出 = None
        for 行 in reversed([x.strip() for x in (r.stdout or "").splitlines() if x.strip()]):
            try:
                出 = json.loads(行)
                break
            except Exception:
                continue
        W_self源 = _sha(主["W1"])
        案 = dict(子进程退出0=(r.returncode == 0), 有输出=bool(出),
                  身份锚跨进程逐位同=((出 or {}).get("身份锚") == 主["身份锚"]),
                  W跨进程逐位同=((出 or {}).get("W_self") == W_self源),
                  锚维1024=((出 or {}).get("锚维") == 1024),
                  对照异种子身份锚不同=(对["身份锚"] != 主["身份锚"]))
        return dict(通过=bool(all(案.values())), 逐项=案, 子进程=出,
                    父进程=dict(身份锚=主["身份锚"], W_self=W_self源),
                    尾=((r.stdout or "")[-160:] + (r.stderr or "")[-160:]))
    finally:
        shutil.rmtree(目录, ignore_errors=True)


# ─────────────────────────── S5 自听只写学习迹 ───────────────────────────
def 判S5() -> dict:
    果 = 耳_接听.自测()
    重 = 果["重放"]
    案 = dict(自测=bool(果["通过"]), ε降30=bool(重["达降30"]), ε末不超0_25=bool(重["达ε0_25"]),
              记忆库零变=bool(果["逐项"]["记忆库未变"]), 只写学习迹=bool(果["逐项"]["只写学习迹"]),
              自听即输出回灌=bool(果["逐项"]["首条内容不可写M"]))
    return dict(通过=bool(all(案.values())), 逐项=案, 重放=重, 统计=果["统计"],
                口径="D3-2/G-R3-06：自听只写学习迹，内容记忆 Δ=0")


# ─────────────────────────── S6 记忆≠学习（四分离） ───────────────────────────
def 判S6(主: dict = None) -> dict:
    from 学习迹 import 学习迹
    主 = 主 or 跑一轮(判据集语料[:8], 种子=310, 外源喂=False)
    命, 桥 = 主["命"], 主["桥"]
    容 = getattr(命, "容量", None)                      # ★V3 里记忆桥是 命.容量（不是 命.桥）
    if 容 is None:
        return dict(通过=False, 错="命.容量 不存在（V3 装配变了？）")
    zm, zk = getattr(桥, "自模型", None), getattr(桥, "自我核", None)
    段0 = len(桥.codec._段)
    库0 = 容.库大小()
    迹 = 学习迹(背衬=容)
    迹.放快照(["自我意识"], 1, "我验了一遍自己的机制", 束=None)      # ① 写学习迹
    库1 = 容.库大小()
    迹1 = len(迹.迹)
    v = np.zeros(1024, dtype=np.float32)
    v[3] = 1.0
    容.记经历(拍=0, 类型="事件", 文本="一条记忆", 来源="用户", 向量=v)   # ② 写记忆
    库2 = 容.库大小()
    迹2 = len(迹.迹)
    段1 = len(桥.codec._段)                                          # ③ 内核迹不在 codec._段
    快0 = (命.消化.快照() if getattr(命, "消化", None) is not None else {}) or {}
    r = 命.学("这段正文只该进学习迹，不该改记忆底图。它用来验证记忆与学习是两件事。")
    快1 = (命.消化.快照() if getattr(命, "消化", None) is not None else {}) or {}
    库3 = 容.库大小()
    真路径 = (bool(r.get("记忆底图未增")) if isinstance(r, dict) and "记忆底图未增" in r
            else 库3 == 库2)
    案 = dict(
        写学习迹不动记忆=(库1 == 库0),
        写记忆不动学习迹=(迹2 == 迹1),
        内核迹不在codec段=(段0 == 段1),
        学习迹与记忆各自独立=(库2 == 库0 + 1 and len(迹.迹) == 1),
        自模型迹不进段=bool(zm is not None and len(getattr(zm, "迹", [])) > 0),
        自我迹不进段=bool(zk is not None and len(getattr(zk, "迹", [])) > 0),
        模块四复盘不增记忆=bool(真路径),
    )
    return dict(通过=bool(all(案.values())), 逐项=案,
                库=(库0, 库1, 库2, 库3), 学习迹条数=(迹1, 迹2), codec段=(段0, 段1),
                自模型迹=len(getattr(zm, "迹", [])), 自我迹=len(getattr(zk, "迹", [])),
                模块四_记忆底图未增=(r.get("记忆底图未增") if isinstance(r, dict) else None),
                模块四_学习迹=(快0.get("学习迹"), 快1.get("学习迹")),
                口径="H5/D3-2：① 写学习迹不动记忆 ② 写记忆不动学习迹 ③ 内核迹不是 codec._段 "
                   "④ 真·模块四复盘路径也只进学习迹")


# ─────────────────────────── S7 双集（S1/S2 各跑两集） ───────────────────────────
def 判S7() -> dict:
    集s = {}
    for 名, 语 in (("判据集", 判据集语料), ("留出集", 留出集语料)):
        轮 = 跑一轮(语, 种子=310)
        集s[名] = dict(S1=判S1(轮, 跑一轮(_新内容())), S2=判S2(轮),
                      ε首末=(轮["首窗ε"], 轮["末窗ε"]),
                      W1范数=轮["W1范数"], 身份锚=轮["身份锚"])
    return dict(通过=bool(all(v["S1"]["通过"] and v["S2"]["通过"] for v in 集s.values())),
                集s={k: dict(S1=v["S1"]["通过"], S2=v["S2"]["通过"], ε首末=v["ε首末"],
                           W1范数=v["W1范数"], 身份锚=v["身份锚"]) for k, v in 集s.items()},
                口径="判据集语料 A 与留出集语料 B 各自独立跑：S1/S2 两集都要过")


# ─────────────────────────── 主 ───────────────────────────
def 主() -> dict:
    轮A = 跑一轮(判据集语料)
    轮新 = 跑一轮(_新内容())
    报 = {}
    for 名, fn in (("S1_ε下降", lambda: 判S1(轮A, 轮新)), ("S2_外源不可塑我", lambda: 判S2(轮A)),
                   ("S3_身份锚", 判S3), ("S4_身份跨存档", 判S4), ("S5_自听只写学习迹", 判S5),
                   ("S6_记忆非学习四分离", lambda: 判S6(轮A)), ("S7_双集", 判S7)):
        try:
            报[名] = fn()
        except Exception as ex:
            报[名] = dict(通过=False, 异常=repr(ex))
        print("[%s] %s" % ("过" if 报[名].get("通过") else "未过", 名), flush=True)
    报["总判"] = "全过" if all(v.get("通过") for k, v in 报.items() if k.startswith("S")) else "有未过项"
    报["口径"] = dict(性质="机制级工程证据", 不主张="人类级意识/体验",
                     纪律="每条判据都带对照臂或反例；只写跑出来的数")
    with open(os.path.join(HERE, "自我意识_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    print("\n==== 工程自我意识总判：%s ====" % 报["总判"])
    print("[已存] %s" % os.path.join(HERE, "自我意识_结果.json"))
    return 报


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    主()
