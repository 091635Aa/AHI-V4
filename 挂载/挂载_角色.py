# -*- coding: utf-8 -*-
"""AHI4/挂载/挂载_角色.py —— ★套模板测试：把「角色包（人设+词语+听见句+记忆）」挂到内生体上。

设计（不加载任何外部模型权重，C10）：
  · 身体不动：D=64 两区室核（V5）继续跑，提供 o用（自我-世界状态）与 内心；
  · 挂载层 = 角色头：词音来自角色包（外采）；词义 = 「词 × 当期状态」共现绑定（内生学习）；
    词序 = 听见句二元统计（学习式）；记忆 = 角色包条目（外源，带来源标签，经归属门只读）。
  · 卸载 = 拆掉角色头，输出回到基座；换包 = 用同一状态序列换一套角色头 ⇒ 输出改变率可测。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_V5目录 = os.path.join(os.path.dirname(HERE), "meeting53", "实施")
sys.path.insert(0, _V5目录)
sys.path.insert(0, HERE)
import V5_双通道 as V5  # noqa: E402
import 角色包 as 包模块  # noqa: E402

# meeting58：移除模块级 np.seterr（import 即污染全进程）；本文件除法均有 +1e-9 保护
界幅 = V5.界幅
_归 = V5._归


class 角色头:
    """挂载层：把状态读成「这个角色会说的词」+ 按状态召回这个角色的记忆。"""

    def __init__(self, 包: dict, D: int = V5.D默认):
        self.包 = 包
        self.D = int(D)
        self.词表 = list(dict.fromkeys(包.get("词语列") or []))
        self.词id = {w: i for i, w in enumerate(self.词表)}
        self.M = len(self.词表)
        self.W词 = _归(np.zeros((self.M, self.D)))
        self.W词_内 = _归(np.zeros((self.M, self.D)))
        self.C2 = np.zeros((self.M, self.M))
        self.C首 = np.zeros(self.M)
        self.记忆 = list(包.get("记忆条目列") or [])
        self.台账 = dict(训练拍=0, 词绑定更新=0, 生成词数=0, 召回数=0)
        self.η = 0.02

    def _句(self, i: int):
        听见 = self.包.get("听见句列") or []
        if not 听见:
            return []
        return [w for w in str(听见[i % len(听见)]).split(" ") if w]

    def 训练(self, 状态列):
        for i, (_, o用, 内心) in enumerate(状态列):
            句 = self._句(i)
            上 = None
            for w in 句:
                if w not in self.词id:
                    continue
                wi = self.词id[w]
                self.W词[wi] = _归((1.0 - self.η) * self.W词[wi] + self.η * o用 / 界幅)
                self.W词_内[wi] = _归((1.0 - self.η) * self.W词_内[wi] + self.η * 内心 / 界幅)
                if 上 is not None:
                    self.C2[上, wi] += 1.0
                else:
                    self.C首[wi] += 1.0
                上 = wi
                self.台账["词绑定更新"] += 1
            self.台账["训练拍"] += 1
        return self

    def 生成(self, o用, 内心, 最多词: int = 4):
        if self.M == 0:
            return []
        u = _归(o用 / (float(np.max(np.abs(o用))) + 1e-9))
        范 = np.linalg.norm(self.W词, axis=1) + 1e-9
        sc = (self.W词 @ u) / 范
        内u = _归(内心 / (float(np.max(np.abs(内心))) + 1e-9))
        范内 = np.linalg.norm(self.W词_内, axis=1) + 1e-9
        sc = sc + 0.9 * ((self.W词_内 @ 内u) / 范内)
        sc = sc - float(np.mean(sc))
        sc = sc / (float(np.max(np.abs(sc))) + 1e-9)
        词序 = []
        上 = None
        for _ in range(int(最多词)):
            if 上 is None:
                分 = sc + 0.6 * (self.C首 / (self.C首.max() + 1e-9))
            else:
                bg = self.C2[上]
                if float(bg.sum()) <= 0:
                    break
                分 = 0.85 * sc + (bg / (bg.max() + 1e-9))
            w = int(np.argmax(分))
            if 上 is not None and float(分[w]) <= 0.05:
                break
            词序.append(self.词表[w])
            上 = w
            if len(词序) >= 最多词:
                break
        self.台账["生成词数"] += len(词序)
        return 词序

    def 召回(self, o用, 拍: int, 每拍: int = 6):
        if not self.记忆 or int(拍) % int(每拍) != int(每拍) - 1:
            return None
        桶 = int(abs(float(np.sum(o用))) * 1000.0) % max(1, len(self.记忆))
        idx = (桶 + int(拍) // int(每拍)) % len(self.记忆)
        self.台账["召回数"] += 1
        return dict(self.记忆[idx], 来源=self.记忆[idx].get("来源", "外源"))

    def 指纹(self) -> str:
        h = hashlib.sha256()
        for a in (self.W词, self.W词_内, self.C2, self.C首):
            h.update(np.asarray(a).tobytes())
        return h.hexdigest()[:24]


class 句池角色头(角色头):
    """must-fail 对照：直接把角色包第一句记忆当正文回放（不看状态）。"""
    def 生成(self, o用, 内心, 最多词: int = 4):
        if not self.记忆:
            return []
        return [self.记忆[0].get("文本", "")]


class 挂载器:
    """把角色包挂到（已有）双通道体上：只读 o用/内心，不改核动力学。"""

    def __init__(self, 体=None, D: int = V5.D默认, 种子: int = V5.种子默认):
        self.体 = 体 if 体 is not None else V5.双通道体(D=D, 种子=种子)
        self.头 = None
        self.已挂 = None
        self.台账 = dict(挂载次数=0, 卸载次数=0, 换包次数=0, 当前="")

    # ── 状态序列：同一身体、同一沙盒种子 ⇒ 所有包共用同一状态轨迹（保证「同状态换包」可比）──
    def 状态列(self, 拍数: int = 120, 种子: int = V5.种子默认):
        环 = V5.接口沙盒(种子=int(种子) + 1)
        列 = []
        for t in range(int(拍数)):
            环.新试次()
            self.体.一拍(观测id=环.cue, 学习=False, 老师=False, 拍=t, 探索率=0.0)
            列.append((int(环.cue), self.体.o用.copy(), self.体.内心.copy()))
        return 列

    def 挂载(self, 包: dict, 状态列, 头类=角色头):
        self.头 = 头类(包, D=self.体.D)
        self.头.训练(状态列)
        self.已挂 = 包.get("角色", "?")
        self.台账["挂载次数"] += 1
        self.台账["当前"] = self.已挂
        return self.头

    def 卸载(self):
        self.头 = None
        self.已挂 = None
        self.台账["卸载次数"] += 1
        self.台账["当前"] = ""

    def 跑(self, 包: dict, 状态列, 头类=角色头, 最多词: int = 4):
        头 = self.挂载(包, 状态列, 头类=头类)
        行 = []
        for t, (cue, o用, 内心) in enumerate(状态列):
            词 = 头.生成(o用, 内心, 最多词=最多词)
            回 = 头.召回(o用, t)
            行.append(dict(拍=t, 观测=cue, 词=词, 正文=" ".join(词), 回忆=回))
        return dict(角色=包.get("角色", "?"), 来源=包.get("来源", ""), 头指纹=头.指纹(),
                    行=行, 台账=dict(头.台账), 头=头)


def 演示(拍数: int = 24, 训练拍: int = 120):
    """直跑：同一状态序列上依次挂 洛天依 / MOSS / 打标日记，打印独白。"""
    器 = 挂载器()
    状态 = 器.状态列(拍数 + 训练拍)
    训练段, 测段 = 状态[:训练拍], 状态[训练拍:训练拍 + 拍数]
    果 = {}
    for 名, 包 in 包模块.全部包().items():
        r = 器.跑(包, 状态, 最多词=4)
        # 只用测段展示
        行 = r["行"][训练拍:训练拍 + 拍数]
        果[名] = dict(角色=r["角色"], 指纹=r["头指纹"],
                      独白=["%s%s" % (x["正文"], (" ｜想起：" + x["回忆"]["文本"]) if x["回忆"] else "") for x in 行],
                      召回=[x["回忆"]["文本"] for x in 行 if x["回忆"]])
    return 果


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 演示()
    for 名, r in 果.items():
        print("=" * 20, 名, "指纹", r["指纹"])
        for 行 in r["独白"]:
            print("  ", 行)
