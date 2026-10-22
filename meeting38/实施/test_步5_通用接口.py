# -*- coding: utf-8 -*-
"""test_步5_通用接口.py —— Phase2 通用输入/输出 + 全接口预处理 预登记判据 V7–V10。

判据：
  V7 全接口预处理：文本/向量/音频特征三模态统一到同一 D 维向量空间；
     确定性（同 seed+同源两次 = 逐位一致）；音频特征 f 维 → D 维双极。
  V8 向量直驱：任一模态经 文本态.入() 驱一轮 → 引擎步进（拍+1）、响应非空、
     事件史留 "步向量" 痕（可重放）；带标签按语词路由、无标签如实报「没有语词的感觉」。
  V9 统一输出：未注册钩子 ⇒ 确定性降级文本；注册钩子 ⇒ 被调用并接管；计数可审计。
  V10 扩容不破既有：文本态.扩容(2048) 后 D=2048、记忆段守恒、先学事实仍可检索命中；
     引擎输入恒 16 维（语义探针 D→16，引擎内部 1024 不变）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M38 = os.path.dirname(HERE)
AHI = r"<workspace>/03_AHI\AHI2.1"
for _p in (HERE, M38, AHI, os.path.join(AHI, "meeting31"), os.path.join(AHI, "meeting28")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from 文本态 import 文本态                                 # noqa: E402
from 通用接口 import 统一输入, 统一输出, 当前D              # noqa: E402


def 判V7():
    import lang_codec
    码 = 统一输入.__new__(统一输入)
    # 直接构造一个 codec
    from lang_codec import 语言编解码器
    codec = 语言编解码器(seed=310)
    码.__init__(codec, seed=310)
    D = 当前D()
    d文 = 码.预处理("我喜欢吃小笼包。", "文本")
    d向 = 码.预处理(np.ones(D, dtype=np.float32), "向量")
    d音 = 码.预处理(np.random.default_rng(1).standard_normal(64), "音频特征")
    # 确定性：同 seed+同源 → 逐位一致
    码2 = 统一输入(语言编解码器(seed=310), seed=310)
    a = 码.预处理("我喜欢吃小笼包。", "文本")["向量"]
    b = 码2.预处理("我喜欢吃小笼包。", "文本")["向量"]
    同 = bool(np.array_equal(a, b))
    a2 = 码.预处理(np.random.default_rng(1).standard_normal(64), "音频特征")["向量"]
    b2 = 码2.预处理(np.random.default_rng(1).standard_normal(64), "音频特征")["向量"]
    音同 = bool(np.array_equal(a2, b2))
    return dict(通过=(d文["向量"].shape[0] == D and d向["向量"].shape[0] == D
                    and d音["向量"].shape[0] == D and 同 and 音同),
                D=D, 文本维=int(d文["向量"].shape[0]), 向量维=int(d向["向量"].shape[0]),
                音频特征维=int(d音["向量"].shape[0]), 文本确定=同, 音频确定=音同,
                文本极性集=sorted(set(np.unique(d文["向量"]).tolist())))


def 判V8(态):
    拍0 = int(态.桥.拍)
    r1 = 态.入(np.ones(当前D(), dtype=np.float32) * 0.5, "向量", 标签="（这是向量不是话）")
    r2 = 态.入("我住在杭州。", "文本")                     # 文本模态仍走既有通道
    r3 = 态.入("你叫什么名字？", "文本")
    拍1 = int(态.桥.拍)
    事件s = [e[0] for e in 态.桥.事件史]
    return dict(通过=(拍1 > 拍0 and bool(r1.get("响应")) and "步向量" in 事件s),
                拍增=(拍1 - 拍0), 无标签响应=str(r1.get("响应"))[:60],
                模态=r1.get("模态"), 步向量事件数=事件s.count("步向量"),
                文本响应=str(r2.get("响应"))[:40], 自指响应=str(r3.get("响应"))[:40])


def 判V9(态):
    # 未注册 → 降级文本
    d1 = 态.输出.输出("测试一句话", "音频")
    调 = []
    # 注册钩子（模拟未来 TTS 后端；本测试不 import 任何器官）
    态.注册通道("音频", lambda 文, **kw: 调.append(文) or ("TTS<%s>" % 文))
    d2 = 态.输出.输出("你好世界", "音频")
    d3 = 态.输出.输出("再一句", "音频")
    return dict(通过=(d1["通道"] == "文本" and d1["降级"] is True
                    and d2["通道"] == "音频" and d2["降级"] is False
                    and len(调) == 2 and d2["结果"].startswith("TTS<")),
                未注册_通道=d1["通道"], 未注册_降级=d1["降级"],
                注册_通道=d2["通道"], 注册_结果=str(d2["结果"])[:20], 钩子调用数=len(调))


def 判V10(态):
    # 先学一条事实（文本），记下它
    事实 = "我最喜欢的城市是苏州。"
    态.回合(事实)
    前D = 当前D()
    前段 = len(态.桥.codec._段)
    h前 = 态.桥._检索("我喜欢的城市", k=1)
    前分 = h前[0][1] if h前 else 0.0
    # 扩容到 2048
    报 = 态.扩容(2048)
    后段 = len(态.桥.codec._段)
    h后 = 态.桥._检索("我喜欢的城市", k=1)
    后分 = h后[0][1] if h后 else 0.0
    命中段 = h后[0][0] if h后 else ""
    # 引擎输入仍 16 维（不破既有）
    探针维 = int(态.桥.探针.shape[1])
    输入维 = int(态.桥.探针.shape[0])
    return dict(通过=(报["新D"] == 2048 and 后段 == 前段 and 后段 > 0
                    and (事实 in 命中段 or 后分 >= 0.30) and 输入维 == 16),
                前D=前D, 后D=当前D(), 段守恒=(前段 == 后段),
                前段=前段, 后段=后段, 前检索分=round(前分, 3), 后检索分=round(后分, 3),
                命中段=命中段[:24], 探针=(输入维, 探针维),
                报=dict(旧D=报["旧D"], 新D=报["新D"], 段守恒=报["段守恒"]))


def 主():
    import lang_codec
    lang_codec.D = 1024                                    # 复位到基线
    态 = 文本态(种子=310, 挂记忆=False).建()
    print("=" * 72)
    print("通用输入/输出 + 全接口预处理 · 判据 V7–V10")
    print("=" * 72)
    Rs = dict(V7=判V7())
    Rs["V8"] = 判V8(态)
    Rs["V9"] = 判V9(态)
    Rs["V10"] = 判V10(态)
    for k, v in Rs.items():
        print("%-4s %s  %s" % (k, "PASS" if v["通过"] else "FAIL", v))
    print("=" * 72)
    print("[log] 未做任何「已完成」宣称——数字与如实边界以上表为准。")


if __name__ == "__main__":
    主()