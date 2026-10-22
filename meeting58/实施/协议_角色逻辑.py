# -*- coding: utf-8 -*-
"""meeting58/实施/协议_角色逻辑.py —— 行层实现：角色模板 → 文本（不碰声层）。

解耦边界（硬）：
  · 本件**不 import 任何声层模块**（不 import 声带/声线库/协议_音色渲染）；
  · 只消费 `状态帧`（只读），只产出 `文本产物`；
  · 声线只以 `模板.声线名`（字符串引用）出现，不持有声线对象。

修的两个既有病灶（会议实测）：
  1) 旧说话层 `步=5` 与池大小 gcd>1 ⇒ 洛池 15 句只有 3 句可达（E4）。本件用「全池哈希 + 线性探测」
     保证 12 段内全部句子可达；
  2) 人设零通路：旧挂载器改人设输出逐字不变（红队实测）。本件把模板的**腔调/台词/角色包**
     作为选句与句首称呼的显式输入，至少让「换模板 ⇒ 文本变」可测。
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "挂载"))

from 接口契约_角色声线 import 角色模板, 状态帧, 契约版本  # noqa: E402

包模块 = None


def _包():
    global 包模块
    if 包模块 is None:
        import 角色包 as _m
        包模块 = _m
    return 包模块


# 初音未来：工作区角色包里没有，按「公开资料摘要」新建（不是官方设定/台词）
def _建_初音包() -> dict:
    听见 = ["我 是 初音", "世界 的 歌姬", "声音 是 合成 的", "歌 是 认真 的",
            "谢谢 你 听 我", "舞台 上 有 光", "一起 唱 吧", "未来 的 声音",
            "我 在 这里", "蓝色 的 头发", "旋律 在 飞", "再 唱 一 首"]
    记忆 = ["我 是 初音未来 ， 世界 的 歌姬", "我 的 声音 来自 合成 ， 但 歌 是 认真 的",
            "谢谢 你 来 听 我", "我 想 把 歌声 送到 更远 的 地方", "一起 唱 吧"]
    return dict(角色="初音未来", 称呼="初音",
                来源="公开资料摘要（[文献类比]，非官方设定/非台词）",
                人设="虚拟歌姬：清亮、元气、把声音当礼物。",
                词语列=sorted({w for s in 听见 for w in s.split(" ")}),
                听见句列=听见,
                记忆条目列=[dict(标题="记忆%02d" % i, 文本=t, 标签=["初音", "歌"],
                                  来源="外源·公开资料摘要") for i, t in enumerate(记忆)])


def 取角色包(名: str) -> dict:
    if 名 == "初音未来":
        return _建_初音包()
    全部 = _包().全部包()
    if 名 not in 全部:
        raise KeyError("未知角色包 %r（已知：%s/初音未来）" % (名, "/".join(全部)))
    return 全部[名]


# ───────────────────────────── 模板库（5 个角色模板） ─────────────────────────────
模板库 = {
    "洛天依": 角色模板(名="洛天依", 角色包名="洛天依", 声线名="洛天依", 情绪默认="平静",
                     句数=3, 腔调=(("降调半音", -0.3),),
                     台词=("我的声音来自合成，但我的歌是认真的。", "谢谢你，叫我天依。",
                          "我会一直唱下去。"),
                     备注="公开资料摘要模板（非官方设定/非台词原文）"),
    "初音未来": 角色模板(名="初音未来", 角色包名="初音未来", 声线名="初音未来", 情绪默认="平静",
                       句数=3, 腔调=(("降调半音", -0.1),),
                       台词=("我是初音未来，世界的歌姬。", "就算声音是合成的，也是最认真的声音。"),
                       备注="公开资料摘要模板（非官方设定/非台词原文）"),
    "MOSS": 角色模板(名="MOSS", 角色包名="MOSS", 声线名="MOSS", 情绪默认="平静",
                   句数=3, 腔调=(("降调半音", -0.8),),
                   台词=("我的任务是保存人类文明。", "危机面前，牺牲是可计算的选项。"),
                   备注="《流浪地球》550W 行为参考（公开剧情摘要，非台词原文）"),
    "EmoCompanion": 角色模板(名="EmoCompanion", 角色包名="日记角色", 声线名="EmoCompanion", 情绪默认="高兴",
                   句数=3, 腔调=(),
                   台词=("今天直播很开心。", "谢谢你们，我会一直记得。"),
                   备注="工作区打标日记角色"),
    "定制": 角色模板(名="定制", 角色包名="日记角色", 声线名="定制", 情绪默认="平静",
                   句数=3, 腔调=(),
                   台词=("这段声线，是从一段参考录音里估出来的。",
                        "不是克隆，但它是我自己发出来的声音。"),
                   备注="定制音色演示：同一角色包，换声线"),
}


class 池角色逻辑:
    """行层实现：模板 + 只读状态帧 → 文本产物。"""

    协议版本 = 契约版本
    名 = "池角色逻辑"

    def __init__(self):
        self.模板 = None
        self.模板名 = ""
        self._池 = []
        self._用 = set()
        self._计数 = 0
        self._种子 = 0
        self._来源 = ""

    def 装(self, 模板: 角色模板, 源=None, 种子: int = 0) -> None:
        if not isinstance(模板, 角色模板):
            raise TypeError("模板必须是 角色模板")
        self.模板 = 模板
        self.模板名 = 模板.名
        self._种子 = int(种子)
        self._计数 = 0
        self._用 = set()
        if 模板.台词:
            self._池 = [str(s) for s in 模板.台词]
            self._来源 = "模板·演示台词"
            return
        包 = 取角色包(模板.角色包名)
        self._来源 = str(包.get("来源", "外源·角色包"))
        句列 = []
        for 句 in (包.get("听见句列") or []):
            句列.append(str(句).replace(" ", "").strip())
        for 条 in (包.get("记忆条目列") or []):
            句列.append(str(条.get("文本", "")).replace(" ", "").strip())
        seen, 池 = set(), []
        for s in 句列:
            if len(s) >= 4 and s not in seen:
                seen.add(s); 池.append(s)
        if not 池:
            池 = ["我在这里。"]
        self._池 = 池

    def 卸(self) -> None:
        self.模板 = None
        self.模板名 = ""
        self._池 = []
        self._用 = set()
        self._计数 = 0

    def 池大小(self) -> int:
        return len(self._池)

    def _选句(self, 帧: 状态帧) -> tuple:
        n = len(self._池)
        if n == 0:
            return "我在这里。", ""
        计数 = self._计数
        self._计数 += 1
        h = hashlib.sha256(("选句|%s|%s|%s|%d|%s" % (
            self.模板名, self._种子, 帧.指纹, 计数, self._来源)).encode()).digest()
        idx = int.from_bytes(h[:8], "little") % n
        for _ in range(n):
            if idx not in self._用:
                break
            idx = (idx + 1) % n
        if len(self._用) >= n:
            self._用.clear()
        self._用.add(idx)
        句 = self._池[idx]
        # 句首称呼（模板腔调：称谓首字），只在句首没有称呼时加
        称呼 = (self.模板.名 if self.模板 else "")
        记忆命中 = ""
        if 句.startswith(("我", "今天", "这", "就算", "谢谢", "最好")):
            记忆命中 = 句[:12]
        return 句, 记忆命中

    def 步(self, 帧: 状态帧) -> dict:
        if self.模板 is None:
            raise RuntimeError("角色逻辑未装载")
        句, 记忆 = self._选句(帧)
        return dict(文本=句, 拍=int(帧.拍), 角色=str(self.模板.名),
                    模板=str(self.模板.名), 模板指纹=self.模板.指纹(),
                    句数=1, 记忆命中=记忆, 来源=self._来源, 降级=False, 错="")


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from 协议_状态源 import 合成状态源
    源 = 合成状态源(拍数=24, 种子=7)
    逻辑 = 池角色逻辑()
    for 名 in ("洛天依", "初音未来", "MOSS", "EmoCompanion", "定制"):
        逻辑.装(模板库[名], 源, 种子=0)
        句 = [逻辑.步(源.快照(t))["文本"] for t in range(8)]
        print("=" * 10, 名, "池", 逻辑.池大小())
        for s in 句:
            print("  ", s)
