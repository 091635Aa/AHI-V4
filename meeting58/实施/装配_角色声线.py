# -*- coding: utf-8 -*-
"""meeting58/实施/装配_角色声线.py —— 总装：状态源 × 行层 × 声层，按句边界热切换。

用法（演示 / 判据都走这里）：
  装配().说角色("洛天依", 句数=2, 起始拍=0)  → 两条渲染产物
  装配().热切换("MOSS")                      → 下一句生效（句边界栅栏）
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from 接口契约_角色声线 import 角色模板, 契约版本  # noqa: E402
import 声线库  # noqa: E402
import 协议_角色逻辑 as 行层  # noqa: E402
import 协议_状态源 as 状态层  # noqa: E402
import 协议_音色渲染 as 声层  # noqa: E402
from 注册表_挂载 import 注册表_挂载  # noqa: E402

产物 = os.path.join(HERE, "产物", "装配")


class 装配_角色声线:
    协议版本 = 契约版本

    def __init__(self, 状态源=None, 渲染器=None):
        self.源 = 状态源 or 状态层.双通道状态源(种子=53001, 拍数=240)
        self.渲染器 = 渲染器 or 声层.声带渲染器()
        self.表 = 注册表_挂载()
        os.makedirs(产物, exist_ok=True)
        # 状态源
        self.表.注册("状态源", "双通道", lambda: self.源, 版本="1.0")
        self.表.试跑("状态源", "双通道", 探针=lambda p: p.快照(0).指纹)
        self.表.切换("状态源", "双通道")
        # 行层：5 个角色模板
        for 名, 模板 in 行层.模板库.items():
            def _工厂(t=模板):
                return 行层.池角色逻辑()
            self.表.注册("角色逻辑", 名, _工厂, 版本=模板.版本)
            self.表.试跑("角色逻辑", 名,
                        探针=lambda p, t=模板: (p.装(t, self.源, 种子=0), hasattr(p, "步"))[1])
        # 声层：渲染器 + 5 个声线档案
        self.表.注册("音色渲染", "原生声带", lambda: self.渲染器, 版本="1.0")
        self.表.试跑("音色渲染", "原生声带",
                    探针=lambda p: p.渲染("自检。", 声线=声线库.取("中性"), 写wav=False)["通过"])
        for 名 in 声线库.全部():
            self.表.注册("声线档案", 名, lambda n=名: 声线库.取(n), 版本="1.0",
                        元数据=dict(来源=声线库.取(名).来源))
            self.表.试跑("声线档案", 名, 探针=lambda p: p.指纹())
        self.表.切换("音色渲染", "原生声带")
        self.表.切换("声线档案", "中性")
        self.当前角色 = ""
        self.当前句 = 0

    # ── 热切换（句边界）──
    def 切换角色(self, 名: str) -> dict:
        if 名 not in 行层.模板库:
            raise KeyError("未知角色模板 %r" % 名)
        r = self.表.切换("角色逻辑", 名, 栅栏="句边界")
        self.当前角色 = 名
        return r

    def 切换声线(self, 名: str) -> dict:
        self.表.切换("声线档案", 名)
        return dict(声线=名)

    def 说一句(self, 模板名: str | None = None, 拍: int | None = None,
             情绪: str | None = None, 写wav: bool = True) -> dict:
        模板名 = 模板名 or self.当前角色
        if not 模板名:
            raise RuntimeError("未指定角色")
        if self.表.表现()["当前"].get("角色逻辑") != 模板名:
            self.切换角色(模板名)
        模板 = 行层.模板库[模板名]
        拍 = self.当前句 if 拍 is None else int(拍)
        self.当前句 = int(拍) + 1
        帧 = self.源.快照(拍)
        调 = self.表.调用("角色逻辑", "步", 帧)
        if 调["降级"]:
            return dict(降级=True, 错=调["错"], 文本="", 拍=拍)
        文本产物 = 调["结果"]
        声线 = 声线库.取(模板.声线名)
        声线.校验()
        r = self.表.调用("音色渲染", "渲染", 文本产物["文本"], 声线=声线,
                        情绪=情绪 or 模板.情绪默认, 状态=帧,
                        腔调=模板.腔调, 写wav=写wav)
        if r["降级"]:
            return dict(降级=True, 错=r["错"], 文本=文本产物["文本"], 拍=拍)
        out = r["结果"]
        out["角色"] = 模板名
        out["模板指纹"] = 模板.指纹()
        out["文本产物"] = 文本产物
        return out

    def 说角色(self, 名: str, 句数: int = 2, 起始拍: int = 0,
             写wav: bool = True) -> list:
        self.切换角色(名)
        模板 = 行层.模板库[名]
        列 = []
        for i in range(int(句数)):
            r = self.说一句(模板名=名, 拍=起始拍 + i, 写wav=写wav)
            r["角色"] = 名
            r["句序"] = i
            r["声线名"] = 模板.声线名
            列.append(r)
        return 列


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    器 = 装配_角色声线()
    for 名 in ("洛天依", "初音未来", "MOSS", "EmoCompanion", "定制"):
        列 = 器.说角色(名, 句数=1, 起始拍=0)
        r = 列[0]
        print("%-5s 文本=%r 声线=%s 时长=%.2fs 通过=%s 自检=%s"
              % (名, r.get("文本"), r.get("声线"), r.get("时长秒", -1),
                 r.get("通过"), r.get("自检")))
    print("注册表:", 器.表.表现()["当前"])
