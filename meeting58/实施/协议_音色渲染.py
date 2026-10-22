# -*- coding: utf-8 -*-
"""meeting58/实施/协议_音色渲染.py —— 声层实现：原生声带 + 音色参数真接线（纯函数式渲染）。

为什么需要它（会议实测根因）：
  · 既有 `说_音色.py` 的 8 参数里，紧喉度/粗糙度**没有出口**，幅度被峰值归一化吃掉，
    共振峰**没有入口**（E2/E4/红队三次独立实测）⇒ 4 档档案实际只差音高。
  · 状态从未进入声带：D核=64，声带要 D=1024；旧路径用 rng(55) 随机向量代替（E3/红队）。
本件用**子类覆盖**（不改 `声带.py`，C14）补上：
  1) 共振峰缩放 f1/f2/f3（三轴独立）；2) 谱倾斜（一阶预加重/去加重）；
  3) 紧喉度 → 声门波形整形（tanh 驱动）；4) 粗糙度 → 33Hz 抖动（AM）；
  5) 气声度 → 高通噪声混入（按包络门控）；6) 增益/音量倍 → 后置（不再被归一化吃掉）；
  7) D核→D声 确定性投影 → 分位读数 → 微韵律；自我标量 → 声区小偏置。
对外口径：**参数化模仿，不是声克隆**；不得宣称是角色本人。
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting32"))

import 声带 as 声带模块  # noqa: E402  只读复用（C14）
from 接口契约_角色声线 import (  # noqa: E402
    声线档案, 状态帧, 契约错误, 契约版本, 默认种子, E1006, E1008,
)
from 工具_确定性 import 确定性向量, 确定性投影, 指纹  # noqa: E402

FS = int(声带模块.FS)
产物 = os.path.join(HERE, "产物", "发声")
os.makedirs(产物, exist_ok=True)

# 情绪偏移（沿用 meeting55/说_音色.py 的已裁决数值；只作韵律倍率）
情绪偏移 = {
    "高兴": dict(基频倍=1.15, 语速倍=1.18, 幅度倍=1.10, 气声加=-0.05, 紧喉加=0.10),
    "低落": dict(基频倍=0.88, 语速倍=0.82, 幅度倍=0.85, 气声加=0.15, 紧喉加=-0.10),
    "急促": dict(基频倍=1.10, 语速倍=1.35, 幅度倍=1.05, 气声加=-0.02, 紧喉加=0.12),
    "平静": dict(基频倍=1.00, 语速倍=1.00, 幅度倍=1.00, 气声加=0.00, 紧喉加=0.00),
}


def 夹(v, lo, hi):
    return float(min(max(float(v), lo), hi))


def 声线律(声线: 声线档案, 情绪: str = "平静", 语速倍: float = 1.0) -> dict:
    m = 情绪偏移.get(情绪, 情绪偏移["平静"])
    return dict(
        基频=夹(声线.基频 * m["基频倍"], 70.0, 400.0),
        语速=夹(声线.语速 * m["语速倍"] * 语速倍, 1.5, 7.0),
        幅度=夹(声线.幅度 * m["幅度倍"], 0.1, 1.0),
        气声度=夹(声线.气声度 + m["气声加"], 0.0, 0.8),
        紧喉度=夹(声线.紧喉度 + m["紧喉加"], 0.0, 1.0),
        粗糙度=夹(声线.粗糙度, 0.0, 1.0),
        送气量=夹(声线.送气量, 0.0, 1.0),
        声区=夹(声线.声区, 0.0, 1.0),
        _ñ=0.5,
    )


class 声带音色(声带模块.声带器官):
    """兼容壳：DSP 已下沉到 声带.py 原生音色接口（合成波形_快）。
    本类只保留构造参数与 韵律 便捷返回；不再覆盖 音节帧/合成波形（避免双重处理）。"""

    def __init__(self, 声线: 声线档案, 律: dict, 音量倍: float = 1.0,
                 种子: int = 默认种子, 腔调=None, 目录: str | None = None):
        super().__init__(seed=int(种子), D=1024, 目录=目录 or 产物,
                         台账路径=os.path.join(产物, "渲染台账.jsonl"))
        self.声线 = 声线
        self._律 = dict(律)
        self.音量倍 = float(音量倍)
        self.腔调 = dict(腔调 or {})

    def 韵律(self, 遥测: dict) -> dict:  # noqa: N802
        return dict(self._律)


class 声带渲染器:
    """声层唯一实现：`渲染(文本, 声线, 情绪, 状态)` 纯函数式（无模块级可变态）。"""

    协议版本 = 契约版本

    def __init__(self, 目录: str | None = None):
        self.目录 = 目录 or 产物
        os.makedirs(self.目录, exist_ok=True)
        self.台账 = dict(渲染次数=0, 总耗时毫秒=0.0)

    def 预热(self, 声线: 声线档案) -> None:
        声线.校验()

    def 渲染(self, 文本: str, *, 声线: 声线档案, 情绪: str = "平静",
             状态: 状态帧 | None = None, 语速倍: float = 1.0,
             音量倍: float = 1.0, 种子: int = 默认种子,
             腔调: tuple = (), 写wav: bool = True) -> dict:
        声线.校验()
        文本 = str(文本)
        律 = 声线律(声线, 情绪, 语速倍)
        if 状态 is not None:
            if int(状态.D核) <= 0:
                raise 契约错误(E1006, "状态帧 D核 非法")
            向量 = 确定性投影(状态.数组("o用"), 1024, 键="o用|%s" % 状态.指纹)
            内心 = 确定性投影(状态.数组("内心"), 1024, 键="内心|%s" % 状态.指纹)
            向量 = 向量 + 0.5 * 内心
            # 自我标量 → 声区小偏置（有界，不主导音色）
            律["声区"] = 夹(律["声区"] + 0.06 * (float(状态.自我标量) - 0.5), 0.0, 1.0)
            状态指纹 = 状态.指纹
        else:
            向量 = 确定性向量("文本|%s|%s|%d" % (文本, 声线.指纹(), int(种子)), 1024)
            状态指纹 = ""
        # meeting58：音色档案直接进 声带.py 原生接口（紧喉/粗糙/幅度/共振峰/带宽/谱倾斜）
        音色 = {k: float(律[k]) for k in 声带模块.参数表}
        音色.update(共振峰缩放=tuple(float(x) for x in 声线.共振峰缩放),
                    带宽缩放=1.0, 谱倾斜=float(声线.谱倾斜),
                    送气增益=0.0, 增益=float(声线.增益) * float(音量倍))
        降 = float(dict(腔调 or {}).get("降调半音", 0.0) or 0.0)
        if abs(降) > 1e-9:
            音色["降调半音"] = 降
        律["_音色"] = 音色
        t0 = time.perf_counter()
        器 = 声带音色(声线, 律, 音量倍=音量倍, 种子=种子, 目录=self.目录, 腔调=腔调)
        qs = 器.分位读数(向量)
        帧 = 器.音节帧(文本, 律, qs)
        波形, 元 = 器.合成波形_快(帧, 音色)
        耗时 = time.perf_counter() - t0
        时长 = 波形.size / FS
        # 自检（与既有 声带.发声 同口径）
        参数s = {k: 律[k] for k in 声带模块.参数表}
        合法, 违例 = 器._参数合法(参数s)
        往返 = 器._数轴往返(参数s)
        无nan = bool(np.all(np.isfinite(波形)))
        声源样 = np.repeat(np.asarray(帧["声源"]), 声带模块.每帧样本)[:波形.size]
        跳变 = 声带模块.数跳变(波形, 声源样, 元.get("f0样"))
        实时率 = 耗时 / max(时长, 1e-9)
        wav名 = ""
        if 写wav:
            wav名 = "渲染_%s_%s.wav" % (声线.指纹(), 指纹(文本, 状态指纹, 种子, 长=8))
            器._写wav(os.path.join(self.目录, wav名), 波形)
        self.台账["渲染次数"] += 1
        self.台账["总耗时毫秒"] += 耗时 * 1000.0
        return dict(
            文本=文本, 采样率=FS, 样本数=int(波形.size), 时长秒=round(float(时长), 4),
            波形=波形, wav=wav名, 声线=声线.名, 声线指纹=声线.指纹(),
            状态指纹=状态指纹, 种子=int(种子), 参数={k: round(float(v), 4) for k, v in 参数s.items()},
            律={k: round(float(v), 4) for k, v in 律.items() if isinstance(v, (int, float))},
            自检=dict(参数合法=bool(合法), 违例=违例, 数轴往返=float(往返),
                      无NaN=无nan, 跳变=int(跳变),
                      峰值=round(float(np.max(np.abs(波形))), 4)),
            实时率=round(float(实时率), 4),
            通过=bool(合法 and 往返 == 1.0 and 无nan and 跳变 == 0 and 实时率 <= 0.5),
            降级=False, 错="",
            溯源=dict(来源=声线.来源, 备注=声线.备注, 契约版本=契约版本,
                      口径="参数化模仿，不是声克隆", 状态接入口="确定性投影+qs微韵律+声区偏置"),
        )


_默认渲染器 = None


def 取渲染器() -> 声带渲染器:
    global _默认渲染器
    if _默认渲染器 is None:
        _默认渲染器 = 声带渲染器()
    return _默认渲染器


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import 声线库
    r = 取渲染器()
    for 名 in ("洛天依", "初音未来", "MOSS", "EmoCompanion"):
        a = 声线库.取(名)
        out = r.渲染("你好，我是%s。今天想听我说话吗？" % 名, 声线=a,
                     情绪="平静", 种子=300)
        print("%-5s %-5s 时长=%.2fs 实时率=%.3f 自检=%s 峰值=%.3f"
              % (名, out["wav"], out["时长秒"], out["实时率"], out["通过"],
                 out["自检"]["峰值"]))
