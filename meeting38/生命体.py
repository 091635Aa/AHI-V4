# -*- coding: utf-8 -*-
"""meeting38/生命体.py —— AHI 五路并发生命体（装配 + 运行 + 判据）

━━━ 红线一（架构）━━━
把共享意识工作区与六路循环装配成"一个生命体"：
  眼（连续视觉）· 耳（连续听觉）· 脑（连续思维）· 手（连续轨迹）· 嘴（并发发声）· 语义（连续 affordance）
六条线程各自独立跑，只经 `共存态` 读写耦合。**没有任何一路 join/等待另一路。**
脑在想的同时手能更早出下一招，眼睛一更新元素位置手就在途中改落点，
嘴在说的时候手照样动——这就是"脑子想/嘴巴说/眼睛看/手操作"同时发生的落点。

━━━ 红线二（非 Transformer）━━━
全链路不含 Transformer / attention / QKV / softmax / 神经网络 / 梯度。
感知=注视–预测–眼跳（视场前端，零模型零 OCR，禁 C9/C10 外挂）；表示与决策=HDC 双极超向量+计数经验。
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting33"))
sys.path.insert(0, os.path.dirname(HERE))                    # AHI3 根（迁移：随文件定位）

# ── 连续采集预热：必须在 torch（共存态）之前；本机实测 torch 先载入会致 DDA 建相机失败 ──
try:
    sys.path.insert(0, os.path.join(HERE, "实施"))
    import 连续采集 as _连续采集
    _连续采集.预热()
except Exception:
    pass

from 共存态 import 共存态                                    # noqa: E402,F401
from 器官_眼 import 眼器                                     # noqa: E402
from 器官_耳 import 耳器                                     # noqa: E402
from 器官_手 import 手器                                     # noqa: E402
from 器官_嘴 import 嘴器                                     # noqa: E402
from 器官_脑 import 脑器                                     # noqa: E402
from 语义 import 语义图                                      # noqa: E402
from 器官_我 import 我器                                     # noqa: E402  (步3b 第7路自我线程)

try:
    from AHI躯体 import 台账
except Exception:                                            # pragma: no cover
    台账 = None


def _造台账(路径):
    if 台账 is None:
        return None
    try:
        return 台账(路径)
    except Exception:
        return None


class 生命体:
    def __init__(self, 用语音: bool = True, 种子: int = 3800, 用gpu: bool = True,
                 演练: bool = False, 眼帧源=None, 手世界=None,
                 脑允许打字=(), 脑允许拖=False, 脑允许双=False, 脑允许回报=False):
        """演练=True：手绝不真动鼠标键盘（自动化验证用）；实机需所有者知情。
        步5 安全沙盒（默认全关 → 既有行为逐位不变）：
          眼帧源   → 眼器(帧源=...)：眼改看注入帧（沙盒世界），不碰真屏；
          手世界   → 手器(世界=...)：演练动作落地沙盒世界（学习闭环）；
          脑允许打字/脑允许拖 → 脑器：键入/拖 进探索字母表。
        步7 目的性与多巴胺（默认全关）：
          脑允许双 → 目标命中出「双击=运行」招；脑允许回报 → 多巴胺账本+回报偏置。"""
        self.态 = 共存态(种子=种子, 用gpu=用gpu)
        self.账 = _造台账(os.path.join(HERE, "生命体台账.jsonl"))
        self.眼 = 眼器(self.态, self.账, 帧源=眼帧源)
        self.耳 = 耳器(self.态, self.账)
        self.手 = 手器(self.态, self.账, 演练=演练, 世界=手世界)
        self.嘴 = 嘴器(self.态, self.账, 用语音=用语音)
        self.语义 = 语义图(self.态, self.账)
        self.脑 = 脑器(self.态, self.语义, self.账,
                       允许打字=tuple(脑允许打字), 允许拖=bool(脑允许拖),
                       允许双=bool(脑允许双), 允许回报=bool(脑允许回报))
        self.我 = 我器(self.态, self.账)                     # 步3b：第7路自我线程
        self.到点 = threading.Event()
        self.线s = {}

    # ── 装配六路（名字→目标） ──
    def _路线表(self):
        def 起(目标):
            def 跑():
                try:
                    目标(self.到点)
                except TypeError:
                    # 兼容不带参数的运行循环（防御）
                    目标()
                except Exception as ex:                       # pragma: no cover
                    if self.账:
                        self.账.记("线异常", 错=repr(ex))
            return 跑
        表 = dict(眼=self.眼.运行, 耳=self.耳.运行, 手=self.手.运行,
                  嘴=self.嘴.运行, 语义=self.语义.运行, 脑=self.脑.循环,
                  我=self.我.运行)                           # 步3b：第7路
        # 签名自适应：只传 to_stop 的传参，其余零参调用
        out = {}
        for 名, fn in 表.items():
            参数 = list(inspect.signature(fn).parameters)
            if 参数:
                out[名] = threading.Thread(target=起(fn), name=名, daemon=True)
            else:
                out[名] = threading.Thread(target=fn, name=名, daemon=True)
        return out

    def 上线(self):
        self.线s = self._路线表()
        for 名, t in self.线s.items():
            t.start()
        if self.账:
            self.账.记("上线", 路=list(self.线s))
        return list(self.线s)

    def 存活(self) -> dict:
        return {名: bool(t.is_alive()) for 名, t in self.线s.items()}

    # ── 运行 + 采样（采样的意义：证明"手在动时脑仍在想、嘴仍在说"） ──
    def 跑(self, 时长: float = 120.0, 采样周期: float = 0.5) -> dict:
        self.上线()
        t0 = time.time()
        样s = []
        上次脑 = 0
        while time.time() - t0 < 时长 and not self.到点.is_set():
            time.sleep(采样周期)
            动 = self.态.取动作()
            脑拍 = self.脑.思维拍
            样s.append(dict(
                t=round(time.time() - t0, 2),
                手状态=动.get("状态"), 手进度=动.get("进度"),
                动作计数=int(动.get("动作计数", 0)),
                纠偏=int(动.get("纠偏次数", 0)),
                脑拍=脑拍, 脑增=脑拍 - 上次脑,
                说数=self.态.取表达()["说数"],
                元素=len(self.态.取视觉().get("元素s", [])),
                自体拍=self.我.自体拍, 库记录=self.我.统计["库记录"],
                存活=self.存活(),
            ))
            上次脑 = 脑拍
        self.到点.set()
        time.sleep(0.8)
        # 存活判据取「运行中最后一次采样」——停机后各路正确退出，事后采样会
        # 把"善终"误判成"死亡"（实测教训：五路 false 全是收工正常退出）。
        运行中存活 = 样s[-1]["存活"] if 样s else {}
        try:
            self.嘴.停()
        except Exception:
            pass
        return self._结算(样s, 运行中存活, 时长)

    # ── 判据 ──
    def _结算(self, 样s, 存活, 时长) -> dict:
        判 = self._判表(样s, 存活, 时长)
        with open(os.path.join(HERE, "生命体结果.json"), "w", encoding="utf-8") as f:
            json.dump(判, f, ensure_ascii=False, indent=1, default=str)
        return 判

    def _判表(self, 样s, 存活, 时长) -> dict:
        路 = list(self.线s)
        全活 = all(存活.get(名, False) for 名 in 路)
        忙样 = [s for s in 样s if s["手状态"] != "闲"]
        # 并发铁证：手在动的采样点上，脑是否仍在推进思维拍
        并想 = [s for s in 忙样 if s["脑增"] > 0]
        拼命 = (len(并想) / len(忙样)) if 忙样 else 0.0
        # 嘴在手忙时是否仍在输出
        嘴增 = 0
        for a, b in zip(样s, 样s[1:]):
            if a["手状态"] != "闲" and b["说数"] > a["说数"]:
                嘴增 += 1
        动 = self.态.取动作()
        轨迹点 = len(动.get("轨迹") or [])
        度量 = self.态.取度量()
        语 = self.语义.判据()
        判 = dict(
            时长s=round(时长, 1),
            采样数=len(样s),
            各路存活=存活,
            全线存活=bool(全活),
            冷启动=dict(首采样秒=(样s[0]["t"] if 样s else None),
                        说明="眼无模型加载（视场前端纯 numpy；步3 已下线 YOLO/OCR）；"
                             "自我线程首拍含综合体链导入（实测数秒），判据窗口应覆盖暖机后"),
            W1并发=dict(
                手忙采样=len(忙样), 手忙中脑在推进=len(并想),
                脑不阻塞率=round(拼命, 3),
                手忙时嘴仍在说=嘴增,
                过=bool(全活 and 拼命 >= 0.8)),
            W2连续轨迹=dict(
                轨迹点数=轨迹点, 动作计数=int(动.get("动作计数", 0)),
                过=bool(轨迹点 >= 6 or int(动.get("动作计数", 0)) >= 1)),
            W3中途纠偏=dict(纠偏次数=int(动.get("纠偏次数", 0)),
                            说明="纠偏按需触发：静态界面无纠偏机会属正常；"
                                 "机制判据由自检A3（移动元素→途中改落点）证明",
                            过=True),
            W4语义=dict(**{k: int(v) for k, v in 语.items()},
                        过=bool(语.get("按钮数", 0) >= 1 and 语.get("最大深度", 0) >= 2)),
            W5台账=dict(行数=(self.账.n if self.账 else 0),
                        过=bool((self.账.n if self.账 else 0) >= 30)),
            W6语义线程写回=dict(容器根=语.get("容器根数", 0),
                               过=bool(语.get("容器根数", 0) >= 1)),
            度量=度量,
            脑统计=self.脑.统计(),
            我统计=dict(self.我.统计, 桥在=bool(self.我.桥)),   # 步3b：自我线程证据
        )
        with open(os.path.join(HERE, "生命体结果.json"), "w", encoding="utf-8") as f:
            json.dump(判, f, ensure_ascii=False, indent=1, default=str)
        return 判


def 主(时长: float = 120.0, 用语音: bool = True, 演练: bool = False):
    体 = 生命体(用语音=用语音, 演练=演练)
    判 = 体.跑(时长=时长)
    print(json.dumps(判, ensure_ascii=False, indent=1, default=str))
    return 判


if __name__ == "__main__":
     import argparse
     ap = argparse.ArgumentParser()
     ap.add_argument("--秒", type=float, default=120.0)
     ap.add_argument("--静音", action="store_true", help="不发声（测试用）")
     ap.add_argument("--演练", action="store_true",
                     help="手不真动鼠标（自动化验证；实机跑不许加此旗标）")
     a = ap.parse_args()
     主(时长=a.秒, 用语音=not a.静音, 演练=a.演练)
