# -*- coding: utf-8 -*-
"""实施/test_手重做原型.py —— 视觉重启 步3 判据（V6：360° 自由控制器）。

依据：`步5_文本对话态与全屏视场_设计.md` §5（手重定位）+ §7 V6。
  · 结构：旧「过渡器」结构（_建段/_计划/点s/_采样轨迹/_fitts）必须**不存在**（L1 回归护栏）
  · V6-1 画圈：目标绕圆 → 光标描圆（半径贴合）且位移方向**覆盖全向**（360° 无死角）
  · V6-2 中途改向：飞行中反方向 → **惯性平滑转向**（首拍仍带原向动量，非瞬移反向）且最终到达
  · V6-3 原地微调：近距目标（<减速半径）仍**收敛到达**（不卡死、不过冲）
  · V6-4 无瞬移：每拍位移 ≤ Vmax·采样 + 微颤（有界步长）

全程 演练=True：控制律/状态机/轨迹全链路照常，但**绝不真动鼠标键盘**（不触真屏）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
根 = os.path.dirname(HERE)
for p in (根, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from 共存态 import 共存态                       # noqa: E402
from 器官_手 import 手器, Vmax                    # noqa: E402
from 器官_脑 import 脑器                          # noqa: E402
from 语义 import 语义图                           # noqa: E402

步上限 = Vmax * 0.016 + 3.0                        # 有界步长上限（+ 微颤容差）≈ 38px


def 合成立体(态, n=8):
    """合成一屏元素（不碰真屏），句柄/字/框/心齐备。"""
    元s = []
    for i in range(n):
        x, y, w, h = 60 + (i % 4) * 220, 80 + (i // 4) * 160, 120, 44
        元s.append(dict(句柄="E%d" % i, 框=(x, y, w, h), 心=(x + w // 2, y + h // 2),
                        字="钮%d" % i, 主色=(200, 200, 200), 形状="横条", 置信=0.9))
    态.写视觉(元s, 帧号=1, 变化率=0.0, 屏幕=(1920, 1080))
    return 元s


def _驱动(手, 目标):
    """直接给控制器一个落点并推进一拍（不经意图槽，测控制律本体）。"""
    手._目标 = np.asarray(目标, dtype=np.float64)
    手._推进()


def 判结构() -> list:
    判 = []
    旧API = ("_建段", "_计划", "_采样轨迹", "_fitts", "_首目标", "_推进一段", "_重定终点")
    残留 = [n for n in 旧API if hasattr(手器, n)]
    新API = ("_读意图", "_纠偏", "_期望速度", "_推进", "_到达")
    缺 = [n for n in 新API if not hasattr(手器, n)]
    判.append(("V6-0a 旧过渡器结构已废除", not 残留, "残留=%s" % (残留,)))
    判.append(("V6-0b 新控制器 API 就位", not 缺, "缺=%s" % (缺,)))
    with open(os.path.join(根, "器官_手.py"), encoding="utf-8") as f:
        码 = f.read()
    无点s = '["点s"]' not in 码 and "'点s'" not in 码
    判.append(("V6-0c 源码无「点s」预采样点列（键/字段级）", 无点s, "含点s键=%s" % (not 无点s)))
    return 判


def 判画圈(态) -> list:
    手 = 手器(态, 演练=True)
    心 = np.array([960.0, 540.0])
    手._移光标(tuple(心))
    R = 300.0
    迹 = [np.array(手._光标(), dtype=np.float64)]
    for i in range(720):                                   # 2 圈，每拍前进 1°
        a = 2.0 * np.pi * i / 360.0
        _驱动(手, 心 + R * np.array([np.cos(a), np.sin(a)]))
        迹.append(np.array(手._光标(), dtype=np.float64))
    迹 = np.array(迹)
    稳 = 迹[120:]                                          # 丢弃起步段
    半径 = np.hypot(稳[:, 0] - 心[0], 稳[:, 1] - 心[1])
    贴圆 = float(np.abs(半径 - R).max())
    判 = []
    判.append(("V6-1a 光标描圆（半径贴合 R=300）", 贴圆 <= 60.0,
               "半径偏差max=%.1f 半径[%.0f,%.0f]" % (贴圆, 半径.min(), 半径.max())))
    # 位移方向覆盖：8 个 45° 扇区各至少一个位移
    移 = np.diff(稳, axis=0)
    大 = 移[np.hypot(移[:, 0], 移[:, 1]) > 1.0]
    角 = (np.degrees(np.arctan2(大[:, 1], 大[:, 0])) % 360.0)
    扇 = set((角 // 45.0).astype(int).tolist())
    判.append(("V6-1b 位移方向覆盖全向（≥7/8 扇区）", len(扇) >= 7,
               "覆盖扇区=%d/8 %s" % (len(扇), sorted(扇))))
    return 判


def 判改向(态) -> list:
    手 = 手器(态, 演练=True)
    手._移光标((300, 540))
    手._目标 = np.array([1500.0, 540.0])
    for _ in range(14):
        手._推进()
    速前 = 手._速度.copy()
    p0 = np.array(手._光标(), dtype=np.float64)
    手.投喂意图(dict(目标点=(300, 100), 来源="自检"))       # 反方向新意图
    步s = []
    到 = False
    for _ in range(80):
        手._推进()
        p1 = np.array(手._光标(), dtype=np.float64)
        步s.append(p1 - p0)
        p0 = p1
        if 手._目标 is None:
            到 = True
            break
    步s = np.array(步s)
    步长 = np.hypot(步s[:, 0], 步s[:, 1]) if len(步s) else np.array([0.0])
    首拍x = float(步s[0, 0]) if len(步s) else 0.0
    判 = []
    判.append(("V6-2a 改向前高速右行（v≈Vmax）",
               float(np.hypot(*速前)) > 0.8 * Vmax,
               "改向前速率=%.0f px/s" % float(np.hypot(*速前))))
    判.append(("V6-2b 首拍仍带原向动量（惯性平滑，非瞬移反向）",
               首拍x > 2.0, "改向后首拍Δx=%.2f（>0=未瞬变）" % 首拍x))
    判.append(("V6-2c 最终到达新落点",
               到 and float(np.hypot(*(p0 - np.array([300.0, 100.0])))) <= 3.0,
               "到达=%s 终误差=%.2f" % (到, float(np.hypot(*(p0 - np.array([300.0, 100.0])))))))
    return 判


def 判微调(态) -> list:
    手 = 手器(态, 演练=True)
    手._移光标((500, 500))
    到 = False
    最大偏差 = 0.0
    终距 = 1e9
    for _ in range(120):
        距 = float(np.hypot(505.0 - 手._光标()[0], 503.0 - 手._光标()[1]))
        最大偏差 = max(最大偏差, 距)
        _驱动(手, (505.0, 503.0))
        if 手._目标 is None:
            到 = True
            终距 = float(np.hypot(505.0 - 手._光标()[0], 503.0 - 手._光标()[1]))
            break
    判 = [("V6-3a 近距目标收敛到达（不卡死）", 到, "到达=%s" % 到),
          ("V6-3b 近距不过冲（全程偏差 ≤ 初始距+2）", 最大偏差 <= 5.83 + 2.0,
           "全程最大偏差=%.2f" % 最大偏差),
          ("V6-3c 终误差 ≤ 到达门 2.0", 终距 <= 2.0 + 1e-6, "终误差=%.3f" % 终距)]
    return 判


def 判无瞬移() -> list:
    态 = 共存态(种子=3800, 用gpu=False)
    手 = 手器(态, 演练=True)
    手._移光标((100, 100))
    手._目标 = np.array([900.0, 700.0])
    迹 = [np.array(手._光标(), dtype=np.float64)]
    for _ in range(400):
        手._推进()
        迹.append(np.array(手._光标(), dtype=np.float64))
        if 手._目标 is None:
            break
    迹 = np.array(迹)
    跳 = np.hypot(*(np.diff(迹, axis=0).T))
    最大跳 = float(跳.max()) if len(跳) else 0.0
    终误 = float(np.hypot(900.0 - 迹[-1, 0], 700.0 - 迹[-1, 1]))
    return [("V6-4a 每拍位移 ≤ 有界步长（%.0fpx）" % 步上限, 最大跳 <= 步上限,
             "最大相邻跳变=%.2f" % 最大跳),
            ("V6-4b 点数 ≥ 6", len(迹) >= 6, "点数=%d" % len(迹)),
            ("V6-4c 终点误差 ≤ 3", 终误 <= 3.0, "终误差=%.2f" % 终误)]


def 判放弃快通道() -> list:
    """手「放弃」→ 脑立刻释放连招位（旧版 str(反馈列表) 恒假，潜伏 bug；步3 修复）。"""
    态 = 共存态(种子=3800, 用gpu=False)
    合成立体(态, 6)
    手 = 手器(态, 演练=True)
    手._移光标((100, 100))
    手.投喂意图(dict(句柄="不存在", 目标点=(500, 300), 来源="自检"))   # 无兜底目标消失
    手._一拍()
    动 = 态.取动作()
    弃 = int(动.get("放弃计数", 0))
    脑 = 脑器(态, 语义图(态))
    脑._待发意图 = dict(占位=True)                    # 模拟"已预置待手消费的下一招"
    脑._归因(态.取视觉(), 动)
    清 = 脑._待发意图 is None and 脑._上次放弃计数 == 弃
    return [("V6-5a 目标消失→手放弃计数+1", 弃 >= 1, "放弃计数=%d" % 弃),
            ("V6-5b 脑据放弃计数立刻释放连招位", 清,
             "待发意图已清=%s 脑计数=%d" % (脑._待发意图 is None, 脑._上次放弃计数))]


def 主():
    判 = 判结构()
    判 += 判画圈(共存态(种子=3800, 用gpu=False))       # 每个子测独立态：
    判 += 判改向(共存态(种子=3800, 用gpu=False))       # 避免上一子测残留意图污染下一子测
    判 += 判微调(共存态(种子=3800, 用gpu=False))
    判 += 判无瞬移()
    判 += 判放弃快通道()
    全过 = True
    for 名, 过, 证 in 判:
        print("[%s] %s  %s" % ("过" if 过 else "败", 名, 证))
        全过 = 全过 and 过
    print("\n==== V6 手重做（360° 自由控制器）总判：%s ====" % ("全过" if 全过 else "未过"))
    return 全过


if __name__ == "__main__":
    sys.exit(0 if 主() else 1)