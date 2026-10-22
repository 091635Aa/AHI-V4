# -*- coding: utf-8 -*-
"""meeting38/器官_眼.py —— AHI 眼睛器官（七路并发之一）· 步3 接线版。

本器官只读写共享工作区（共存态），不调用任何其他器官：它连续截屏、喂养视场前端、
把「误差场涌现的结构」与「注视确认的在场者」写进视觉槽；手/脑/语义各读所需。
眼从不 join、从不 call。

「看」的本体（会议_视觉 00_会议纪要 §3；步2 已验收 = 实施/视场前端.py）：
  · P1 视场码（位置绑定视网膜图，保拓扑）｜P2 误差场（预测误差）｜P3 眼跳链（主动采样）
  · 视场槽由前端单写（视场/注视/误差场/意外度/逐格意外/眼跳数/注视链）

本器官每轮（默认 15 Hz 上限；取帧=**连续视觉流最新帧**，mss 仅在降级时用）：
  ① 取帧（连续采集：桌面复制 DDA 常驻线程最新帧槽 ≈0 成本；不可用→mss 单帧）
  ② 察变（absdiff>25 占比）→ 既有字段「变化率」（保留不改语义）
  ③ 视场前端一拍 → 写视场槽（含 下一注视/意外度/逐格意外）
  ④ 元素s 涌现（**保字段**：句柄/框/心/主色/形状/置信；**字恒空**）——
     · 惊讶通道：逐格意外>在场阈值 → 4-邻域 BFS 连通区（48 格小图上做，非 cv2）
     · 确认通道：注视停留≥门 且 注视格低惊讶 → 在场者记忆（当前+近两组，object file 式）；
       每组交 **注视窗**（3×3，当前高分辨采样范围）+ **中心格**（1×1）——两者天然嵌套，
       使 语义.容器树 的「最小包含」层级不失功能（W4 深度≥2 可失败判据见 步3 接线判据）
  ⑤ 跨帧句柄匹配（贪心重叠/近心，沿用旧眼做法）→ 写视觉槽（帧号/变化率/屏幕原样）

★ 步3.5（2026-10-05，所有者："我要连贯的，不是截图"）：逐拍 mss 截图（本机 30–45ms/拍，
整拍 P95 84ms）→ **连续采集流**（桌面复制，新帧 4.7ms、与 mss 逐像素一致）；目标节拍
6Hz→15Hz（预登记工程下限）。`最近帧`/`采集后端` 为只读窗口供 实施/可视桥.py 显示实况。

铁律 C9/C10（所有者下达 2026-10-05，登记于项目记忆/06 规格/步0台账 §10.3）：
**禁 OCR、禁外挂模型当器官**。本文件不 import/引用 ultralytics(YOLO)、rapidocr(OCR)、
torch 之任何检测/识文设施——「看」不外包给模型；文字不再从像素「认」（字恒空），
「知道」走 语义.py 的功能签名（亲历交互）＋ 预测签名（步4）。

确定性：全部码/误差/眼跳由固定投影 + 绑/束 + 格序 tie 决定；同种子同帧逐位一致。
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_实施 = os.path.join(HERE, "实施")
if _实施 not in sys.path:
    sys.path.insert(0, _实施)

from 视场前端 import 视场前端, 在场阈值, 中央凹半径   # noqa: E402
from 连续采集 import 连续采集, 预热                             # noqa: E402

确认停留门 = 2        # 01 号专家 §2.2 步10：停留拍 ≥ K 且低惊讶 → 在场确认（K=2 随注视保持律）
在场者组上限 = 3      # 在场者记忆上限（object file ~4 项的同构极小版：当前＋近两组）


def 记(s: str):
    print("[%s][眼] %s" % (time.strftime("%H:%M:%S"), s), flush=True)


class 眼器:
    """眼睛：看屏、察变、注视–预测–眼跳。只写视觉槽/视场槽，不调用任何其他器官。"""

    def __init__(self, 态, 台账=None, 种子: int = 3800, 目标周期: float = 1.0 / 15.0,
                 区域=None, 格=None, 帧源=None):
        """帧源：注入式取帧（步5 安全沙盒）。为可调用对象时 `_截屏` 改取 帧源() 的
        RGB uint8 [H,W,3] 帧（沙盒世界），**不建桌面复制相机、不 mss**——演练/判据
        全程不碰真屏（确定性 C13、无头可复现）。默认 None → 既有行为**逐位不变**。"""
        self.态 = 态
        self.台账 = 台账
        self.帧源 = 帧源
        self.采集 = mss_开()                       # 降级路径（连续采集不可用时用）
        self.连续 = 连续采集()                     # 步3.5：连续视觉流（桌面复制；最新帧槽）
        self.轮 = 0
        self.最近帧 = None                         # 只读窗口：供 可视桥/演播台 当底图（不进任何槽）
        self.采集后端 = "未采样"
        self.句柄自增 = 0
        self.上帧元素s = []                        # 上一帧元素（做跨帧句柄匹配）
        self.上灰 = None                           # 上一帧灰度（做变化率）
        # 世界窗口（步3.6）：区域=(x0,y0,w,h) 时只截取该子区（元素/视场坐标均区域系）。
        # 默认 None=整屏——既有调用行为不变（生命体/可视桥/测试）。
        self.区域 = tuple(int(v) for v in 区域) if 区域 else None
        self.目标周期 = float(目标周期)             # 默认 15 Hz 上限（预登记工程下限）
        kw = {}
        if 格 is not None:
            kw = dict(格x=int(格[0]), 格y=int(格[1]))
        self.前端 = 视场前端(态, 种子=种子, **kw)    # 步3：注视–预测–眼跳前端（P1/P2/P3）
        self.确认组: deque = deque(maxlen=在场者组上限)   # 队首=最新确认 格=(gx,gy), 意外
        # 步8 中央凹流：人眼构造的"看清楚"通道——注视处全分辨 patch → 形状束（色无关结构）。
        # 描述子只依赖 编码（无训练）；失败安全（无结构→None，视场槽照常写）。
        try:
            from 形状束 import 形状描述子
            self.形状 = 形状描述子(态.编码)
        except Exception:
            self.形状 = None
        self.中央凹半径 = 48                          # 注视中心 ±48px 的 patch（96×96）

    # ══════════ 取帧（连续流优先）+ 变化率 ══════════
    def _裁(self, f):
        """世界窗口裁剪（步3.6）：区域=(x0,y0,w,h) → 只保留该子区（区域坐标系）。
        区域=None 时**原样返回**（既有调用逐位不变）。border 自动夹紧，越界不抛。"""
        if self.区域 is None:
            return f
        x0, y0, w, h = self.区域
        H, W = f.shape[:2]
        x0 = max(0, min(int(x0), max(0, W - 1)))
        y0 = max(0, min(int(y0), max(0, H - 1)))
        x1 = min(W, x0 + max(1, int(w)))
        y1 = min(H, y0 + max(1, int(h)))
        return np.ascontiguousarray(f[y0:y1, x0:x1])

    def _截屏(self):
        """取帧：优先**连续视觉流最新帧**（≈0 成本）；流不可用/线程死且帧过旧 → mss 单帧降级。
        帧源 注入（步5 沙盒）时直接取注入帧（RGB），不碰桌面复制/mss。
        取到后按 世界窗口 裁剪（区域=None 时等同整屏）。"""
        if self.帧源 is not None:
            self.采集后端 = "注入"
            img = np.ascontiguousarray(self.帧源())
            img = self._裁(img)
            self.最近帧 = img
            return img, (int(img.shape[1]), int(img.shape[0]))
        f, t, 活 = self.连续.取()
        if f is not None and (活 or (time.time() - t) < 3.0):
            self.采集后端 = "连续"
            img = self._裁(f)
            self.最近帧 = img
            return img, (int(img.shape[1]), int(img.shape[0]))
        self.采集后端 = "mss"
        g = self.采集.grab(self.采集.monitors[1])
        a = np.asarray(g)
        try:
            img = cv2.cvtColor(a, cv2.COLOR_BGRA2RGB)   # 与 numpy 翻转逐位一致（接线预检断言过），约快 3×
        except Exception:
            img = np.ascontiguousarray(a[:, :, :3][:, :, ::-1])   # BGRA→RGB（防御回退）
        img = self._裁(img)
        self.最近帧 = img
        return img, (int(img.shape[1]), int(img.shape[0]))

    def _察变(self, img) -> float:
        """与上帧灰度的 absdiff>25 占比（既有字段，保留）。缩小后比对，省 CPU。"""
        try:
            灰 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            h, w = 灰.shape[:2]
            if w > 320:
                灰 = cv2.resize(灰, (320, max(1, int(h * 320.0 / w))),
                                interpolation=cv2.INTER_AREA)
            if self.上灰 is None or self.上灰.shape != 灰.shape:
                self.上灰 = 灰
                return 0.0
            d = cv2.absdiff(灰, self.上灰)
            self.上灰 = 灰
            return round(float((d > 25).mean()), 4)
        except Exception:
            return 0.0

    # ══════════ 跨帧句柄匹配（沿用旧眼：贪心，重叠×2 − 距离/尺度） ══════════
    def _配句柄(self, 框s):
        """本帧框 ↔ 上帧元素 按重叠面积/最近心贪心匹配，返回每框对应的上帧元素。"""
        旧 = self.上帧元素s
        候选 = []
        for i, (x, y, w, h) in enumerate(框s):
            cx, cy = x + w / 2.0, y + h / 2.0
            for j, e in enumerate(旧):
                ex, ey, ew, eh = e["框"]
                ix = max(0, min(x + w, ex + ew) - max(x, ex))
                iy = max(0, min(y + h, ey + eh) - max(y, ey))
                叠比 = (ix * iy) / max(1.0, float(min(w * h, ew * eh)))
                距 = ((cx - (ex + ew / 2.0)) ** 2
                      + (cy - (ey + eh / 2.0)) ** 2) ** 0.5
                尺度 = float(max(w, h, ew, eh, 1))
                if 叠比 >= 0.1 or 距 <= 0.6 * 尺度:
                    候选.append((叠比 * 2.0 - 距 / 尺度, i, j))
        候选.sort(key=lambda z: z[0], reverse=True)
        占 = set()
        果 = [None] * len(框s)
        for _分, i, j in 候选:
            if 果[i] is not None or j in 占:
                continue
            果[i] = 旧[j]
            占.add(j)
        return 果

    # ══════════ 元素s 涌现（步3：误差场＋注视确认，不再有检测器/OCR） ══════════
    def _屏框(self, bx, by, bw, bh, 屏幕):
        """格坐标（含宽高格数）→ 区域像素框（确定性整数；网格取本眼前端实例）。"""
        W, H = 屏幕
        gx, gy = self.前端.格x, self.前端.格y
        x0 = int(bx * W / gx)
        x1 = int((bx + bw) * W / gx)
        y0 = int(by * H / gy)
        y1 = int((by + bh) * H / gy)
        return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def _记确认(self, gx, gy, 意外):
        """注视确认 → 在场者记忆刷新（同格上提队首，超出上限淘汰最旧；确定性）。"""
        项 = dict(格=(int(gx), int(gy)), 意外=float(意外))
        剩 = [x for x in self.确认组 if x["格"] != 项["格"]]
        剩.insert(0, 项)
        self.确认组 = deque(剩[:在场者组上限], maxlen=在场者组上限)

    def _候选(self, 果, 屏幕):
        """本拍候选结构：① 惊讶区（预测失败=新东西）＋② 确认组（低惊讶稳定=在场者）。"""
        逐格 = 果["逐格意外"]
        gx, gy = 果["注视"]
        停留 = int(self.前端.注视停留)
        候选s = []
        for r in self.前端.在场者(逐格, 阈值=在场阈值):          # 通道①：高惊讶连通区
            bx, by, bw, bh = r["格"]
            候选s.append(dict(框=self._屏框(bx, by, bw, bh, 屏幕),
                              意外=float(r["意外"]), 源="惊讶"))
        gi = int(gy) * self.前端.格x + int(gx)
        if 停留 >= 确认停留门 and float(逐格[gi]) < 在场阈值:     # 通道②：注视确认（低惊讶稳定）
            self._记确认(gx, gy, float(逐格[gi]))
        for 组 in self.确认组:                                   # 在场者记忆（当前＋近几组）
            bx, by = 组["格"]
            x0, x1 = max(0, bx - 中央凹半径), min(self.前端.格x, bx + 中央凹半径 + 1)
            y0, y1 = max(0, by - 中央凹半径), min(self.前端.格y, by + 中央凹半径 + 1)
            if (x1 - x0) * (y1 - y0) > 1:                        # 注视窗（3×3；边缘自动裁）
                候选s.append(dict(框=self._屏框(x0, y0, x1 - x0, y1 - y0, 屏幕),
                                  意外=组["意外"], 源="确窗"))
            候选s.append(dict(框=self._屏框(bx, by, 1, 1, 屏幕),  # 中心格（1×1）
                              意外=组["意外"], 源="确格"))
        return 候选s

    def _元素s(self, img, 候选s):
        """候选 → 元素表（保字段：句柄/框/心/主色/形状/置信；字恒空=C9）。"""
        if not 候选s:
            return []
        旧配 = self._配句柄([c["框"] for c in 候选s])
        元素s = []
        for i, c in enumerate(候选s):
            旧e = 旧配[i]
            if 旧e is not None:
                句柄 = 旧e["句柄"]
            else:
                句柄 = "E%d" % self.句柄自增
                self.句柄自增 += 1
            x, y, w, h = c["框"]
            e = dict(句柄=句柄, 框=(int(x), int(y), int(w), int(h)),
                     心=(int(x + w // 2), int(y + h // 2)), 字="",
                     主色=(0, 0, 0), 形状="方块",
                     置信=(round(c["意外"], 2) if c["源"] == "惊讶"
                           else round(1.0 - c["意外"], 2)))
            crop = img[y:y + h, x:x + w]
            if crop.size:
                平 = crop.reshape(-1, 3)
                平 = 平[:min(4000, 平.shape[0])]
                e["主色"] = tuple(int(v) for v in np.median(平, axis=0))
            比 = w / max(h, 1)
            e["形状"] = "横条" if 比 > 2.2 else ("方块" if 比 > 0.55 else "竖条")
            元素s.append(e)
        return 元素s

    # ══════════ 一轮 ══════════
    def 一轮(self):
        t0 = time.perf_counter()
        img, 屏幕 = self._截屏()
        变化率 = self._察变(img)
        果 = self.前端.一拍(img, 屏幕)                  # P1/P2/P3；前端写视场槽（单写者=眼）
        候选s = self._候选(果, 屏幕)
        元素s = self._元素s(img, 候选s)
        # 步8 中央凹流：看清"最近的那件东西"——按**元素框对齐**裁 patch → 形状束
        # （色无关结构统计；框对齐=位置/尺度归一）。匹配时框拟合有 ±数px 抖动 →
        # 输出 **5 移位束组**（基线±4px），脑侧匹配取最大相似（识别门按组最大值）。
        # 无元素/无结构 → None。失败安全不波及主链路。
        try:
            if self.形状 is not None and 元素s:
                gx, gy = self.前端.注视[0], self.前端.注视[1]
                cw, ch = 屏幕[0] / self.前端.格x, 屏幕[1] / self.前端.格y
                注心 = ((gx + 0.5) * cw, (gy + 0.5) * ch)

                def _凹选(e):
                    心 = e.get("心") or (1e9, 1e9)
                    框 = e.get("框") or (0, 0, 0, 0)
                    return (float(np.hypot(心[0] - 注心[0], 心[1] - 注心[1])),
                            框[2] * 框[3])               # 距离近者优；同心取小面积（看东西不看容器）
                近 = min(元素s, key=_凹选)
                框 = 近.get("框")
                if 框 and 框[2] >= 16 and 框[3] >= 16:
                    束s, H, W = [], img.shape[0], img.shape[1]
                    for dx, dy in ((0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)):
                        m = 6
                        x0 = max(0, int(框[0]) - m + dx)
                        y0 = max(0, int(框[1]) - m + dy)
                        x1 = min(W, int(框[0] + 框[2]) + m + dx)
                        y1 = min(H, int(框[1] + 框[3]) + m + dy)
                        束s.append(self.形状.形状束(img[y0:y1, x0:x1]))
                    心 = (int(框[0] + 框[2] / 2), int(框[1] + 框[3] / 2))
                    self.态.写视场(中央凹束=束s[0], 中央凹束s=束s, 中央凹心=心,
                                   中央凹句柄=近.get("句柄"))
        except Exception:
            pass

        self.轮 += 1
        self.态.写视觉(元素s, 帧号=self.轮, 变化率=变化率, 屏幕=屏幕)
        self.上帧元素s = 元素s
        if self.台账:
            self.台账.记("眼", 轮=self.轮, 元素=len(元素s), 变化率=变化率,
                         意外度=round(float(果["意外度"]), 4),
                         采集=self.采集后端,
                         耗ms=round((time.perf_counter() - t0) * 1000, 1))
        return len(元素s)

    # ══════════ 连续循环 ══════════
    def 运行(self, to_stop: threading.Event):
        好 = False if self.帧源 is not None else self.连续.起()
        if self.帧源 is not None:
            记("眼睛上线：注视–预测–眼跳（视场前端 P1/P2/P3）；零模型零 OCR，只写共享工作区。")
            记("采集=注入帧源（沙盒世界，不碰真屏；目标节拍 %.1f Hz）" % (1.0 / max(self.目标周期, 1e-6)))
        else:
            记("眼睛上线：注视–预测–眼跳（视场前端 P1/P2/P3）；零模型零 OCR，只写共享工作区。")
            记("采集=%s（目标节拍 %.1f Hz）"
               % ("连续视觉流（桌面复制，最新帧槽）" if 好
                  else "连续采集不可用（%s）→ mss 单帧降级" % self.连续.状态().get("错误", "?"),
                  1.0 / max(self.目标周期, 1e-6)))
        try:
            while not to_stop.is_set():
                t0 = time.perf_counter()
                try:
                    self.一轮()
                except Exception as ex:
                    if self.台账:
                        self.台账.记("眼异常", 错=repr(ex))
                    记("一轮异常（继续）：%r" % (ex,))
                    time.sleep(0.2)
                余 = self.目标周期 - (time.perf_counter() - t0)
                if 余 > 0:
                    to_stop.wait(余)
        finally:
            self.连续.停()


def mss_开():
    import mss
    return mss.mss()


def 主():
    态 = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        预热()                                       # 必须在 共存态(torch) 之前（次序依赖，见 连续采集.py）
        from 共存态 import 共存态
        态 = 共存态()
    except Exception as ex:
        记("无法建立工作区：%r" % (ex,))
        return
    停 = threading.Event()
    器 = 眼器(态, 台账=None)
    线 = threading.Thread(target=器.运行, args=(停,), daemon=True)
    线.start()
    time.sleep(5.0)
    停.set()
    线.join(timeout=3.0)
    v = 态.取视觉()
    f = 态.取视场()
    记("收工：帧号=%d 元素=%d 变化率=%s" % (v["帧号"], len(v["元素s"]), v["变化率"]))
    记("视场：注视=%s 意外度=%.4f 眼跳数=%d 注视链=%d"
       % (f["注视"], f["意外度"], f["眼跳数"], len(f["注视链"])))
    记("采集后端=%s 连续状态=%s" % (器.采集后端, 器.连续.状态()))


if __name__ == "__main__":
    主()