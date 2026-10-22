# -*- coding: utf-8 -*-
"""meeting38/器官_耳.py —— AHI 耳朵器官（五路并发之一）。

本器官只读写共享工作区（共存态），不调用任何其他器官：它连续从系统回环（WASAPI
loopback）收音，算能量/主频、做起音检测，把结果写进听觉槽；脑各自读它要读的，
耳从不 join、从不 call 任何别的器官。

它做什么：
  · 沿 WASAPI 回环采系统正在播的声音（比麦克风干净：无环境噪声）
  · 每 ~50ms 读一小段：能量 = sqrt(mean(x^2))、主频 = rfft 取峰（滤直流）
  · 起音检测：能量越过自适应门槛、且距上次起音 >120ms → 记一次起音事件
  · 写 态.写听觉(事件=..., 主频=..., 能量=..., 起音=True/False)

门槛自适应：假警就抬、漏报就降，绝不写死绝对值。

降级路径：
  ① 回环设备：遍历 sd.query_devices() 找 max_input_channels>0 且名字含
     loopback / 立体声混音 / Stereo Mix 的设备
  ② 找不到 → sounddevice 新版 WasapiSettings(loopback=True)（默认输出回环）
  ③ 再不行 → 默认输入设备 / 首个可用输入设备
  ④ 完全没有可用输入设备 → 不崩，记一行日志后低频空转（仍活着，只是无声可写）

非 Transformer：耳朵只做能量/频域峰检测（受体级传感），不做任何 attention /
QKV / softmax / 位置编码；语义交工作区 HDC 编码器，无梯度、无反向传播。
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np


def 记(s: str):
    print("[%s][耳] %s" % (time.strftime("%H:%M:%S"), s), flush=True)


class 耳器:
    """耳朵：听系统在响什么，判起音。只写听觉槽，不调用任何其他器官。"""

    关键词 = ("loopback", "立体声混音", "stereo mix", "what u hear", "wave out")

    def __init__(self, 态, 台账=None):
        self.态 = 态
        self.台账 = 台账
        self.轮 = 0
        self.sd = None
        self.设备号 = None
        self.回环设置 = None
        self.声道 = 1
        self.采样 = 16000
        self.块长 = 1600
        self.块秒 = 1600 / 16000.0

        # ── 起音检测状态 ──
        self.静默EMA = 1e-4                            # 平静基线的能量 EMA
        self.上次能量 = 0.0
        self.起音中 = False
        self.上次起音t = -9.0
        self.倍数 = 2.5                                # 门槛 = 静默EMA * 倍数（自适应）
        self.近峰 = 0.0
        self.近峰t = time.time()
        self.事件数 = 0

        self._接入()

    # ══════════ 设备接入（回环优先，逐级降级） ══════════
    def _接入(self):
        try:
            import sounddevice as sd
        except Exception as ex:
            记("sounddevice 不可用（%r）——进入空转。" % (ex,))
            return
        self.sd = sd
        try:
            设备s = sd.query_devices()
        except Exception as ex:
            记("查询音频设备失败（%r）——进入空转。" % (ex,))
            return
        if not 设备s:
            记("系统没有任何音频设备——进入空转。")
            return

        # ① 名字含关键词的输入设备
        for i, d in enumerate(设备s):
            if d.get("max_input_channels", 0) > 0:
                nm = str(d.get("name", "")).lower()
                if any(k in nm for k in self.关键词):
                    self.设备号 = i
                    记("回环设备：#%d %s（in=%d）"
                       % (i, d.get("name", "?"), d.get("max_input_channels", 0)))
                    break

        # ② sounddevice 新版 WASAPI loopback（借默认输出设备做回环）
        if self.设备号 is None and hasattr(self.sd, "WasapiSettings"):
            try:
                self.回环设置 = self.sd.WasapiSettings(loopback=True)
                try:
                    出 = self.sd.default.device[1]
                except Exception:
                    出 = None
                if 出 is not None and 出 >= 0:
                    self.设备号 = int(出)
                    记("回环设备：WASAPI loopback 借默认输出 #%d。" % 出)
                else:
                    self.回环设置 = None
            except Exception:
                self.回环设置 = None

        # ③ 默认输入
        if self.设备号 is None:
            try:
                入 = self.sd.default.device[0]
            except Exception:
                入 = None
            if 入 is not None and 入 >= 0:
                try:
                    d = self.sd.query_devices(入)
                    if d.get("max_input_channels", 0) > 0:
                        self.设备号 = int(入)
                        记("回环未找到，用默认输入设备 #%d %s。"
                           % (入, d.get("name", "?")))
                except Exception:
                    pass

        # ④ 首个可用输入
        if self.设备号 is None:
            for i, d in enumerate(设备s):
                if d.get("max_input_channels", 0) > 0:
                    self.设备号 = i
                    记("回环未找到，用首个可用输入 #%d %s。"
                       % (i, d.get("name", "?")))
                    break

        if self.设备号 is None:
            记("没有任何可用输入设备——进入低频空转（仍活着，只是没有声音可写）。")
            return

        # 采样率/声道/块长：约 50ms 一块
        try:
            d = self.sd.query_devices(self.设备号)
            默认率 = int(d.get("default_samplerate", 16000) or 16000)
        except Exception:
            默认率 = 16000
        self.采样 = 默认率 if 默认率 > 0 else 16000
        try:
            声道数 = int(self.sd.query_devices(self.设备号)
                       .get("max_input_channels", 1) or 1)
        except Exception:
            声道数 = 1
        self.声道 = 1 if 声道数 < 1 else min(2, 声道数)
        self.块长 = max(256, int(self.采样 * 0.05))
        self.块秒 = self.块长 / float(self.采样)

    # ══════════ 一块分析 ══════════
    def _分析(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0.0, 0.0
        能量 = float(math.sqrt(float(np.mean(x * x))))
        w = x - float(x.mean())
        w = w * np.hanning(w.size)
        sp = np.abs(np.fft.rfft(w))
        fr = np.fft.rfftfreq(w.size, 1.0 / self.采样)
        掩 = fr >= 40.0                                # 滤掉直流附近
        if 掩.any():
            k = int(np.argmax(sp[掩]))
            主频 = float(fr[掩][k])
        else:
            主频 = 0.0
        return 能量, 主频

    # ══════════ 起音判定（门槛自适应） ══════════
    def _判起音(self, 能量, now):
        门槛 = max(1e-5, self.静默EMA * self.倍数)
        起音 = False
        if 能量 > 门槛 and not self.起音中 and (now - self.上次起音t) > 0.12:
            起音 = True
            self.起音中 = True
            self.事件数 += 1
            self.上次起音t = now
            if 能量 < 门槛 * 1.2:                      # 勉强越过 → 假警，抬门槛
                self.倍数 = min(8.0, self.倍数 + 0.15)
        elif self.起音中 and 能量 < 门槛 * 0.6:         # 声音落下 → 允许下次起音
            self.起音中 = False
        # 漏报：久无起音而能量曾逼近门槛 → 降门槛
        if 能量 > self.近峰:
            self.近峰 = 能量
            self.近峰t = now
        if (now - self.上次起音t) > 3.0 and (now - self.近峰t) < 0.05 \
                and self.近峰 > 门槛 * 0.75:
            self.倍数 = max(1.5, self.倍数 - 0.1)
            self.近峰t = now
        # 平静基线 EMA：仅在未起音时缓慢跟随
        if not 起音 and 能量 < 门槛:
            self.静默EMA = 0.98 * self.静默EMA + 0.02 * 能量
        self.静默EMA = max(1e-6, self.静默EMA)
        return 起音, 门槛

    # ══════════ 一块 ══════════
    def 一块(self, x):
        now = self.态.秒()
        能量, 主频 = self._分析(x)
        起音, 门槛 = self._判起音(能量, now)
        if 起音:
            self.态.写听觉(事件=dict(t=round(now, 3), 主频=round(主频, 1),
                                    能量=round(能量, 5)),
                          主频=主频, 能量=能量, 起音=True)
            if self.台账:
                self.台账.记("耳", 轮=self.轮, 起音=True,
                            主频=round(主频, 1), 能量=round(能量, 5),
                            门槛=round(门槛, 5))
        else:
            self.态.写听觉(主频=主频, 能量=能量)
        self.上次能量 = 能量
        self.轮 += 1
        if self.台账 and self.轮 % 20 == 0:
            self.台账.记("耳", 轮=self.轮, 起音=False,
                        主频=round(主频, 1), 能量=round(能量, 5),
                        门槛=round(门槛, 5))
        return 能量, 主频, 起音

    # ══════════ 连续循环 ══════════
    def _空转(self, to_stop, 说明):
        记("耳朵空转：%s" % 说明)
        while not to_stop.is_set():
            to_stop.wait(0.5)

    def 运行(self, to_stop: threading.Event):
        if self.sd is None or self.设备号 is None:
            self._空转(to_stop, "无可用输入设备。")
            return
        记("耳朵上线：回环收音，约 %.0f Hz 块。只写共享工作区。"
           % (1.0 / self.块秒))
        流 = None
        try:
            流 = self.sd.InputStream(samplerate=self.采样, channels=self.声道,
                                     dtype="float32", blocksize=self.块长,
                                     device=self.设备号,
                                     extra_settings=self.回环设置)
            流.start()
        except Exception as ex:
            if self.台账:
                self.台账.记("耳开流失败", 错=repr(ex))
            记("开流失败（%r）——降级空转。" % (ex,))
            if 流 is not None:
                try:
                    流.close()
                except Exception:
                    pass
            self._空转(to_stop, "流未开成。")
            return

        try:
            while not to_stop.is_set():
                try:
                    数据, _溢出 = 流.read(self.块长)      # sounddevice 返回 (数据, 溢出)
                    self.一块(数据)
                except Exception as ex:
                    if self.台账:
                        self.台账.记("耳异常", 错=repr(ex))
                    记("读块异常（继续）：%r" % (ex,))
                    time.sleep(0.05)
        finally:
            try:
                流.stop()
                流.close()
            except Exception:
                pass
            记("耳朵收工：起音事件 %d 次。" % self.事件数)


def 主():
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 迁移：随文件定位
    try:
        from 共存态 import 共存态
        态 = 共存态()
    except Exception as ex:
        记("无法建立工作区：%r" % (ex,))
        return
    停 = threading.Event()
    器 = 耳器(态, 台账=None)
    线 = threading.Thread(target=器.运行, args=(停,), daemon=True)
    线.start()
    time.sleep(3.0)
    停.set()
    线.join(timeout=3.0)
    a = 态.取听觉()
    记("收工：最近主频=%.1f 能量=%.5f 事件=%d"
       % (a["最近主频"], a["最近能量"], len(a["事件s"])))


if __name__ == "__main__":
    主()