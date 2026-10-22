# -*- coding: utf-8 -*-
"""预检_连续采集2.py —— torch 已导入前提下：枚举适配器 + 逐 device_idx 试建 + bettercam 对照。

用法：python 预检_连续采集2.py
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch  # 复现失败条件：torch 先导
print("torch 已导入", torch.__version__)

import dxcam

try:
    print("---- dxcam.output_info() ----")
    print(dxcam.output_info())
except Exception as e:
    print("output_info 失败:", repr(e)[:130])

for 设备 in (None, 0, 1, 2):
    for 输出 in (None, 0):
        try:
            c = dxcam.create(device_idx=设备, output_idx=输出, output_color="RGB")
            f = c.grab()
            print("dxcam device_idx=%s output_idx=%s -> OK 帧=%s"
                  % (设备, 输出, None if f is None else f.shape))
            try:
                c.release()
            except Exception:
                pass
        except Exception as e:
            print("dxcam device_idx=%s output_idx=%s -> FAIL %s"
                  % (设备, 输出, repr(e)[:90]))

try:
    import bettercam
    print("---- bettercam 在 ----")
    for 设备 in (None, 0, 1):
        try:
            b = bettercam.create(device_idx=设备, output_idx=0, output_color="RGB")
            f = b.grab()
            print("bettercam device_idx=%s -> %s"
                  % (设备, "OK 帧=%s" % (None if f is None else f.shape)))
        except Exception as e:
            print("bettercam device_idx=%s -> FAIL %s" % (设备, repr(e)[:90]))
except ImportError:
    print("bettercam 未安装")