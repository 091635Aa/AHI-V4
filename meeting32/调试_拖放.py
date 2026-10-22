# -*- coding: utf-8 -*-
"""meeting32/调试_拖放.py —— 定位 tkinter 为何不响应 pyautogui 拖放。"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import pyautogui
import numpy as np
import cv2

pyautogui.FAILSAFE = False
子 = subprocess.Popen([sys.executable, os.path.join(HERE, "画布窗5.py")],
                     creationflags=subprocess.CREATE_NEW_CONSOLE)
time.sleep(3)


def 找色(低, 高):
    img = np.asarray(pyautogui.screenshot().crop((460, 100, 1460, 800)),
                     dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, 低, 高)
    M = cv2.moments(mask)
    if M["m00"] < 3000:
        return None
    return (int(M["m10"] / M["m00"]) + 460, int(M["m01"] / M["m00"]) + 100)


红 = 找色((170, 120, 150), (180, 255, 255))
print("拖前红块:", 红, flush=True)
pyautogui.moveTo(红[0], 红[1])
time.sleep(0.3)
# 方案A：pyautogui.dragTo
pyautogui.dragTo(1250, 350, duration=1.0, button="left")
time.sleep(1.0)
红2 = 找色((170, 120, 150), (180, 255, 255))
print("dragTo 后红块:", 红2, flush=True)
if 红2 is None or abs(红2[0] - 1250) > 160:
    # 方案B：显式 press/move/release 分步
    print("dragTo 未触发，改分步 press/moveUp/release…", flush=True)
    红3 = 找色((170, 120, 150), (180, 255, 255)) or 红
    pyautogui.moveTo(红3[0], 红3[1])
    time.sleep(0.3)
    pyautogui.mouseDown()
    for i in range(1, 11):
        pyautogui.moveTo(红3[0] + (1250 - 红3[0]) * i / 10,
                         红3[1] + (350 - 红3[1]) * i / 10, duration=0.1)
    pyautogui.mouseUp()
    time.sleep(1.0)
    红4 = 找色((170, 120, 150), (180, 255, 255))
    print("分步后红块:", 红4, flush=True)
子.terminate()
