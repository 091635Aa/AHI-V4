# -*- coding: utf-8 -*-
"""meeting38/器官_嘴.py —— AHI 第五路器官：嘴（并发发声通道）

本器官只读表达槽发声，不生成语言、不调用其他器官、不阻塞他路。

━━━ 职责边界 ━━━
嘴是"发声的嘴"，不是语言模型：它每拍从共享意识工作区的**表达槽**取走待说的句子，
把它变成声音（或文字）送出去。句子内容由脑写入表达槽，嘴不生成任何语言。

━━━ 并发纪律 ━━━
  · 运行() 是连续的 ~10Hz 轮询循环，与眼/耳/脑/手各自独立线程并行。
  · 说() 只把文本塞进内部队列即返回；真正的合成/播放由一个**发声子线程**顺序完成。
    因此 SAPI 的同步阻塞 Speak 不会卡住 运行() 循环，也不会阻塞任何其他器官。
  · 任何异常都被吞掉并继续循环，绝不让嘴的故障中断生命体。

━━━ 非 Transformer 红线 ━━━
本文件不含任何 attention / QKV / softmax / 自注意力 / Transformer 层，
不引入任何神经网络 TTS（VITS/Tacotron/Transformer-TTS 等）。
"发声"仅用操作系统自带语音合成（Windows SAPI，经 win32com.client 的 SAPI.SpVoice）
或 pyttsx3；两者都不可用时降级为纯文本打印/写台账，降级**不抛异常**。

━━━ 台账 ━━━
若传入台账，每说一句调用 台账.记("嘴", 文=...)（台账接口 .记(类:str, **rec)）。
另有可选的 态.写动作(反馈=...) 回流，供手/眼感知"嘴说了什么"。
"""
from __future__ import annotations

import queue
import threading
import time


class 嘴器:
    """嘴器官：只读表达槽发声，不生成语言、不调用其他器官、不阻塞他路。"""

    def __init__(self, 态, 台账=None, 用语音: bool = True):
        """尝试初始化 SAPI / pyttsx3 语音后端；失败则降级为"只打印/写台账"，绝不抛异常。

        态    : 共存态 实例（只读其表达槽；可选写其动作槽回流）
        台账  : 可选，需提供 .记(类:str, **rec)
        用语音: False 时直接走纯文本降级路径（冒烟测试用，免得真出声）
        """
        self.态 = 态
        self.台账 = 台账
        self.用语音 = bool(用语音)

        self._q: "queue.Queue" = queue.Queue(maxsize=128)
        self._停 = threading.Event()
        self._子 = None
        self._sapi = None
        self._pyttsx3 = None
        self._后端 = "无"

        # 探测可用语音后端（失败静默降级）
        if self.用语音:
            try:
                self._后端 = self._探测后端()
            except Exception:
                self._后端 = "无"

        # 常驻发声子线程：说() 只入队，播放全在此线程，与运行() 解耦
        try:
            self._子 = threading.Thread(target=self._发声循环, name="嘴-发声", daemon=True)
            self._子.start()
        except Exception:
            self._子 = None

    # ══════════════════════ 语音后端 ══════════════════════
    def _探测后端(self) -> str:
        """探测可用的系统语音后端名，不抛异常。"""
        try:
            import win32com.client
            v = win32com.client.Dispatch("SAPI.SpVoice")
            del v
            return "sapi"
        except Exception:
            pass
        try:
            import pyttsx3
            e = pyttsx3.init()
            del e
            return "pyttsx3"
        except Exception:
            pass
        return "无"

    def _建后端(self) -> bool:
        """在发声子线程内建立本线程归属的语音对象（COM 线程亲和）。失败返回 False。"""
        if not self.用语音 or self._后端 == "无":
            return False
        if self._后端 == "sapi":
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass
            try:
                import win32com.client
                self._sapi = win32com.client.Dispatch("SAPI.SpVoice")
                return True
            except Exception:
                self._sapi = None
                self._后端 = "无"
        if self._后端 == "pyttsx3":
            try:
                import pyttsx3
                self._pyttsx3 = pyttsx3.init()
                return True
            except Exception:
                self._pyttsx3 = None
                self._后端 = "无"
        return False

    def _收后端(self):
        """发声线程退出前释放后端资源，不抛异常。"""
        try:
            if self._pyttsx3 is not None:
                self._pyttsx3.stop()
        except Exception:
            pass
        self._pyttsx3 = None
        self._sapi = None
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass

    # ══════════════════════ 发声子线程 ══════════════════════
    def _发声循环(self):
        """顺序消费队列里的文本并播放。全程 try/except，绝不因单句失败而退出。"""
        self._建后端()
        try:
            while not self._停.is_set():
                try:
                    文本 = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if 文本 is None:                     # 停机哨兵
                    break
                self._发声(str(文本))
        except Exception:
            # 子线程异常不得扩散（daemon 线程静默收尾）
            pass
        finally:
            self._收后端()

    def _发声(self, 文本: str):
        """真正把一句文本送出去：SAPI 同步阻塞 Speak（在本子线程内，故不阻塞他路）。"""
        说出 = False
        try:
            if self._sapi is not None:
                self._sapi.Speak(文本)            # 同步阻塞：放在子线程里才安全
                说出 = True
            elif self._pyttsx3 is not None:
                self._pyttsx3.say(文本)
                self._pyttsx3.runAndWait()
                说出 = True
        except Exception:
            说出 = False                          # 语音失败即降级，不抛
        try:
            if not 说出:
                print(f"[嘴] {文本}")
        except Exception:
            pass
        self._回流(文本)

    def _回流(self, 文本: str):
        """可选：把"嘴说了什么"写回动作槽，供手/眼感知。失败静默。"""
        try:
            self.态.写动作(反馈={"器官": "嘴", "文": str(文本), "t": self.态.秒()})
        except Exception:
            pass

    # ══════════════════════ 对外接口 ══════════════════════
    def 说(self, 文本: str):
        """把一句话送出去：只入队（非阻塞），由发声子线程顺序播放。"""
        try:
            self._q.put_nowait(str(文本))
        except queue.Full:
            pass
        except Exception:
            pass

    def 运行(self, to_stop: threading.Event):
        """连续循环（约 10Hz）：取表达槽待说句 → 说 → 记台账；无则短睡。

        仅在 to_stop 置位（或本器官 停/卒 被调用）时退出；全程异常不中断。
        """
        while not to_stop.is_set() and not self._停.is_set():
            try:
                got = self.态.取下一句()
                if got is None:
                    time.sleep(0.1)               # ~10Hz
                    continue
                _t, 文本 = got[0], got[1]         # 队列元素=(t,文,会话)；本器官不关心窗口
                self.说(文本)
                self._记台账(文本)
            except Exception:
                try:
                    time.sleep(0.1)
                except Exception:
                    pass

    def _记台账(self, 文本: str):
        if self.台账 is None:
            return
        try:
            self.台账.记("嘴", 文=str(文本))
        except Exception:
            pass

    # ══════════════════════ 停机钩子 ══════════════════════
    def 停(self):
        """置停机标志并投哨兵，唤醒发声子线程退出（不 join）。"""
        self._停.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass

    def 卒(self):
        """停机并等待发声子线程收尾（带超时，不抛异常）。"""
        self.停()
        try:
            if self._子 is not None and self._子.is_alive():
                self._子.join(timeout=3.0)
        except Exception:
            pass
