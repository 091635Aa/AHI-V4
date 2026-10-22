# -*- coding: utf-8 -*-
"""meeting38/实施/连续采集.py —— 连续视觉流采集（Windows 桌面复制 DDA / dxcam）。

背景（所有者："我要连贯的，不是截图"）：眼原先每拍用 mss 截图一次（本机 30–45ms/拍，
整拍 P95 84ms 未达预登记下限 67ms）。本模块把「逐拍截图」换成**常驻连续采集**：
  · 后台线程持续取桌面复制（Desktop Duplication）流的**最新帧** → 最新帧槽
    （写者=本模块线程；读者=眼，每拍取最新帧 ≈0 成本，不再有逐拍 BitBlt 的 30–45ms）
  · 预检实测（本机 1920×1080，2026-10-05）：新帧处理 **4.7ms/帧**（mss 30–45ms）、
    静屏空转 ~50µs/次、与 mss **逐像素一致**（静屏全图平均差 0.00、corr(R,R)=1.0）
    ——同值换快 7–10×，且走 GPU 复制引擎（不占推理 GPU 算力）
  · 采样上限（默认 30fps）：只在有新帧时处理，连续动画时最高 30 帧/s（省 CPU）
  · 任何失败（dxcam 缺失/建相机失败/首帧超时/线程异常）→ 状态=不可用 →
    调用方退回 mss 单帧（优雅降级，与安全事实源同一纪律）

★ 次序依赖（本机实测，2026-10-05，见 实施/预检_连续采集*.py）：**torch 先载入时，
DDA 建相机会失败**（dxcam/bettercam 同报 DXGI_ERROR_UNSUPPORTED=-2005270524，仅有 1 个
DXGI 适配器）；**先建相机、后载 torch 则一切正常**（预热相机 torch 后 3s 仍出 32 帧，
后建相机亦可用）。故提供 `预热()`：入口进程必须在 `import torch`（即 import 共存态）
**之前**调用一次，其创建的相机被本模块持有、由 起() 接管复用。

纪律：本模块只做**采集**（受体硬件之一种，摄像头式），不做任何「看」的判断——
不是外挂模型（C10 不涉）；不写任何共享槽；不 join/不 call 任何器官。
"""
from __future__ import annotations

import threading
import time

_预热点 = None            # (相机, 首帧, t)：预热创建并持有（必须早于 torch；见文件头）


def 预热(输出: int = 0) -> bool:
    """在 torch 导入**之前**调用：创建并持有 DDA 相机（次序依赖绕行，见文件头）。
    失败（无 dxcam/建设备失败/首帧超时）→ 返回 False，采集届时走 mss 降级。"""
    global _预热点
    if _预热点 is not None:
        return True
    try:
        import dxcam
        cam = dxcam.create(output_idx=int(输出), output_color="RGB")
        if cam is None:
            return False
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.5:
            f = cam.grab()
            if f is not None:
                _预热点 = (cam, f, time.time())
                return True
            time.sleep(0.002)
        return False
    except Exception:
        return False


class 连续采集:
    """桌面复制连续采集：常驻线程 + 最新帧槽。用法：起() / 取() / 停()。"""

    def __init__(self, 输出: int = 0, 采样上限fps: float = 30.0):
        self._输出 = int(输出)
        self._上限 = max(1.0, float(采样上限fps))
        self._相机 = None
        self._线程 = None
        self._停 = threading.Event()
        self._锁 = threading.Lock()
        self._帧 = None
        self._帧t = 0.0
        self._新帧数 = 0
        self._起t = 0.0
        self._错误 = ""
        self._已停用 = False

    # ── 生命周期 ──
    def 起(self) -> bool:
        if self._线程 is not None and self._线程.is_alive():
            return True
        if self._已停用:
            return False
        global _预热点
        try:
            cam, 借预热 = None, False
            if _预热点 is not None:                          # 接管预热相机（绕开次序依赖）
                cam, 帧0, t0 = _预热点
                _预热点 = None
                借预热 = True
                with self._锁:
                    self._帧, self._帧t, self._新帧数 = 帧0, t0, 1
            else:
                import dxcam
                cam = dxcam.create(output_idx=self._输出, output_color="RGB")
            if cam is None:
                raise RuntimeError("dxcam.create 返回 None")
            if not 借预热:                                    # 新相机：等首帧（拿不到=相机无效）
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 1.5:
                    f = cam.grab()
                    if f is not None:
                        with self._锁:
                            self._帧, self._帧t, self._新帧数 = f, time.time(), 1
                        break
                    time.sleep(0.002)
                else:
                    raise RuntimeError("首帧超时（1.5s）")
            self._相机 = cam
            self._起t = time.time()
            self._停.clear()
            self._线程 = threading.Thread(target=self._环, name="连续采集", daemon=True)
            self._线程.start()
            return True
        except Exception as ex:
            self._错误 = repr(ex)
            self._已停用 = True
            return False

    def _环(self):
        """轮询取新帧 → 入槽；连续动画时按上限节流（静屏空转极廉）。"""
        cam = self._相机
        while not self._停.is_set():
            try:
                with self._锁:
                    上帧t = self._帧t
                if time.time() - 上帧t < 1.0 / self._上限:
                    self._停.wait(0.002)
                    continue
                f = cam.grab()
                if f is None:
                    self._停.wait(0.002)
                    continue
                with self._锁:
                    self._帧, self._帧t = f, time.time()
                    self._新帧数 += 1
            except Exception as ex:                            # 相机坏了 → 标死，调用方降级
                self._错误 = repr(ex)
                self._已停用 = True
                return

    def 停(self, 超时: float = 1.5):
        self._停.set()
        t = self._线程
        if t is not None and t.is_alive():
            t.join(timeout=超时)
        # 只停本模块线程并放引用：本机 comtypes 上相机 release() 有崩溃先例（预检见过 access
        # violation），且进程内采集为一次性（起→跑→停即退出），进程退出自然回收，不冒 release 风险。
        self._相机 = None

    # ── 读 ──
    def 取(self):
        """返回 (最新帧 | None, 帧时间戳, 线程是否活着)。帧只读，任何一方都不得原地改写。"""
        with self._锁:
            return self._帧, self._帧t, bool(self._线程 is not None
                                           and self._线程.is_alive())

    def 状态(self) -> dict:
        with self._锁:
            数, 帧t, 有帧 = self._新帧数, self._帧t, self._帧 is not None
        活 = bool(self._线程 is not None and self._线程.is_alive())
        运行s = (time.time() - self._起t) if self._起t else 0.0
        return dict(可用=bool(活 and 有帧), 线程活=活, 新帧数=数,
                    帧率=(round(数 / 运行s, 1) if (运行s > 0.5 and 活) else None),
                    帧年龄ms=(round((time.time() - 帧t) * 1000, 1) if 帧t else None),
                    错误=self._错误)


def 主(秒: float = 5.0):
    """自测：起连续采集 → 打印帧率/帧年龄/新帧耗时；用于接线预检留痕。"""
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    采 = 连续采集()
    ok = 采.起()
    print("连续采集 起(): %s" % ("OK" if ok else "失败 → %s" % 采.状态()["错误"]))
    if not ok:
        return 1
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 秒:
        time.sleep(0.5)
    st = 采.状态()
    采.停()
    print("状态: %s" % st)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(主())