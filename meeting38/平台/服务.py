# -*- coding: utf-8 -*-
"""服务.py —— AHI 认知架构展示平台 · 后端（纯标准库：HTTP + SSE，零第三方依赖）。

定位（2026-10-06 与所有者对齐）：本平台是 **HDC 认知架构的展示平台**，不是一个通用
对话产品。它把 meeting38「纯文本对话态」这颗活体直接服务化，让外部**看得见**这些性质：

  · 同一生命 · 多窗口：对话流按窗口隔离，记忆（综合体 + 容量桥）全生命共享——
    A 窗口听来的八卦，B 窗口能回忆起来（跨窗口召回）。
  · 内在状态驱动：睡/醒、检索深度、接受门、回执框定 全由引擎当下遥测
    （能量 e / 新异 ñ / 消化债 / 饱足）决定，不是定时器、不是写死阈值。
  · 确定性可复现（C13）：/api/replay 用固定探针 + 同 seed 重跑，输出逐字可复现（字回归）。

只经 文本态 门面 + 共存态 槽位；**不 import 任何器官、不改引擎文件**。
铁律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。

启动： python 服务.py [端口]        （默认 8760；浏览器打开 http://127.0.0.1:8760/）
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))          # .../meeting38/平台
M38 = os.path.dirname(HERE)                                # .../meeting38
实施 = os.path.join(M38, "实施")
AHI = r"<workspace>/03_AHI\AHI2.1"
for _p in (实施, M38, AHI, os.path.join(AHI, "meeting31")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from 文本态 import 文本态                                    # noqa: E402


def _f(x, 默认=0.0):
    try:
        return round(float(x), 3)
    except Exception:
        return 默认


# ══════════════════════ 事件总线（SSE 扇出） ══════════════════════
class 事件总线:
    def __init__(self):
        self._锁 = threading.Lock()
        self._订阅者: dict = {}
        self._号 = 0

    def 订阅(self) -> "queue.Queue":
        with self._锁:
            self._号 += 1
            q: "queue.Queue" = queue.Queue(maxsize=4000)
            self._订阅者[self._号] = q
            return q

    def 退订(self, q) -> None:
        with self._锁:
            for k, v in list(self._订阅者.items()):
                if v is q:
                    self._订阅者.pop(k, None)

    def 发布(self, 事件: dict) -> None:
        with self._锁:
            qs = list(self._订阅者.values())
        for q in qs:
            try:
                q.put_nowait(事件)
            except queue.Full:
                pass


# ══════════════════════ 生命调度（把活体驱动起来 + 推流） ══════════════════════
class 生命调度:
    """驱动 文本态：有未读消息的窗口走回合；没有就让它自己作主（自主一拍）。
    每拍把「它说的话」（表达槽）与「脑内念头」按窗口推给订阅者。"""

    def __init__(self, 态: 文本态, 总线: 事件总线, 拍间隔: float = 0.2):
        """拍间隔=自主节律（秒）。0.2s ≈ 5 拍/秒：既能看它"活着"，又不至于把开放域
        学习速率拉成天文数字（实测 0.08s 时数分钟即涨上千记忆段）。"""
        self.态, self.总线, self.拍间隔 = 态, 总线, float(拍间隔)
        self._停 = threading.Event()
        self._线程 = None
        self._脑内号: dict = {}                 # 会话 → 已推条数（推增量）
        self._上次状态 = 0.0

    def 起(self) -> None:
        self._线程 = threading.Thread(target=self._跑, name="平台-生命", daemon=True)
        self._线程.start()

    def 停(self) -> None:
        self._停.set()
        if self._线程 is not None:
            self._线程.join(timeout=1.5)

    def _推说(self) -> None:
        while True:
            got = self.态.态.取下一句()
            if got is None:
                return
            t, 文, 会话 = got[0], got[1], got[2]
            self.总线.发布(dict(类型="说", 会话=会话, 文=文, t=t))

    def _推脑内(self) -> None:
        for 会话 in self.态.态.会话们():
            xs = self.态.态.取脑内(上限=512, 会话=会话)
            n = self._脑内号.get(会话, 0)
            if n > len(xs):
                n = 0
            for t, 文 in xs[n:]:
                self.总线.发布(dict(类型="脑内", 会话=会话, 文=文, t=t))
            self._脑内号[会话] = len(xs)

    def 遥测(self) -> dict:
        桥 = self.态.桥
        return dict(
            拍=int(getattr(桥, "拍", 0)),
            能量=_f(getattr(桥, "e", None)),
            新异=_f(getattr(桥, "ñ", None)),
            消化债=_f(getattr(桥, "消化债", 0.0)),
            饱足=_f(getattr(桥, "饱足", 1.0)),
            记忆段=len(getattr(getattr(桥, "codec", None), "_段", []) or []),
            账本=len(getattr(桥, "账本", []) or []),
            自言=len(getattr(getattr(桥, "自言", None), "流", []) or []),
            记忆库=(self.态.容量.库大小() if self.态.容量 else 0),
            会话们=self.态.态.会话们(),
            醒=int(self.态.态.对话态(会话="主")["醒"]),
            统计=dict(self.态.统计))

    def _推状态(self) -> None:
        now = time.time()
        if now - self._上次状态 < 0.5:
            return
        self._上次状态 = now
        self.总线.发布(dict(类型="遥测", 数据=self.遥测()))

    def _跑(self) -> None:
        while not self._停.is_set():
            try:
                待 = self.态.态.待处理会话()
                if 待:
                    for s in 待:
                        try:
                            self.态.一拍(会话=s)        # 人来了 → 回合（先想后说）
                        except Exception:
                            pass
                else:
                    try:
                        self.态.自主一拍(会话="主")      # 没人说话 → 它自己作主（想 / 睡）
                    except Exception:
                        pass
                self._推说()
                self._推脑内()
                self._推状态()
            except Exception:
                pass
            self._停.wait(self.拍间隔)


# ══════════════════════ HTTP 处理器 ══════════════════════
class 处理器(BaseHTTPRequestHandler):
    server_version = "AHI-Platform/1.0"
    protocol_version = "HTTP/1.1"        # ★ SSE 必须：默认 HTTP/1.0 会让浏览器立即 abort 流
    态: 文本态 = None
    总线: 事件总线 = None
    调度: 生命调度 = None
    根: str = ""

    def log_message(self, *a):                                  # 静音访问日志
        pass

    # ── 基础 ──
    def _json(self, obj, code: int = 200) -> None:
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def _html(self, 路径: str) -> None:
        with open(路径, "rb") as f:
            b = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _体(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:
            return {}

    def _快照(self, 会话: str) -> dict:
        态 = self.态
        return dict(类型="快照", 会话=会话, 遥测=self.调度.遥测(),
                    对话流=[[t, 谁, 文] for (t, 谁, 文)
                            in 态.态.取对话(上限=300, 会话=会话)],
                    脑内=[[t, 文] for (t, 文)
                          in 态.态.取脑内(上限=80, 会话=会话)])

    def _记忆(self, qs: dict) -> dict:
        桥 = self.态.桥
        q = (qs.get("q") or [""])[0]
        段s = getattr(桥.codec, "_段", []) or []
        if q:
            try:
                hits = 桥._检索(q, k=int((qs.get("k") or ["20"])[0]))
            except Exception:
                hits = []
            return dict(查询=q, 命中=[
                dict(段=段, 分=_f(分), 源=桥.段源.get(段, "用户"),
                     说话人=桥.段说话人.get(段)) for 段, 分 in hits])
        条 = [dict(段=段, 源=桥.段源.get(段, "用户"), 说话人=桥.段说话人.get(段))
              for 段 in 段s]
        return dict(总数=len(条), 条=条[::-1][:300])

    # ── GET ──
    def do_GET(self):
        u = urlparse(self.path)
        p, qs = u.path, parse_qs(u.query)
        if p in ("/", "/index.html"):
            return self._html(os.path.join(self.根, "控制台.html"))
        if p == "/api/state":
            return self._json(self._快照((qs.get("会话") or ["主"])[0]))
        if p == "/api/memory":
            return self._json(self._记忆(qs))
        if p == "/api/stream":
            return self._SSE()
        return self._json(dict(错="未知路径 %s" % p), 404)

    # ── POST ──
    def do_POST(self):
        p = urlparse(self.path).path
        d = self._体()
        if p == "/api/say":
            会话 = str(d.get("会话") or "主")
            谁 = str(d.get("谁") or "我")
            文 = str(d.get("文") or "").strip()
            if not 文:
                return self._json(dict(错="空消息"), 400)
            self.态.态.写用户说(文, 谁=谁, 会话=会话)
            return self._json(dict(ok=True, 会话=会话))
        if p == "/api/session":
            名 = str(d.get("名") or "").strip()
            if 名:
                self.态.态._槽(名)                              # 预建窗口（记忆仍全生命共享）
            return self._json(dict(会话们=self.态.态.会话们()))
        if p == "/api/replay":
            return self._replay()
        return self._json(dict(错="未知路径 %s" % p), 404)

    def _replay(self):
        """确定性字回归（C13）：固定探针 + 同 seed 重跑，与基线对比。"""
        try:
            import probe_字回归 as P
            记 = P.采集()
            m = P.指标(记)
            基 = None
            if os.path.exists(P.基线):
                with open(P.基线, encoding="utf-8") as f:
                    b = json.load(f)
                基 = P.指标(b["记"])
            return self._json(dict(记=记, 指标=m, 基线=基))
        except Exception as ex:
            return self._json(dict(错=repr(ex)), 500)

    # ── SSE ──
    def _写SSE(self, 事件: dict) -> None:
        self.wfile.write(("data: " + json.dumps(事件, ensure_ascii=False) + "\n\n")
                         .encode("utf-8"))
        self.wfile.flush()

    def _SSE(self):
        q = self.总线.订阅()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")     # 禁代理缓冲，逐帧下发
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self._写SSE(dict(类型="遥测", 数据=self.调度.遥测()))
            self._写SSE(self._快照("主"))
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                self._写SSE(ev)
        except Exception:
            pass
        finally:
            self.总线.退订(q)
            self.close_connection = True      # 流结束即关连接（HTTP/1.1 下无 Content-Length）


def main(端口: int = 8760, 种子: int = 310):
    print("[平台] 正在唤醒 AHI 生命（种子 %d）…" % 种子, flush=True)
    t0 = time.time()
    态 = 文本态(种子=种子).建(接开放域=True)
    总线 = 事件总线()
    调度 = 生命调度(态, 总线)
    调度.起()
    处理器.态, 处理器.总线, 处理器.调度, 处理器.根 = 态, 总线, 调度, HERE
    服 = ThreadingHTTPServer(("127.0.0.1", int(端口)), 处理器)
    print("[平台] 就绪（建模 %.1fs）： http://127.0.0.1:%d/" % (time.time() - t0, 端口),
          flush=True)
    try:
        服.serve_forever()
    except KeyboardInterrupt:
        print("\n[平台] 收到中断，停机…", flush=True)
    finally:
        调度.停()
        try:
            态.停机()
        except Exception:
            pass
        服.server_close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8760)