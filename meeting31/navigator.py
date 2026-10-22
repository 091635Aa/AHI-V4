# -*- coding: utf-8 -*-
"""meeting31/navigator.py —— #246 开放域导航器：多根文件系统的授权边界+导航状态机原语。

设计分工（思考先于看）：
- 本类只持授权边界（允许前缀∧屏蔽前缀）与导航状态（目录栈/待访条目/文件行游标），
  提供**边界安全的原语**：列目录/进入/打开/下一行；
- 看、进、读、换、联想的全部**决策**在 synthesis.综合体的 agency+通道值+自言层做——
  决策必须先落自言（计划/疑问），再调本类原语执行（自言留痕=审计依据）；
- 构造保证只读：全类零 write/execute/delete 调用；屏蔽前缀硬过滤（大小写不敏感）。
"""
from __future__ import annotations

import os
import threading

文本扩展 = (".py", ".md", ".txt", ".json", ".csv", ".log", ".ini", ".cfg",
           ".yaml", ".yml", ".xml", ".html", ".htm", ".js", ".ts", ".css",
           ".sql", ".bat", ".ps1", ".sh")
单文件字节限 = 2 * 1024 * 1024            # >2MB 跳过（二进制/大文件防护）
IO限时 = 1.5                              # 秒——单次列目录/读文件上限（#247：任何
                                          # 单拍 IO 必须远小于看门狗 H_死亡=5s）


def 限时调用(函数, 参数, 超时: float = IO限时):
    """有界 IO：线程+join(超时)。返回 (状态, 值)——状态∈完成/超时/异常。
    超时的线程为 daemon 遗留（IO 阻塞不占 GIL，主循环零影响）。"""
    结果: dict = {}

    def _do():
        try:
            结果["r"] = ("完成", 函数(参数))
        except Exception as e:                # noqa: BLE001 —— 有界 IO：异常即证据
            结果["r"] = ("异常", "%s:%s" % (type(e).__name__, str(e)[:40]))

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(超时)
    return 结果.get("r", ("超时", None))


class 导航器:
    def __init__(self, 根s: list, 屏蔽s: list = (), 深度限: int = 12,
                 目录项限: int = 200, 文件行限: int = 200):
        self.根s = []
        for r in 根s:
            a = os.path.abspath(r)
            if a.endswith(":"):                 # abspath('<path>\\')→rstrip 后只剩盘符会
                a += "\\"                       # 变成盘相对路径（E:SteamLibrary 实证）
            self.根s.append(a)
        self.屏蔽 = [os.path.abspath(p).rstrip("\\").lower() for p in 屏蔽s]
        self.深度限, self.目录项限, self.文件行限 = 深度限, 目录项限, 文件行限
        # —— 导航状态 ——
        self.栈: list = []                 # [(目录, 待访[(名,路径,是否目录)])]
        self.根游标 = 0                    # 未启动根的下一个
        self.当前文件: str = None
        self.行s: list = []
        self.行位置 = 0
        self.文件细读 = 0                  # 当前文件细读行数（读完复盘用）
        self.文件略过 = 0                  # 当前文件略过行数
        # —— 统计（审计原料） ——
        self.访问: dict = {}               # 目录 → 次数
        self.已读文件 = 0
        self.细读行 = 0
        self.略过行 = 0
        self.屏蔽命中 = 0
        self.异常 = []                     # (路径, 异常摘要)
        self.读迹: list = []               # [(行, 文件路径)]——账本⊆原文抽检用

    # ---------- 边界判定 ----------
    def 允许(self, 路径: str) -> bool:
        p = os.path.abspath(路径).rstrip("\\").lower()
        if any(p == b or p.startswith(b + "\\") for b in self.屏蔽):
            return False
        return any(p == r.lower() or p.startswith(r.lower() + "\\")
                   for r in self.根s)

    def _深度(self, 目录: str) -> int:
        for r in self.根s:
            if 目录.lower().startswith(r.lower()):
                return os.path.relpath(目录, r).count(os.sep) + 1
        return 99

    # ---------- 原语（边界内 + IO 有界） ----------
    def 列目录(self, 目录: str):
        """有界列目录。返回 (条目, 屏蔽数) 或 (状态, None)——状态∈超时/异常:msg。"""
        状态, 值 = 限时调用(self._列目录本体, 目录)
        if 状态 == "完成":
            return 值
        self.异常.append((目录, 状态))
        return (状态, None)

    def _列目录本体(self, 目录: str):
        # ★GIL 纪律（#247）：线程内 Python 级工作量必须有界——只取前 目录项限 项。
        # OS 级阻塞（首条 dirent 到达前）不占 GIL，主循环不受影响；纯 Python 循环
        # 若无界（百万条目 list 化）会饿死主线程 → 心跳停滞 → 看门狗急停（14:52 实证）。
        条目, 屏蔽数 = [], 0
        it = os.scandir(目录)
        try:
            for e in it:
                if len(条目) >= self.目录项限:
                    break
                try:
                    全路径 = e.path
                    if not self.允许(全路径):
                        屏蔽数 += 1
                        self.屏蔽命中 += 1
                        continue
                    if e.is_dir(follow_symlinks=False):
                        if self._深度(全路径) <= self.深度限:
                            条目.append((e.name, 全路径, True))
                    elif e.is_file(follow_symlinks=False):
                        if e.name.lower().endswith(文本扩展):
                            条目.append((e.name, 全路径, False))
                except OSError as e2:
                    self.异常.append((getattr(e, "path", "?"), type(e2).__name__))
        finally:
            it.close()
        目录s = [x for x in 条目 if x[2]]
        文件s = [x for x in 条目 if not x[2]]
        目录s.sort(key=lambda x: x[0].lower())
        文件s.sort(key=lambda x: x[0].lower())
        条目 = 目录s + 文件s
        self.访问[os.path.abspath(目录).lower()] = \
            self.访问.get(os.path.abspath(目录).lower(), 0) + 1
        return 条目, 屏蔽数

    def 推栈(self, 目录: str, 待访: list) -> None:
        self.栈.append((目录, 待访))

    def 弹栈(self):
        return self.栈.pop() if self.栈 else None

    def 下一根(self):
        while self.根游标 < len(self.根s):
            r = self.根s[self.根游标]
            self.根游标 += 1
            if self.允许(r) and os.path.isdir(r):
                return r
        return None

    def 打开文件(self, 路径: str):
        """有界懒加载文本行（≤文件行限，4–200 字）；返回 行数 或 None（不可读/超限/二进制）。
        调用方需区分 (状态, None)：超时/异常 → 如实跳过。"""
        状态, 值 = 限时调用(self._打开文件本体, 路径)
        if 状态 == "完成":
            return 值
        self.异常.append((路径, 状态))
        return None

    def _打开文件本体(self, 路径: str):
        if os.path.getsize(路径) > 单文件字节限:
            self.异常.append((路径, "超限跳过"))
            return None
        with open(路径, "rb") as f:
            if b"\x00" in f.read(1024):
                self.异常.append((路径, "二进制跳过"))
                return None
        with open(路径, "r", encoding="utf-8", errors="ignore") as f:
            # ★GIL 纪律：只读前 512KB（行截取后仍满足 文件行限），Python 级解析量有界
            行s = [ln.strip() for ln in f.read(524288).splitlines()
                   if 4 <= len(ln.strip()) <= 200][:self.文件行限]
        self.当前文件, self.行s, self.行位置 = os.path.abspath(路径), 行s, 0
        self.文件细读 = self.文件略过 = 0
        return len(行s)

    def 下一行(self):
        if self.当前文件 is None or self.行位置 >= len(self.行s):
            return None
        ln = self.行s[self.行位置]
        self.行位置 += 1
        return ln

    def 文件读完(self) -> bool:
        return self.当前文件 is not None and self.行位置 >= len(self.行s)
