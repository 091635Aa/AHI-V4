# -*- coding: utf-8 -*-
"""meeting40/实施/快照.py —— 备B 中枢侧收尾：**快照（存档 / 载入 / 事件重放）**。

口径（终稿 §5 备B「中枢」条末句 + §7.1 持久化行）：
    「…+ 快照（**沿 `synthesis.py:1244` 事件重放**）」；
    §7.1：`快照(npz+manifest, temp+os.replace)`。
    `synthesis.py:1244 保存()` 的做法是：把 **seed + 教史 + 事件史 + D** 存档，`载入()` 重建后
    **按事件序列重放** ⇒「同 seed + 同事件序列 = 逐位再生」。本模块把这套做法**通用化**成备B 可用的快照件。

三条纪律（都可失败）：
  1. **原子替换**：先写 `路径.tmp`，再 `os.replace` 覆盖（同盘原子）⇒ 目标文件**要么旧要么新**，不会半截。
  2. **manifest + sha256**：`manifest.json` 记 `payload_sha256`（在**规范载荷**上算，不是 zip 文件字节——
     npz 的 zip 头带本地时间，算文件字节会不可复现）。载入时**重算比对**，不符 ⇒ **拒载**（fail-closed）。
  3. **事件重放**：`重放(事件s, 施)` 顺序施加，确定性；同一事件序列 ⇒ 同一状态（C13）。

安全：`np.load(..., allow_pickle=False)`（**不反序列化任意对象**）；只接受 numpy 数组 + JSON 标量。
本模块不 import 任何器官；**不写记忆**（它只碰文件）。

用法：
    快照.存("s.npz", dict(段s=arr, 拍=12), 事件s=[("记经历", {...}), …])
    状, 事 = 快照.载("s.npz")
    快照.重放(事, lambda e: 桥.记经历(**e))
"""
from __future__ import annotations

import hashlib
import json
import os

格式版本 = 1


def _规范(标量s: dict) -> bytes:
    return json.dumps(标量s, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str).encode("utf-8")


def _载荷指纹(数组s: dict, 标量s: dict, 事件s: list) -> str:
    """**规范载荷**指纹：数组按 (键, dtype, shape, 原始字节) 排序 + 标量 JSON + 事件 JSON。"""
    h = hashlib.sha256()
    for k in sorted(数组s):
        a = 数组s[k]
        h.update(("%s|%s|%s|" % (k, a.dtype.str, list(a.shape))).encode("utf-8"))
        h.update(a.tobytes())
    h.update(_规范(标量s))
    h.update(_规范(list(事件s or [])))
    return h.hexdigest()


def _拆数组(事s: list) -> tuple:
    """把**事件里的 numpy 值**提到顶层数组表，事件里留 `{"__数组__": key}`（保证事件可 JSON 化、
    重放时仍拿到**同一份数组**）。返回 `(事件s, 数组s)`。"""
    import numpy as np
    出, 数组s = [], {}
    for i, e in enumerate(事s or []):
        if isinstance(e, dict):
            ne = {}
            for k, v in e.items():
                if hasattr(v, "shape") or hasattr(v, "dtype"):
                    key = "事件数组/%d/%s" % (i, k)
                    数组s[key] = np.asarray(v)
                    ne[k] = {"__数组__": key}
                else:
                    ne[k] = v
            出.append(ne)
        else:
            出.append(e)
    return 出, 数组s


def _并数组(事s: list, 数组s: dict) -> list:
    def _还原(v):
        if isinstance(v, dict) and "__数组__" in v:
            k = v["__数组__"]
            if k not in 数组s:
                raise 快照损坏("事件引用的数组缺失：%s" % k)
            return 数组s[k]
        return v
    return [({k: _还原(v) for k, v in e.items()} if isinstance(e, dict) else e)
            for e in (事s or [])]


def 存(路径: str, 状态: dict = None, 事件s: list = None) -> dict:
    """把状态（numpy 数组 + JSON 标量）与事件序列存成 `路径`（npz）+ `路径.manifest.json`。

    · 事件里的 numpy 值会**自动提到 npz**（事件本身保持 JSON 可序列化），重放时原位还原。
    · **原子**：先写 `路径.tmp` → `os.replace`。返回 manifest（含 `payload_sha256`）。"""
    import numpy as np
    状态 = dict(状态 or {})
    数组s = {k: np.asarray(v) for k, v in 状态.items() if hasattr(v, "shape") or hasattr(v, "dtype")}
    标量s = {k: v for k, v in 状态.items() if k not in 数组s}
    事, 事件数组 = _拆数组(list(事件s or []))
    数组s.update(事件数组)
    sha = _载荷指纹(数组s, 标量s, 事)
    清单 = dict(格式版本=格式版本, 数组键s=sorted(数组s), 标量s=标量s, 事件s=事,
                payload_sha256=sha, 字节数=sum(int(a.nbytes) for a in 数组s.values()))
    tmp = str(路径) + ".tmp"
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **数组s)
        os.replace(tmp, str(路径))                       # ★原子替换（同盘）
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    with open(str(路径) + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(清单, f, ensure_ascii=False, indent=1, default=str)
    return 清单


def 载(路径: str, 校验: bool = True) -> tuple:
    """载入快照 → `(状态 dict, 事件列表)`。**校验不过 ⇒ 抛 `快照损坏`（fail-closed）**。"""
    import numpy as np
    mp = str(路径) + ".manifest.json"
    if not os.path.exists(str(路径)) or not os.path.exists(mp):
        raise 快照损坏("缺文件：%s 或 %s" % (路径, mp))
    with open(mp, encoding="utf-8") as f:
        清单 = json.load(f)
    if int(清单.get("格式版本", 0)) != 格式版本:
        raise 快照损坏("格式版本不符：%r ≠ %d" % (清单.get("格式版本"), 格式版本))
    数组s = {}
    with np.load(str(路径), allow_pickle=False) as z:
        for k in 清单.get("数组键s", []):
            if k not in z:
                raise 快照损坏("清单里的数组键缺失：%s" % k)
            数组s[k] = z[k]
    if 校验:
        sha = _载荷指纹(数组s, 清单.get("标量s", {}), 清单.get("事件s", []))
        if sha != 清单.get("payload_sha256"):
            raise 快照损坏("payload_sha256 不符（篡改或损坏）：%s ≠ %s"
                        % (sha[:16], str(清单.get("payload_sha256"))[:16]))
    状态 = dict(清单.get("标量s", {}))
    状态.update(数组s)
    return 状态, _并数组(清单.get("事件s", []), 数组s)


def 重放(事件s: list, 施) -> list:
    """按序施加事件（`施(事件) -> 任意`），返回结果列表。**同序列 ⇒ 同状态**（§7.1 事件重放）。"""
    出 = []
    for e in (事件s or []):
        出.append(施(e))
    return 出


class 快照损坏(Exception):
    """快照不可用（缺件/版本不符/校验失败）。**fail-closed**：宁可不载，不载坏档。"""


def 特征码(对象) -> str:
    return hashlib.sha256(json.dumps(对象, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode("utf-8")
                          ).hexdigest()


# ─────────────────────────── 自测 ───────────────────────────
def 自测(临时目录: str = None) -> dict:
    import shutil
    import tempfile
    import numpy as np
    目录 = 临时目录 or tempfile.mkdtemp(prefix="meeting40_快照_")
    路径 = os.path.join(目录, "s.npz")
    案 = {}
    try:
        段s = np.arange(24, dtype=np.float32).reshape(2, 12)
        状 = dict(段s=段s, 拍=12, 标记="备B")
        事 = [dict(类型="记经历", 文本="A"), dict(类型="记经历", 文本="B")]
        清1 = 存(路径, 状, 事)
        # ① 无 tmp 残留 + 目标存在
        案["无tmp残留"] = (not os.path.exists(路径 + ".tmp") and os.path.exists(路径))
        # ② round trip 逐位同
        状2, 事2 = 载(路径)
        案["数组逐位同"] = bool(np.array_equal(状2["段s"], 段s))
        案["标量同"] = (状2["拍"] == 12 and 状2["标记"] == "备B")
        案["事件同"] = (事2 == 事)
        案["payload指纹可复现"] = (清1["payload_sha256"]
                            == 存(路径, 状, 事)["payload_sha256"])
        # ③ 篡改数组 ⇒ 校验拒载（fail-closed）
        坏 = dict(状)
        坏["段s"] = (段s + 1.0).astype(np.float32)
        import numpy as _np
        with open(路径, "wb") as f:
            _np.savez_compressed(f, **{"段s": 坏["段s"]})          # 只改 npz，不动 manifest
        拒 = False
        try:
            载(路径)
        except 快照损坏:
            拒 = True
        案["篡改拒载"] = 拒
        # ④ 覆盖写 ⇒ 内容更新为新值（仍是原子路径）
        存(路径, dict(段s=np.zeros((1, 3), dtype=np.float32), 拍=99), [])
        状3, 事3 = 载(路径)
        案["覆盖更新"] = (状3["拍"] == 99 and 事3 == [] and 状3["段s"].shape == (1, 3))
        # ⑤ 事件重放确定性：同序列两次 ⇒ 逐位同
        记 = []
        重放([dict(文本=t) for t in ("甲", "乙", "丙")], lambda e: 记.append(e["文本"]))
        案["重放顺序"] = (记 == ["甲", "乙", "丙"])
        return dict(通过=bool(all(案.values())), 逐项=案, 清单=清1)
    finally:
        if 临时目录 is None:
            shutil.rmtree(目录, ignore_errors=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[快照] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
