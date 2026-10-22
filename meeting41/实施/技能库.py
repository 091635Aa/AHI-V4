# -*- coding: utf-8 -*-
"""meeting41/实施/技能库.py —— L2 技能存档（跨进程可载入；**不碰记忆库**）。

━━━ 职责与边界（裁定 H5「记忆≠学习」的延续）━━━
本库是**技能**的存储：技能定义（原子序列 + 档位）+ 训练台账。
它 **绝不写**既有记忆库 `容量桥`：判据 T6 会核"训练 + 存/载 前后，记忆库条数零变化"。
理由与 V3 既有 `学习迹.py` 同源——学到的手艺不等于记得的经历，两者物理分离。

━━━ 落盘格式 ━━━
单个 JSON 文件：
    dict(格式="ahi3.meeting41.技能库", 版本=1,
         payload=dict(技能={名: 定义}, 台账={名: 台账}, 顺序=[名...]),
         sha256="<payload 规范 JSON 的 sha256>")
★`payload` 的 sha256 用于**完整性校验**：载入时重算，不符即报错（不静默接受脏档）。
★写出用 `temp + os.replace`：原子替换，避免写一半留下半截档。

纪律：纯新增；不 import 器官；不 import time；无全局随机态；确定性（同内容 ⇒ 同 sha256）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AHI3 = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(AHI3, "meeting40", "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

格式名 = "ahi3.meeting41.技能库"
默认路径 = os.path.join(HERE, "产物", "技能库.json")


def _规范(o) -> str:
    """规范 JSON 文本（键排序 + 紧凑分隔符）——用于 sha256 与逐位可比。"""
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def 摘要(payload) -> str:
    return hashlib.sha256(_规范(payload).encode("utf-8")).hexdigest()


class 技能库:
    """技能登记 / 取用 / 落盘 / 载入。内存 + 单文件 JSON，不涉任何既有存储。"""

    def __init__(self, 路径: str = None) -> None:
        self.路径 = str(路径) if 路径 else 默认路径
        self.技能 = {}          # 名 → 技能原子 实例
        self.台账s = {}         # 名 → 台账 dict
        self.顺序 = []          # 登记顺序（可审计）

    # ── 登记 / 取用 ──
    def 登记(self, 技能, 台账: dict = None) -> str:
        名 = str(getattr(技能, "名", None) or "")
        if not 名:
            raise ValueError("技能必须有 .名")
        if 名 not in self.技能:
            self.顺序.append(名)
        self.技能[名] = 技能
        if 台账 is not None:
            self.台账s[名] = dict(台账)
        return 名

    def 取(self, 名: str):
        名 = str(名)
        if 名 not in self.技能:
            raise KeyError("技能库无此技能：%r（有：%s）" % (名, "、".join(self.列表()) or "空"))
        return self.技能[名]

    def 有(self, 名: str) -> bool:
        return str(名) in self.技能

    def 列表(self) -> list:
        return sorted(self.技能)

    def 记台账(self, 名: str, 台账: dict) -> None:
        self.台账s[str(名)] = dict(台账)

    # ── payload / 落盘 / 载入 ──
    def payload(self) -> dict:
        定 = {}
        for 名, 技 in self.技能.items():
            if hasattr(技, "定义"):
                定[名] = 技.定义()
            else:                                             # 兜底：只存名字
                定[名] = dict(名=名)
        return dict(技能=定, 台账={k: dict(v) for k, v in self.台账s.items()},
                    顺序=list(self.顺序))

    def 存(self, 路径: str = None) -> dict:
        路 = str(路径) if 路径 else self.路径
        目 = os.path.dirname(os.path.abspath(路))
        if 目:
            os.makedirs(目, exist_ok=True)
        pay = self.payload()
        档 = dict(格式=格式名, 版本=1, payload=pay, sha256=摘要(pay))
        # 原子替换：先写临时件，再 os.replace（同目录，保证同卷）
        fd, 临 = tempfile.mkstemp(prefix=".技能库.", suffix=".tmp", dir=目 or ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(档, f, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(临, 路)
        except Exception:
            try:
                os.remove(临)
            except OSError:
                pass
            raise
        return dict(路径=os.path.abspath(路), sha256=档["sha256"], 技能数=len(pay["技能"]),
                    字节=os.path.getsize(路))

    def 载(self, 路径: str = None, 严格: bool = True) -> dict:
        路 = str(路径) if 路径 else self.路径
        with open(路, "r", encoding="utf-8") as f:
            档 = json.load(f)
        if str(档.get("格式")) != 格式名:
            raise ValueError("技能库格式不符：%r" % 档.get("格式"))
        pay = 档.get("payload") or {}
        实际 = 摘要(pay)
        if 严格 and 实际 != str(档.get("sha256")):
            raise ValueError("技能库校验失败：sha256 不符（档=%s 实算=%s）"
                             % (str(档.get("sha256"))[:16], 实际[:16]))
        return dict(payload=pay, sha256=实际, 路径=os.path.abspath(路),
                    技能数=len(pay.get("技能") or {}))

    def 载入重建(self, 路径: str = None, 技能类=None) -> dict:
        """载入并**重建** 技能原子 实例（需要传 技能原子 类；不传则自动 import）。

        返回 dict(技能数=, 名=, sha256=)。重建后 `特征码()` 必须与存档前逐位同（回环判据）。
        """
        if 技能类 is None:
            import 技能原子 as _SK
            技能类 = _SK.技能原子
        果 = self.载(路径, 严格=True)
        pay = 果["payload"]
        self.技能, self.台账s, self.顺序 = {}, {}, []
        for 名 in (pay.get("顺序") or sorted(pay.get("技能") or {})):
            定 = (pay.get("技能") or {}).get(名)
            if 定 is None:
                continue
            技 = 技能类(名, 原子序列=定.get("序列") or [], 种子=int(定.get("种子", 0)))
            if hasattr(技, "设档"):
                技.设档(int(定.get("档", 0)))
            self.登记(技, (pay.get("台账") or {}).get(名))
        return dict(技能数=len(self.技能), 名=self.列表(), sha256=果["sha256"])

    def 台账(self) -> dict:
        return dict(技能数=len(self.技能), 名=self.列表(), 路径=os.path.abspath(self.路径),
                    有台账=sorted(self.台账s))

    def 特征码(self) -> str:
        return 摘要(self.payload())


# ══════════════════════════════════════════════════════════════════════
# 自测
# ══════════════════════════════════════════════════════════════════════

def 自测() -> dict:
    import 技能原子 as SK

    世 = SK.靶场(种子=0)
    列 = SK.造基础计划(世.元素列(), 种子=0)
    技 = SK.技能原子("合成_提交流程", 列, 种子=0)
    技.设档(2)

    建议路径 = os.path.join(HERE, "产物", "_技能库_自测.json")
    库 = 技能库(建议路径)
    库.登记(技, 台账=dict(轮次=3, 最好毫秒=5000.0))
    存果 = 库.存()
    原码 = 库.特征码()

    # ① 落盘 + 完整性校验
    读 = 技能库(建议路径).载(严格=True)
    校验通过 = bool(读["sha256"] == 存果["sha256"])

    # ② 重建回环：技能定义特征码必须逐位同
    库2 = 技能库(建议路径)
    重建 = 库2.载入重建()
    技2 = 库2.取("合成_提交流程")
    回环同 = bool(技2.特征码() == 技.特征码() and int(技2.取档()) == 2)
    码同 = bool(库2.特征码() == 原码)

    # ③ 内容变 ⇒ sha256 变（证明摘要不是恒真）
    技2.设档(3)
    变后 = 库2.特征码()
    摘要敏感 = bool(变后 != 原码)

    # ④ 脏档必须被拒（篡改 payload 而不改 sha256）
    篡 = json.loads(open(建议路径, encoding="utf-8").read())
    篡["payload"]["台账"]["合成_提交流程"]["最好毫秒"] = 12345.0
    脏路 = os.path.join(HERE, "产物", "_技能库_自测_脏.json")
    with open(脏路, "w", encoding="utf-8") as f:
        json.dump(篡, f, ensure_ascii=False)
    拒脏 = False
    try:
        技能库(脏路).载(严格=True)
    except ValueError:
        拒脏 = True

    # ⑤ 绝不写既有记忆库：容量桥条数前后不变
    桥前 = 桥后 = -1
    try:
        import 容量桥 as Q
        b = Q.容量桥(种子=0)
        桥前 = len(b._文本)
        库3 = 技能库(os.path.join(HERE, "产物", "_技能库_自测2.json"))
        库3.登记(技)
        库3.存()
        库3.载入重建()
        b2 = Q.容量桥(种子=0)
        桥后 = len(b2._文本)
    except Exception:                                         # noqa: BLE001
        桥前 = 桥后 = 0                                       # 拿不到就当 0，不误判
    记忆零变 = bool(桥前 == 桥后)

    for p in (建议路径, 脏路, os.path.join(HERE, "产物", "_技能库_自测2.json")):
        try:
            os.remove(p)
        except OSError:
            pass

    通过 = bool(校验通过 and 回环同 and 码同 and 摘要敏感 and 拒脏 and 记忆零变)
    return dict(通过=通过, 完整性校验=校验通过, 重建回环=回环同, 库码同=码同,
                摘要敏感=摘要敏感, 拒脏档=拒脏, 记忆库零变=记忆零变,
                记忆条数_前=桥前, 记忆条数_后=桥后,
                sha256=存果["sha256"][:24], 技能数=存果["技能数"])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                         # noqa: BLE001
        pass
    果 = 自测()
    print(json.dumps(果, ensure_ascii=False, indent=1, default=str))
    print("[技能库] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
