# -*- coding: utf-8 -*-
"""meeting40/实施/代码任务.py —— 基础任务②：**写代码**（合成 → 落盘 → 执行自证）。

工程口径（可验证，绝不宣称「通用编程能力」）：
  · 输入是**结构化规格**：`{任务名, 函数名, 操作, 用例s=[{输入, 期望}], 说明}`；
  · 合成 = **确定性模板库**（有限几种操作）+ 参数替换 ⇒ 一个**完整可运行**的 .py
    （含 docstring、函数体、以及把用例嵌进文件自身的 `__main__` 自证块）；
  · 落盘后在**子进程**里真执行（用当前解释器 `sys.executable`），解析它打印的 JSON，
    与 `期望` 逐条比对 ⇒ **产物正确性有机器证据**，不是「看起来像」；
  · **同规格 ⇒ 同字节**（C13）；`写代码` 的产物 + 证据落盘到 产物目录。

诚实边界：模板库覆盖的操作是有限的（求和/计数/最大/去重排序/前缀累加/偶数筛选/计数满足）。
它**不会**从自然语言写任意程序；「写代码」在这里 = **把结构化规格落成能跑且自证正确的代码**。
零外挂模型（C1/C10 守住）：模板 + JSON 自证，没有任何生成式模型参与。

用法：
    果 = 写代码(dict(任务名="统计网页段数", 函数名="段数", 操作="计数",
                     用例s=[dict(输入=[1,2,3], 期望=3)]), 产物目录="…")
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

操作s = {
    "求和": ("求和", "def {fn}(xs):\n    \"\"\"返回所有元素之和（空序列=0）。\"\"\"\n    return sum(xs)\n"),
    "计数": ("计数", "def {fn}(xs):\n    \"\"\"返回元素个数。\"\"\"\n    return len(xs)\n"),
    "最大值": ("最大值", "def {fn}(xs):\n    \"\"\"返回最大元素（空序列=0）。\"\"\"\n    return max(xs) if xs else 0\n"),
    "去重排序": ("去重排序", "def {fn}(xs):\n    \"\"\"去重并升序排列。\"\"\"\n    return sorted(set(xs))\n"),
    "前缀累加": ("前缀累加", "def {fn}(xs):\n    \"\"\"返回前缀累加序列（长度与输入相同）。\"\"\"\n"
                            "    出, 累 = [], 0\n    for x in xs:\n        累 += x\n        出.append(累)\n    return 出\n"),
    "偶数筛选": ("偶数筛选", "def {fn}(xs):\n    \"\"\"只保留偶数（原序）。\"\"\"\n"
                            "    return [x for x in xs if x % 2 == 0]\n"),
}

文件头 = ('# -*- coding: utf-8 -*-\n'
        '"""本文件由 AHI（meeting40 代码任务）依据**结构化规格**合成，非人手编写。\n\n'
        '规格指纹：{指纹}\n操作：{操作}\n说明：{说明}\n'
        '自证方式：运行本文件，会逐条执行内嵌用例并打印 JSON（ok=全部通过）。\n"""\n'
        'from __future__ import annotations\n\nimport json\n\n')

自证块 = ('\n_用例s = {用例s}\n\n\n'
        'if __name__ == "__main__":\n'
        '    果 = []\n'
        '    for i, c in enumerate(_用例s):\n'
        '        try:\n'
        '            得 = {fn}(c["输入"])\n'
        '            ok = (得 == c["期望"])\n'
        '        except Exception as ex:\n'
        '            得, ok = "异常:%r" % (ex,), False\n'
        '        果.append(dict(序号=i, 期望=c["期望"], 实得=得, ok=bool(ok)))\n'
        '    print(json.dumps(dict(用例s=果, ok=all(x["ok"] for x in 果)), ensure_ascii=False))\n')


def 合成(spec: dict) -> str:
    """结构化规格 → 完整 .py 源码（确定性）。未知操作 ⇒ 抛 KeyError（fail-fast）。"""
    操作 = str(spec.get("操作", ""))
    if 操作 not in 操作s:
        raise KeyError("模板库没有这个操作：%r（现有：%s）" % (操作, sorted(操作s)))
    名, 体 = 操作s[操作]
    fn = str(spec.get("函数名") or "处理")
    if not fn.isidentifier():
        raise ValueError("函数名必须是合法标识符：%r" % fn)
    用例s = list(spec.get("用例s") or [])
    指纹 = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), default=str).encode("utf-8")
                        ).hexdigest()[:16]
    头 = 文件头.format(指纹=指纹, 操作=名, 说明=str(spec.get("说明", ""))[:120])
    源 = 头 + 体.format(fn=fn) + 自证块.format(用例s=json.dumps(用例s, ensure_ascii=False), fn=fn)
    return 源


def 写代码(spec: dict, 产物目录: str = None, 文件名: str = None, 桥=None,
         学习迹=None) -> dict:
    """合成 → 落盘 → **子进程执行自证** → 证据 JSON。可选把这次经历写进记忆/学习迹。"""
    源 = 合成(spec)
    目录 = 产物目录 or os.path.join(os.path.dirname(os.path.abspath(__file__)), "产物")
    os.makedirs(目录, exist_ok=True)
    名 = 文件名 or ("代码_%s" % hashlib.sha256(
        json.dumps(spec, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8])
    路径 = os.path.join(目录, 名 + ".py")
    with open(路径, "w", encoding="utf-8") as f:
        f.write(源)
    # 编译期检查（语法）
    try:
        compile(源, 路径, "exec")
        语法 = True
    except SyntaxError as ex:
        语法 = False
        编译错 = repr(ex)
    # ★执行自证：子进程真跑
    环 = dict(os.environ)
    环["PYTHONIOENCODING"] = "utf-8"
    出, 码 = "", None
    try:
        r = subprocess.run([sys.executable, 路径], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60, env=环,
                           cwd=目录)
        码, 出 = r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        出 = "【超时>60s】"
    except Exception as ex:
        出 = "【异常】%r" % (ex,)
    判 = None
    for 行 in reversed([x.strip() for x in (出 or "").splitlines() if x.strip()]):
        try:
            判 = json.loads(行)
            break
        except Exception:
            continue
    用例果 = (判 or {}).get("用例s", [])
    全过 = bool(语法 and 判 and 判.get("ok") and 码 == 0)
    期望数 = len(list(spec.get("用例s") or []))
    证据 = dict(任务="写代码", 规格=dict(操作=spec.get("操作"), 函数名=spec.get("函数名"),
                                    任务名=spec.get("任务名"), 用例数=期望数),
                产物=dict(py=路径, 字节=len(源.encode("utf-8")),
                         sha256=hashlib.sha256(源.encode("utf-8")).hexdigest()),
                语法通过=bool(语法), 子进程退出码=码, 用例结果=用例果,
                用例全过=全过, 执行输出尾=出[-400:],
                口径="结构化规格 → 模板合成 → 子进程执行 → 用例逐条比对")
    if not 语法:
        证据["编译错"] = 编译错

    # 「学进去」：任务经历入记忆（带向量，可检索）；学习结论入学习迹（物理分离，D3-2/H5）
    if 桥 is not None:
        try:
            import numpy as np
            txt = "代码任务：%s（操作=%s，用例 %d 条，%s）" % (
                str(spec.get("任务名", "")), str(spec.get("操作", "")), 期望数,
                "全过" if 全过 else "有未过")
            v = np.zeros(1024, dtype=np.float32)
            v[int(证据["产物"]["sha256"][:8], 16) % 1024] = 1.0
            rid = 桥.记经历(拍=0, 类型="任务经历", 文本=txt, 来源="代码任务", 向量=v)
            证据["入记忆"] = dict(rid=rid, 文本=txt)
        except Exception as ex:
            证据["入记忆"] = dict(异常=repr(ex))
    if 学习迹 is not None:
        try:
            号 = 学习迹.放快照(["代码任务", str(spec.get("操作", ""))], 1,
                             "我学会了把「%s」规格落成可运行代码（用例 %d 条，%s）"
                             % (str(spec.get("操作", "")), 期望数, "全过" if 全过 else "有未过"),
                             束=None)
            证据["入学习迹"] = dict(号=号, 记忆库大小=学习迹.记忆库大小())
        except Exception as ex:
            证据["入学习迹"] = dict(异常=repr(ex))

    证据路径 = os.path.join(目录, 名 + "_证据.json")
    with open(证据路径, "w", encoding="utf-8") as f:
        json.dump(证据, f, ensure_ascii=False, indent=1, default=str)
    证据["证据文件"] = 证据路径
    return 证据


# ─────────────────────────── 自测 ───────────────────────────
def 自测() -> dict:
    import shutil
    import tempfile
    目录 = tempfile.mkdtemp(prefix="meeting40_代码_")
    案 = {}
    try:
        规 = dict(任务名="统计网页段数", 函数名="段数", 操作="计数",
                 说明="数一数正文有几段",
                 用例s=[dict(输入=[1, 2, 3], 期望=3), dict(输入=[], 期望=0),
                      dict(输入=[9], 期望=1)])
        果 = 写代码(规, 产物目录=目录, 文件名="A")
        案["落盘"] = bool(os.path.exists(果["产物"]["py"]))
        案["语法通过"] = bool(果["语法通过"])
        案["子进程退出0"] = (果["子进程退出码"] == 0)
        案["用例全过"] = bool(果["用例全过"])
        案["用例数对"] = (len(果["用例结果"]) == 3)
        # 同规格 ⇒ 同字节
        案["字节可复现"] = (写代码(规, 产物目录=目录, 文件名="B")["产物"]["sha256"]
                       == 果["产物"]["sha256"])
        # sham：期望故意写错 ⇒ 必须判失败（证明判据能失败）
        坏 = dict(规, 用例s=[dict(输入=[1, 2, 3], 期望=99)])
        案["反例必失败"] = (写代码(坏, 产物目录=目录, 文件名="C")["用例全过"] is False)
        # 未知操作 ⇒ fail-fast
        拒 = False
        try:
            合成(dict(操作="不存在的操作"))
        except KeyError:
            拒 = True
        案["未知操作拒"] = 拒
        # 非法函数名 ⇒ 拒
        拒2 = False
        try:
            合成(dict(操作="计数", 函数名="1坏名"))
        except ValueError:
            拒2 = True
        案["非法函数名拒"] = 拒2
        return dict(通过=bool(all(案.values())), 逐项=案, 证据=果, 目录=目录)
    finally:
        shutil.rmtree(目录, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    果 = 自测()
    print(json.dumps({k: v for k, v in 果.items() if k != "证据"}, ensure_ascii=False, indent=1,
                     default=str))
    print("[代码任务] 自测：%s" % ("PASS" if 果["通过"] else "FAIL"))
    raise SystemExit(0 if 果["通过"] else 1)
