# -*- coding: utf-8 -*-
"""meeting62/产物/原生路线/探针_HDC零外挂审计.py —— 路线③「规模化学习→内部网格」最小可证伪实验。

本件**扩展**只读件 `团队/HDC内在动力学/探针_谐振音素格.py`（不改它，C14），补两件它没做的事：
  ① **零外挂三断言自动审计**（HDC R1 §2 断言 A/B/C）：
     A 不读参考音频：AST 扫「open(...,'rb') 式音频读入」＋ **运行期 `builtins.open` 钩子**（读入计数须 0）；
     B 不载模型/权重：AST 扫 import 名（框架/外挂名单）＋ 扫**权重/数据加载调用与检查点·张量文件后缀**
       ＋ `sys.modules` 运行期断言 ＋ **参数只来自显式种子的可证伪检验**（同种子两实例 W表逐位同 /
       **异种子必定改变由种子派生的算子** 绑·解绑·读 —— 若存在隐藏外部参数源，它们不会随种子变）；
     C 确定性：同一调用两趟**记录逐位同**（C13）。
  ② **Sham-直达对照**（HDC R1 §2 列出的**最便宜 must-fail**）：去掉谐振迭代、只 `读(码本A, target)`
     ⇒ 命中率**必须掉到 chance ≈ 1/每因子数**（主臂 = 谐振 100%）；
     并附 `Sham-置乱`（探针原有 P3 臂）作对照。

红线：C1 无 attention/QKV/softmax（AST + 片段拼装禁词自扫）｜ C9 不涉及 ｜ C10 无权重/checkpoint、
不引 TTS/RVC/声码器 ｜ C13 补齐（**本件把「两趟逐位同」推广到「同种子/异种子」的可证伪对照**）｜
C14 本件**只新增于 meeting62/**，只读 import 既有件。
探针 CPU-only、单次 <10s、不占声卡、不启动 RVC/训练、不占 GPU。
标记口径：[实测] 本件真跑读数 ｜ [源码] 读自源码 ｜ [推导] ｜ [提案] ｜ 未测。
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

本目录 = os.path.dirname(os.path.abspath(__file__))                     # …/meeting62/产物/原生路线
MEET62 = os.path.dirname(os.path.dirname(本目录))                        # …/meeting62
AHI4 = os.path.dirname(MEET62)
探针路径 = os.path.join(MEET62, "团队", "HDC内在动力学", "探针_谐振音素格.py")


def 加载探针():
    """只读加载既有探针模块（不改它）：复用 载核/能力核/生成码本/跑/D默认/种子。"""
    spec = importlib.util.spec_from_file_location("HDC谐振探针_只读_62", 探针路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


探针 = 加载探针()
能力核 = 探针.能力核
生成码本 = 探针.生成码本
D默认 = 探针.D默认
种子 = 探针.种子

# ── 审计名单（片段拼装，避免扫描器命中本文件自身的审计字符串）──
_禁词片段 = (("at", "tention"), ("at", "tn"), ("q", "kv"), ("soft", "max"),
             ("multi", "head"), ("trans", "former"), ("rw", "kv"),
             ("位置", "编码"), ("to", "rch"))
_外挂片段 = (("py", "ttsx3"), ("sa", "pi"), ("r", "vc"), ("so", "vits"),
             ("whis", "per"), ("fun", "asr"), ("li", "brosa"), ("sound", "file"),
             ("torch", "audio"), ("audio", "read"), ("py", "dub"))
_权重片段 = (("from", "file"), ("load", "txt"), ("read_", "csv"), ("job", "lib"),
             ("pick", "le"), ("np", ".load"))
_权重后缀片段 = ((".", "p", "th"), (".", "on", "nx"), (".", "safet", "ensors"),
              (".", "ck", "pt"), (".", "p", "t"), (".", "n", "py"),
              (".", "n", "pz"), (".", "b", "in"))
_权重后缀 = tuple("".join(p) for p in _权重后缀片段)


def 禁词() -> list:
    return ["".join(p) for p in _禁词片段]


def 外挂名单() -> list:
    return ["".join(p) for p in _外挂片段]


def 权重API() -> list:
    return ["".join(p) for p in _权重片段]


# ═══════════════════════════════ 1. Zero-external 三断言审计 ═══════════════════════════════

def AST审计(路径列) -> dict:
    """静态审计：被禁 import / 外挂 import / 音频读入 open(...,'rb') / 权重·数据加载调用与后缀。"""
    框架 = set(禁词())
    外挂 = set(外挂名单())
    权重名 = {w for w in 权重API() if w}
    命中框架, 命中外挂, 命中读入, 命中权重 = [], [], [], []
    for p in 路径列:
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8-sig") as f:
            原文 = f.read()
        树 = ast.parse(原文)
        小写 = 原文.lower()
        for 后缀 in _权重后缀:
            if 后缀 in 小写:
                命中权重.append((os.path.basename(p), 后缀))
        for node in ast.walk(树):
            if isinstance(node, ast.Import):
                名s = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                名s = [node.module or ""]
            else:
                名s = []
            for nm in 名s:
                根 = nm.split(".")[0].lower()
                if 根 in 框架:
                    命中框架.append((os.path.basename(p), nm))
                if 根 in 外挂:
                    命中外挂.append((os.path.basename(p), nm))
            if isinstance(node, ast.Call):
                fn = node.func
                名 = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else "")
                if 名 == "open":
                    模s = [str(c.value) for c in node.args
                           if isinstance(c, ast.Constant) and isinstance(c.value, str)]
                    if any(("b" in m and "r" in m and "+" not in m) for m in 模s):
                        命中读入.append((os.path.basename(p), 模s))
                if 名 in 权重名 or (名 == "load" and isinstance(fn, ast.Attribute)):
                    命中权重.append((os.path.basename(p), 名))
    return dict(文件数=len([p for p in 路径列 if os.path.isfile(p)]),
                命中框架=命中框架, 命中外挂模型=命中外挂,
                命中音频读入=命中读入, 命中权重加载=命中权重,
                通过=bool(not 命中框架 and not 命中外挂 and not 命中读入 and not 命中权重))


class _读入钩子:
    """运行期证明「零参考读入」：劫持 builtins.open，记录一切以读模式打开的路径。"""

    def __init__(self):
        self.读入 = []
        self._原 = None

    def __enter__(self):
        self._原 = builtins.open
        记 = self.读入
        原 = self._原

        def 钩(file, mode="r", *a, **k):
            if "r" in str(mode) and "+" not in str(mode):
                记.append(str(file))
            return 原(file, mode, *a, **k)

        builtins.open = 钩
        return self

    def __exit__(self, *exc):
        builtins.open = self._原
        return False


def 参数来源检验() -> dict:
    """[可证伪] 「参数只来自 default_rng(显式种子)」：同种子两实例 W表逐位同，**异种子必须异**。"""
    核1 = 能力核(D=D默认, 种子=种子)
    核2 = 能力核(D=D默认, 种子=种子)
    核3 = 能力核(D=D默认, 种子=种子 + 1)
    名 = sorted(核1.W表)
    同 = all(np.array_equal(核1.W表[k], 核2.W表[k]) for k in 名)
    变化 = [k for k in 名 if not np.array_equal(核1.W表[k], 核3.W表[k])]
    常数 = [k for k in 名 if k not in 变化]
    指纹 = hashlib.sha256(b"".join(核1.W表[k].tobytes() for k in 名)).hexdigest()[:16]
    须变 = {"绑", "解绑", "读"}          # 由 default_rng(显式种子) 派生的算子
    通过 = bool(同 and 须变 <= set(变化))
    return dict(算子数=len(名), 同种子两实例逐位同=bool(同),
                异种子变化算子=变化, 异种子不变算子_内部常数=常数,
                异种子须变算子齐备=bool(须变 <= set(变化)),
                W表指纹=指纹, 通过=通过)


def 运行期框架断言() -> dict:
    根s = {s.split(".")[0].lower() for s in sys.modules}
    命中 = sorted((根s & set(禁词())) | (根s & set(外挂名单())))
    return dict(已加载模块数=len(sys.modules), 命中=命中, 通过=bool(not 命中))


# ═══════════════════════════════ 2. Sham-直达对照臂 ═══════════════════════════════

def 直达臂(每因子数=8, 组数=24):
    """Sham-直达：**不做谐振迭代**，直接 `读(码本A, target)` 取最近原子 ⇒ 必须掉到 chance。"""
    核 = 能力核(D=D默认, 种子=种子)
    码本A = 生成码本(每因子数, D默认, 种子 + 1)
    码本B = 生成码本(每因子数, D默认, 种子 + 2)
    rng = np.random.default_rng(种子 + 7)
    命中 = 0
    记录 = []
    for _ in range(组数):
        i = int(rng.integers(0, 每因子数))
        j = int(rng.integers(0, 每因子数))
        target = 核.绑(码本A[i], 码本B[j])
        ai = int(核.读(码本A, target, k=1)["索引"][0])
        命中 += int(ai == i)
        记录.append([i, j, ai, bool(ai == i)])
    return dict(命中率=命中 / 组数, 记录=记录, 每因子数=每因子数, 组数=组数,
                chance=1.0 / 每因子数)


# ═══════════════════════════════ 3. 主流程 ═══════════════════════════════

def 跑一遍() -> dict:
    路径列 = [os.path.abspath(__file__), 探针路径]
    参数来源 = 参数来源检验()
    读入列 = []
    with _读入钩子() as 钩:
        主1 = 探针.跑(8, 12, 24, False)      # 主臂：谐振迭代
        直1 = 直达臂(8, 24)                  # Sham-直达（最便宜 must-fail）
        乱1 = 探针.跑(8, 12, 24, True)       # Sham-置乱（探针原 P3 臂）
        读入列 = list(钩.读入)
    静 = AST审计(路径列)
    运 = 运行期框架断言()
    零外挂 = dict(
        断言A_零音频读入=dict(静态命中=静["命中音频读入"], 运行期读入路径=读入列,
                        通过=bool(not 静["命中音频读入"] and not 读入列)),
        断言B_无外部模型权重=dict(静态命中框架=静["命中框架"], 静态命中外挂模型=静["命中外挂模型"],
                          静态命中权重加载=静["命中权重加载"], 运行期=运,
                          参数来源=参数来源,
                          通过=bool(not 静["命中框架"] and not 静["命中外挂模型"]
                                  and not 静["命中权重加载"] and 运["通过"] and 参数来源["通过"])),
        审计文件=路径列,
    )
    return dict(主臂=主1, 直达=直1, 置乱=乱1, 零外挂=零外挂)


def 主():
    果1 = 跑一遍()
    果2 = 跑一遍()
    同 = bool(果1["主臂"]["记录"] == 果2["主臂"]["记录"]
              and 果1["直达"]["记录"] == 果2["直达"]["记录"]
              and json.dumps(果1, ensure_ascii=False, sort_keys=True)
              == json.dumps(果2, ensure_ascii=False, sort_keys=True))

    chance = 果1["直达"]["chance"]
    判据 = dict(
        断言A_零音频读入=果1["零外挂"]["断言A_零音频读入"]["通过"],
        断言B_无外部模型权重=果1["零外挂"]["断言B_无外部模型权重"]["通过"],
        断言C_两趟逐位同=同,
        主臂恢复率_ge_0p9=bool(果1["主臂"]["恢复率"] >= 0.9),
        Sham直达_须掉到chance=bool(果1["直达"]["命中率"] <= 2.0 * chance),
        Sham置乱_须退化=bool(果1["置乱"]["恢复率"] < 果1["主臂"]["恢复率"] - 1e-9),
    )
    报 = dict(
        工具="探针_HDC零外挂审计.py",
        扩展="只读件 团队/HDC内在动力学/探针_谐振音素格.py（不改它）",
        主臂=dict(说明="谐振迭代（探针原 P1 臂）", 恢复率=果1["主臂"]["恢复率"],
                直读命中率=果1["主臂"]["直读命中率"], 步数=果1["主臂"]["步数"],
                组数=果1["主臂"]["组数"], 每因子数=果1["主臂"]["每因子数"]),
        Sham_直达=dict(说明="去掉谐振迭代、只读乘积向量 ⇒ 必须掉到 chance",
                    命中率=果1["直达"]["命中率"], chance=chance,
                    每因子数=果1["直达"]["每因子数"], 组数=果1["直达"]["组数"]),
        Sham_置乱=dict(说明="置乱熵源（探针原 P3 臂）", 恢复率=果1["置乱"]["恢复率"]),
        零外挂=果1["零外挂"],
        判据=判据,
    )
    报["J4_确定性"] = dict(两趟逐位同=同,
                        主臂记录sha256=hashlib.sha256(
                            json.dumps(果1["主臂"]["记录"], ensure_ascii=False).encode()).hexdigest()[:16])
    报["通过"] = bool(all(判据.values()))
    with open(os.path.join(本目录, "读数_HDC零外挂审计.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)

    print("=" * 88)
    print("探针_HDC零外挂审计 —— 路线③「规模化学习→内部网格」零外挂三断言 + Sham-直达")
    print("=" * 88)
    print("主臂(谐振) 恢复率 = %.4f ｜ Sham-直达 命中率 = %.4f（chance = %.4f = 1/%d）"
          % (报["主臂"]["恢复率"], 报["Sham_直达"]["命中率"],
             报["Sham_直达"]["chance"], 报["Sham_直达"]["每因子数"]))
    print("Sham-置乱 恢复率 = %.4f（必须 < 主臂）" % 报["Sham_置乱"]["恢复率"])
    print("断言A：静态音频读入命中=%s ｜ 运行期读入路径=%s"
          % (报["零外挂"]["断言A_零音频读入"]["静态命中"],
             报["零外挂"]["断言A_零音频读入"]["运行期读入路径"]))
    参 = 报["零外挂"]["断言B_无外部模型权重"]["参数来源"]
    print("断言B：静态框架=%s 外挂=%s 权重加载=%s ｜ 运行期框架命中=%s"
          % (报["零外挂"]["断言B_无外部模型权重"]["静态命中框架"],
             报["零外挂"]["断言B_无外部模型权重"]["静态命中外挂模型"],
             报["零外挂"]["断言B_无外部模型权重"]["静态命中权重加载"],
             报["零外挂"]["断言B_无外部模型权重"]["运行期"]["命中"]))
    print("        参数来源：同种子逐位同=%s ｜ 异种子变化算子=%s ｜ 不变(内部常数)=%s ｜ 指纹=%s"
          % (参["同种子两实例逐位同"], 参["异种子变化算子"],
             参["异种子不变算子_内部常数"], 参["W表指纹"]))
    print("断言C 两趟逐位同 = %s" % 同)
    print("判据：" + " ".join("%s=%s" % (k, v) for k, v in 判据.items()))
    print("通过 = %s" % 报["通过"])
    return 报


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    主()
