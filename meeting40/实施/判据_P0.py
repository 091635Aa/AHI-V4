# -*- coding: utf-8 -*-
"""判据_P0.py —— meeting40 终稿 §8「P0 基线冻结」验收（[正交] 必做项，逐条可失败）。

判据清单（每条都有对照臂/反例，且都写机器可读产物）：
  P0-1 C13 修复：**跨 PYTHONHASHSEED** 同键投影阵特征码相等（4 次独立子进程）；
       sham 对照臂 = 回退旧路径（`AHI_C13_旧哈希=1`）⇒ 两个 hash seed 下**必须不等**。
  P0-2 时基：40/150/300ms 落在 0.4/1.5/3.0 拍（定点换算）；重放逐位；**把 time.time 打坏**仍可打点。
  P0-3 事件契约：禁字段·枚举·动作/不可逆缺令牌 fail-closed·内容指纹·仲裁全序。
  P0-4 R 契约：空/冲突拒答、事实级须引用、医学断言级上限、**默认关不接线** + 真 容量桥 只读适配。
  P0-5 provenance 写闸：三闸走查 + 与 `内核_自我回路` 的**真实枚举**逐字对齐。
  P0-6 测量台：自证（能判 FAIL、能抓不确定源、sham 有效、留出集同判）。
  P0-7 C14：10 文件帧冻结 MD5 + §13.4 两文件 sha256 + `通用接口.py` 改动面（删除行数=1）。
  P0-8 旧判据 A1–A7：与改前基线逐项比对（计时/并发计数类字段按口径排除，如实登记）。

用法：python 判据_P0.py     产物：P0_结果.json
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # AHI3 根
M38 = os.path.join(ROOT, "meeting38")
M38S = os.path.join(M38, "实施")
M39S = os.path.join(ROOT, "meeting39", "实施")
for _p in (HERE, M38, M38S, M39S, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 事件契约                                                   # noqa: E402
import 时基                                                       # noqa: E402
import 测量台                                                     # noqa: E402
import 回忆契约                                                   # noqa: E402

基线目录 = os.path.join(HERE, "基线")
通用接口路径 = os.path.join(M38S, "通用接口.py")
通用接口备份 = os.path.join(ROOT, "备份_迁移前", "meeting38", "实施", "通用接口.py.P0前")

# §13.4 冻结哈希（终稿口径；sha256 全值 + 字节数）
冻结哈希 = {
    "meeting38/实施/输出层.py": dict(size=16139,
        sha256="4a301439f30876be96a8f2bb434a031b910deef33675315196bc8aa890ec40c3"),
    "meeting38/实施/学习_三阶段.py": dict(size=13840,
        sha256="58364882cec3cde08fad61049fd18bdfb038e9257fc867037478a0c220445c1e"),
}
# P0-8 比对时排除的字段（墙钟/并发计数：天然逐次变动，非机制差异）
排除键 = ("耗时s", "读写次数", "最长写ms")


# ─────────────────────────── 小工具 ───────────────────────────
def sha256文件(路径: str) -> str:
    with open(路径, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def 跑子(脚本: str, 额外环: dict = None, 超时: int = 300):
    """跑一段子进程脚本，返回 (returncode, 合并输出)。"""
    环 = dict(os.environ)
    环["PYTHONIOENCODING"] = "utf-8"
    环.update(额外环 or {})
    try:
        r = subprocess.run([sys.executable, "-c", 脚本], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=超时, env=环)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return None, "【超时>%ds】" % 超时


def 末行(出: str) -> str:
    行s = [l.strip() for l in (出 or "").splitlines() if l.strip()]
    return 行s[-1] if 行s else ""


# ─────────────────────────── P0-1 C13 ───────────────────────────
_子码 = r'''
import hashlib, sys
import numpy as np
sys.path.insert(0, r"%s")
from 通用接口 import 统一输入, 修复C13
from lang_codec import 语言编解码器
码 = 统一输入(语言编解码器(seed=310), seed=310)
P = np.ascontiguousarray(码._投影阵("音频特征", 64), dtype=np.float32)
print(hashlib.sha256(P.tobytes()).hexdigest())
'''


def 判C13() -> dict:
    脚 = _子码 % M38S
    修复 = {}
    for seed in ("0", "12345"):
        for i in (1, 2):                      # 同设置跑两遍：证「同设置逐位同」
            rc, 出 = 跑子(脚, 额外环={"PYTHONHASHSEED": seed})
            修复["修复_H%s_第%d遍" % (seed, i)] = dict(rc=rc, 特征码=末行(出),
                                                    错误=None if rc == 0 else 出[-300:])
    旧 = {}
    for seed in ("0", "12345"):
        rc, 出 = 跑子(脚, 额外环={"PYTHONHASHSEED": seed, "AHI_C13_旧哈希": "1"})
        旧["旧路径_H%s" % seed] = dict(rc=rc, 特征码=末行(出),
                                    错误=None if rc == 0 else 出[-300:])
    修码 = sorted({v["特征码"] for v in 修复.values()})
    旧码 = sorted({v["特征码"] for v in 旧.values()})
    修全同 = (len(修码) == 1 and all(v["rc"] == 0 for v in 修复.values()))
    旧两异 = (len(旧码) == 2)                  # ★对照臂：旧路径在两个 hash seed 下必须不同
    return dict(通过=bool(修全同 and 旧两异),
                修复路径跨seed全同=修全同, 修复特征码=修码,
                旧路径跨seed相异=旧两异, 旧特征码=旧码,
                对照臂说明="sham = AHI_C13_旧哈希=1（内建 hash 旧行为）；它必须相异，否则判据恒真",
                明细=dict(修复=修复, 旧路径=旧))


# ─────────────────────────── P0-2 时基 ───────────────────────────
def 判时基() -> dict:
    自 = 时基.自测()
    # ③ 禁墙钟：把 time.time 打坏，拍钟仍必须能打点（证明判决量不依赖墙钟）
    import time as _t
    原 = _t.time
    def _坏():
        raise RuntimeError("墙钟被本判据故意打坏（禁 wall-clock 作判据）")
    坏钟 = None
    try:
        _t.time = _坏
        钟 = 时基.拍钟(t0=5.0, 钟=时基.回放钟([5.0, 5.2]))
        钟.打点("τ_in"); 钟.打点("τ_q")
        坏钟 = 钟.延迟("τ_in", "τ_q")
    except Exception as ex:
        坏钟 = dict(异常=repr(ex))
    finally:
        _t.time = 原
    墙钟无关 = bool(isinstance(坏钟, dict) and 坏钟.get("拍") == 2.0)
    return dict(通过=bool(自["通过"] and 墙钟无关), 自测=自,
                墙钟打坏后打点=坏钟, 墙钟无关=墙钟无关,
                未接埋点=时基.拍钟(t0=0.0).未接埋点(),
                对照臂说明="对照臂=把 time.time 换成抛异常函数；若实现依赖墙钟 ⇒ 必崩")


# ─────────────────────────── P0-3 事件契约 ───────────────────────────
def 判事件契约() -> dict:
    自 = 事件契约.自测()
    好 = 事件契约.Provenance(prov_acq="用户", prov_org="自源", prov_route="实感")
    甲 = 事件契约.OutputEvent(τ=1.0, seq=1, 通道="文本", R版本=1, 断言级="事实",
                             内容sha256="b" * 64, 优先级="P2")
    乙 = 事件契约.OutputEvent(τ=1.0, seq=2, 通道="文本", R版本=1, 断言级="推测",
                             内容sha256="c" * 64, 优先级="P0")
    丙 = 事件契约.OutputEvent(τ=0.5, seq=3, 通道="文本", R版本=1, 断言级="回忆",
                             内容sha256="d" * 64, 优先级="P2")
    序对 = (事件契约.仲裁比较(乙, 甲) == -1               # P0 抢占 P2
          and 事件契约.仲裁比较(丙, 甲) == -1             # 同优先级比 τ（先到先出）
          and 事件契约.仲裁比较(甲, 甲) == 0)
    帧 = 事件契约.SensorFrame(模态="视觉", t_mono=12.5, seq=1, 形状=(4, 4), dtype="uint8",
                             数据=bytes(range(16)), 来源="合成")
    帧同 = (帧.特征码() == 事件契约.SensorFrame(模态="视觉", t_mono=12.5, seq=1, 形状=(4, 4),
                                               dtype="uint8", 数据=bytes(range(16)),
                                              来源="合成").特征码())
    栅 = 事件契约.写闸(好)
    return dict(通过=bool(自["通过"] and 序对 and 帧同 and 栅["可塑W_self"] is True),
                自测=自, 仲裁全序=序对,
                帧特征码=帧.特征码()[:16], 同帧同码=帧同,
                自源写闸=栅,
                对照臂说明="反例=附加{'得分'}、动作通道缺令牌、不可逆缺令牌、枚举越界 —— 四条都必须被拒")


# ─────────────────────────── P0-4 R 契约 ───────────────────────────
def 判回忆契约() -> dict:
    sys.path.insert(0, M38S)
    自 = 回忆契约.自测()
    真 = {}
    try:
        import numpy as np
        from 容量桥 import 容量桥
        桥 = 容量桥(种子=3800)
        v = np.ones(1024, dtype=np.float32)
        桥.记经历(拍=1, 类型="事件", 文本="苏州的天气", 来源="用户", 向量=v)
        r关 = 回忆契约.从容量桥(桥, v, 版本=1, 相似下限=0.0, 开关=False)
        r开 = 回忆契约.从容量桥(桥, v, 版本=1, 相似下限=0.0, 开关=True)
        只读前后 = (len(桥._文本), len(桥._向量))
        真 = dict(默认关返回=(None if r关 is None else r关.字典()),
                  开关关时为None=bool(r关 is None),
                  开关开可发布=bool(r开 is not None and r开.可发布()),
                  候选=(r开.字典()["候选"][:1] if r开 else None),
                  库状态=dict(文本条=只读前后[0], 向量条=只读前后[1]))
    except Exception as ex:
        真 = dict(异常=repr(ex))
    适配对 = bool(真.get("开关关时为None") and 真.get("开关开可发布"))
    return dict(通过=bool(自["通过"] and 适配对), 自测=自, 真容量桥适配=真,
                对照臂说明="对照臂=开关 False（默认关）必须返回 None 且不改库；空R/冲突R 必须不可发布")


# ─────────────────────────── P0-5 provenance 与真内核对齐 ───────────────────────────
def 判写闸() -> dict:
    走查 = {}
    情况 = [("自源_实感_用户", dict(prov_acq="用户", prov_org="自源", prov_route="实感"), False),
            ("外源_实感_网页", dict(prov_acq="网页", prov_org="外源", prov_route="实感"), False),
            ("自源_自说_输出", dict(prov_acq="自源", prov_org="自源", prov_route="自说"), False),
            ("外源_回忆", dict(prov_acq="未构", prov_org="外源", prov_route="回忆"), False),
            ("自源_想象", dict(prov_acq="自源", prov_org="自源", prov_route="想象"), False)]
    for 名, d, _ in 情况:
        d = dict(d)
        d["prov_acq"] = {"未构": "未知"}.get(d["prov_acq"], d["prov_acq"])
        p = 事件契约.Provenance(**d)
        走查[名] = 事件契约.写闸(p, 输出回灌=(d["prov_route"] == "自说"))
    期望 = {"自源_实感_用户": dict(M=True, W=True),
            "外源_实感_网页": dict(M=True, W=False),
            "自源_自说_输出": dict(M=False, W=True),
            "外源_回忆": dict(M=False, W=False),
            "自源_想象": dict(M=False, W=True)}
    走查对 = all(走查[k]["可写M"] == v["M"] and 走查[k]["可塑W_self"] == v["W"]
               for k, v in 期望.items())
    # 与真内核枚举逐字对齐
    对齐 = {}
    try:
        import 内核_自我回路 as 回路
        对齐 = dict(内核_自源=getattr(回路, "来源标_自源", None),
                    内核_外源=getattr(回路, "来源标_外源", None),
                    契约_内容源=list(事件契约.内容源枚举))
        对齐对 = (对齐["内核_自源"] in 事件契约.内容源枚举
                and 对齐["内核_外源"] in 事件契约.内容源枚举)
    except Exception as ex:
        对齐, 对齐对 = dict(异常=repr(ex)), False
    return dict(通过=bool(走查对 and 对齐对), 走查=走查, 期望=期望, 走查对=走查对,
                内核对齐=对齐, 内核对齐对=对齐对,
                对照臂说明="反例=自说/输出回灌必须 M=False；外源必须 W=False（J5：外源不可塑造我）")


# ─────────────────────────── P0-6 测量台 ───────────────────────────
def 判测量台() -> dict:
    自 = 测量台.本台自证()
    return dict(通过=bool(自["通过"]), 自证=自,
                对照臂说明="台自证内置 3 条必失败臂（恒不过/主零同过/带不确定源）")


# ─────────────────────────── P0-7 C14 ───────────────────────────
def 判C14() -> dict:
    import importlib
    sys.path.insert(0, M39S)
    步7 = importlib.import_module("步7_全量回归自证")
    md5 = 步7.判MD5()
    冻 = {}
    for 相对, 期 in 冻结哈希.items():
        路径 = os.path.join(ROOT, 相对.replace("/", os.sep))
        实 = sha256文件(路径)
        sz = os.path.getsize(路径)
        冻[相对] = dict(实=实, 期望=期["sha256"][:16] + "…", 同=(实 == 期["sha256"]),
                        size=sz, size期望=期["size"], size同=(sz == 期["size"]))
    # 改动面：与备份 diff（纯追加？删除行数必须=1：只换 hash(key) 那一行）
    with open(通用接口备份, encoding="utf-8") as f:
        旧源 = f.read().splitlines()
    with open(通用接口路径, encoding="utf-8") as f:
        新源 = f.read().splitlines()
    差 = list(difflib.unified_diff(旧源, 新源, lineterm="", n=0))
    删 = [l for l in 差 if l.startswith("-") and not l.startswith("---")]
    增 = [l for l in 差 if l.startswith("+") and not l.startswith("+++")]
    面 = dict(删除行数=len(删), 新增行数=len(增), 删除内容=删,
              仅一处替换=bool(len(删) == 1 and "hash(key)" in 删[0]),
              备份sha256=sha256文件(通用接口备份))
    return dict(通过=bool(md5["通过"] and all(v["同"] and v["size同"] for v in 冻.values())
                        and 面["仅一处替换"]),
                帧冻结10文件=dict(通过=md5["通过"], 条数=md5["条数"]),
                终稿13_4两文件=冻, 通用接口改动面=面,
                对照臂说明="对照臂=帧冻结 10 文件任一 MD5 变动 ⇒ 立即 FAIL；本判据在此逐位比对")


# ─────────────────────────── P0-8 A1–A7 ───────────────────────────
def _去噪(d):
    if isinstance(d, dict):
        return {k: _去噪(v) for k, v in d.items() if k not in 排除键}
    if isinstance(d, list):
        return [_去噪(x) for x in d]
    return d


def 判A1A7() -> dict:
    基线路径 = os.path.join(基线目录, "基线_自检结果.json")
    if not os.path.exists(基线路径):
        return dict(通过=False, 错="缺基线 %s" % 基线路径)
    with open(基线路径, encoding="utf-8") as f:
        基线 = json.load(f)
    rc, 出 = 跑子('import sys; sys.path.insert(0, r"%s"); import 自检; 自检.主()'
                 % M38, 额外环={"PYTHONIOENCODING": "utf-8"}, 超时=900)
    if rc != 0:
        return dict(通过=False, 错="自检.py 退出码=%s" % rc, 尾=出[-500:])
    新路径 = os.path.join(M38, "自检结果.json")
    with open(新路径, encoding="utf-8") as f:
        新 = json.load(f)
    逐 = {}
    for 名 in 基线["项"]:
        a, b = _去噪(基线["项"][名]), _去噪(新["项"].get(名, {}))
        异 = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        逐[名] = dict(通过同=(a.get("通过") == b.get("通过")), 异项=异)
    全同 = all(v["通过同"] and not v["异项"] for v in 逐.values())
    return dict(通过=bool(全同 and 基线["总过"] and 新["总过"]),
                基线总过=基线["总过"], 新总过=新["总过"],
                排除键=list(排除键),
                排除理由="墙钟/并发计数类字段天然逐次变动（如 A1 读写次数），非机制差异——如实登记而非掩盖",
                逐项=逐,
                对照臂说明="对照臂=若任一 A 项由过转不过或结构字段变动 ⇒ FAIL")


# ─────────────────────────── 主 ───────────────────────────
def 主() -> dict:
    项 = [("P0-1_C13跨PYTHONHASHSEED", 判C13),
          ("P0-2_时基", 判时基),
          ("P0-3_事件契约", 判事件契约),
          ("P0-4_R契约", 判回忆契约),
          ("P0-5_provenance写闸", 判写闸),
          ("P0-6_测量台", 判测量台),
          ("P0-7_C14冻结", 判C14),
          ("P0-8_旧判据A1A7", 判A1A7)]
    报 = {}
    for 名, fn in 项:
        try:
            报[名] = fn()
        except Exception as ex:
            报[名] = dict(通过=False, 异常=repr(ex))
        print("[%s] %s" % ("过" if 报[名].get("通过") else "未过", 名), flush=True)
    总 = all(v.get("通过") for v in 报.values())
    报["总判"] = "全过" if 总 else "有未过项"
    报["判据数"] = len(项)
    报["口径"] = dict(拍长="1 拍 = 100ms（τ=floor((t_mono−t0)/0.1)）",
                     双集="判据集 seed 0–5 / 留出集 seed 6–11（测量台口径）",
                     通用接口改动="C13 修复：内建 hash → sha256 派生（回退开关 AHI_C13_旧哈希=1）")
    报["本判据特征码"] = 测量台.特征码({k: v for k, v in 报.items() if k != "本判据特征码"})
    with open(os.path.join(HERE, "P0_结果.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1, default=str)
    print("\n==== P0 总判：%s ====" % 报["总判"])
    print("[已存] %s" % os.path.join(HERE, "P0_结果.json"))
    return 报


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    主()
