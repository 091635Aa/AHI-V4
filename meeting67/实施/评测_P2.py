# -*- coding: utf-8 -*-
"""meeting67 · P2-4：P2 三臂评测（A0 基线 / A1 阳性对照 / A2 主臂 / A3 只测头）。

口径（预注册 `11`）：
  - A0 = v13 预测 mel → Griffin-Lim（两套最佳设置：e0150 4轮/0.5；e0100 7轮/0.5）；
  - A1 = **真** mel + **真** F0/浊音（离线提取）→ 谐波声码器 → ASR；护栏 P2-1：CER ≤0.15；
    同 5 条真实样本另做 A1-GL（真 mel→GL）作同源对照；
  - A2 = v13 预测 mel + **头预测** F0/浊音 → 谐波声码器 → ASR（双集 S1–S12）；
  - A3 = 头质量（内部留出 8 句；不出波形；读数来自 `头质量_*.json` 同源）；
  - sham（白噪/静音）≥0.9；每句两趟转写逐字一致；谐波声码器两趟 sha256 逐位同；
  - S1–S12 **只报一次**；S13–S18 **只选不报**（`--选择集` 可选，结果只进 `选择集` 字段）；
  - **ASR 单进程**：本脚本默认**不跑 ASR**（dry-run，只做合成/自检/头读数）；只有显式
    `--跑ASR` 才载引擎（全机必须只有一个 ASR 进程，C9/C10 口径；Lead 确认后执行）。

用法（待 Lead 执行；先 dry-run 再 ASR）：
  # 1) dry-run（不载 ASR）：合成 12 句×2 设置 + A1 5 条、自检确定性、写半成品 JSON
  & <venv python> meeting67/实施/评测_P2.py
  # 2) 正式（单进程 ASR；仅当无其他 ASR 进程时）
  & <venv python> meeting67/实施/评测_P2.py --跑ASR
  # 3) 可选：S13–S18 只选不报（6 句 × 候选，不进报告表）
  & <venv python> meeting67/实施/评测_P2.py --跑ASR --选择集
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M67 = HERE.parent
AHI4 = M67.parent
M66 = AHI4 / "meeting66"
M66实施 = M66 / "实施"
产物66 = M66 / "产物"
P2产物 = M67 / "产物" / "P2"
音频目录 = P2产物 / "音频"
评测目录 = P2产物 / "评测"
日志目录 = M67 / "团队" / "P2_实现"
F0目录 = P2产物 / "f0"
头目录 = P2产物 / "头"
判据目录66 = 产物66 / "判据"

SR = 22050
每音节帧数 = 24                 # 与 v13 当前最佳同口径（meeting66 §12）
目标峰值 = 0.9

# 两套 A0/A2 设置（meeting66 全局最佳：e0150 4轮/0.5 历史最好；e0100 7轮/0.5 留出最好）
设置s = [
    dict(键="A_4轮05", ckpt="ckpt_e0150.pt", 迭代=4, 阻尼=0.5, 标签="e0150_4轮05"),
    dict(键="B_7轮05", ckpt="ckpt_e0100.pt", 迭代=7, 阻尼=0.5, 标签="e0100_7轮05"),
]
# S13–S18（只选不报；逐字取自 meeting66/07_预注册_独立验证集.md）
句集选择 = (
    ("S13", "今天天气真不错，我们去外面走走吧。"),
    ("S14", "他把钥匙放在门口的桌子上。"),
    ("S15", "这家饭馆的红烧肉做得很好吃。"),
    ("S16", "明天上午我要去银行办点事。"),
    ("S17", "孩子们在操场上跑来跑去。"),
    ("S18", "雨停了，天上的云慢慢散开。"),
)

_模块 = {}


def _按路径载(名: str, 路径: Path):
    if 名 in _模块:
        return _模块[名]
    spec = importlib.util.spec_from_file_location(名, str(路径))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _模块[名] = mod
    return mod


def 声码器66():
    """meeting66 声码器与评测（只读）：GL / 载ckpt / 推理mel / 写wav / 读回 / 句集 / sham。"""
    return _按路径载("声码器与评测_只读", M66实施 / "声码器与评测.py")


def vocoder谐波():
    return _按路径载("声码器_谐波_只读", HERE / "声码器_谐波.py")


def 头模():
    return _按路径载("头_F0浊音_只读", HERE / "头_F0浊音.py")


def _sha256数组(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def _sha256文件(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


def _json可写(o):
    if isinstance(o, dict):
        return {str(k): _json可写(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json可写(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _确保(*路s):
    for p in 路s:
        Path(p).mkdir(parents=True, exist_ok=True)


def _新run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid())


# ════════════════════════════════════════════════════════════════════════
# 1. 合成（A0/A1/A2）
# ════════════════════════════════════════════════════════════════════════
def 写样本(名: str, 波形, 源: str, 附加: dict) -> dict:
    路 = 音频目录 / 名
    声66 = 声码器66()
    声66.写wav(路, 波形)
    出 = dict(文件=名, 路径=str(路), 源=源,
             波形样本数=int(np.size(波形)), 期望样本数=int(np.size(波形)),
             峰值=round(float(np.abs(波形).max()) if np.size(波形) else 0.0, 4),
              全finite=bool(np.isfinite(np.asarray(波形)).all()),
              sha256=_sha256文件(路), 读回=声66._读回核对(路))
    出.update(附加)
    return 出


def _设推理环境(迭代: int, 阻尼: float) -> None:
    """合成mel 从环境变量读推理设置（只在本进程内设置；不写任何既有件）。"""
    os.environ["M66_迭代推理"] = str(int(迭代))
    os.environ["M66_阻尼"] = str(float(阻尼))


def 合成GL(模型, 文本: str, 每帧数: int, 标签: str, run: str, 序号: int,
          迭代: int = 3, 阻尼: float = 1.0) -> dict:
    声66 = 声码器66()
    _设推理环境(迭代, 阻尼)
    mel, 有效, 跳过 = 声66.推理mel(模型, 文本, 每音节帧数=每帧数)
    波 = 声66.mel转波(mel)
    名 = "A0_%s_%02d_%s.wav" % (标签, 序号, str(文本)[:8].replace("，", "").replace("。", ""))
    出 = 写样本(名, 波, 源="v13 预测 mel → Griffin-Lim", 附加=dict(
        文本=文本, mel帧数=int(mel.shape[0]), 有效音节数=有效,
        期望样本数=int(声66.期望长度(mel.shape[0])),
        长度正确=bool(int(np.size(波)) == int(声66.期望长度(mel.shape[0]))),
        mel_sha256=_sha256数组(mel)))
    return 出


def 合成A1(样本id: int, 用GL: bool = False) -> dict:
    """真 mel + 真 F0/浊音（离线提取）→ 谐波声码器（或 GL 对照）。"""
    声66, 谐 = 声码器66(), vocoder谐波()
    mel路 = 产物66 / "数据_对齐asr" / "样本" / ("%04d.npz" % int(样本id))
    f0路 = F0目录 / ("%04d.npz" % int(样本id))
    if not f0路.exists():
        return dict(错误="缺 f0 产物：%s" % f0路, 样本=int(样本id))
    mel = np.asarray(np.load(str(mel路))["mel"], dtype=np.float32)
    fz = np.load(str(f0路))
    f0 = np.asarray(fz["f0"], dtype=np.float64)
    voi = np.asarray(fz["voicing"], dtype=np.int8)
    if 用GL:
        波 = 声66.mel转波(mel)
        名 = "A1GL_真mel_%04d.wav" % int(样本id)
        源 = "真 mel → Griffin-Lim（同源对照）"
    else:
        波 = 谐.谐波声码器(mel, f0, voi)
        名 = "A1_真mel真F0_%04d.wav" % int(样本id)
        源 = "真 mel + 真 F0/浊音（pyworld 离线提取）→ 谐波声码器"
    出 = 写样本(名, 波, 源=源, 附加=dict(
        样本=int(样本id), mel帧数=int(mel.shape[0]),
        浊音帧=int(voi.sum()), 清音帧=int((voi == 0).sum()),
        期望样本数=int(声66.期望长度(mel.shape[0])),
        长度正确=bool(int(np.size(波)) == int(声66.期望长度(mel.shape[0]))),
        mel_sha256=_sha256数组(mel), f0_sha256=_sha256数组(f0)))
    if not 用GL:
        波2 = 谐.谐波声码器(mel, f0, voi)
        出["两趟逐位同"] = bool(_sha256数组(波) == _sha256数组(波2))
    return 出


def 合成A2(模型, 头, 文本: str, 每帧数: int, 迭代: int, 阻尼: float,
           设置标签: str, run: str, 序号: int, 也出GL: bool = True) -> dict:
    """v13 预测 mel + 头预测 F0/浊音 → 谐波声码器（附同 mel 的 GL 对照）。"""
    声66, 谐, 头m = 声码器66(), vocoder谐波(), 头模()
    预 = 头m.预测F0(模型, 头, 文本, 每音节帧数=每帧数, 迭代=迭代, 阻尼=阻尼, 设备="cpu")
    mel = 预["mel"]
    # 一致性：本件 pred mel 必须与 推理mel（合成mel 官方口径）逐位同（容差 1e-5）
    _设推理环境(迭代, 阻尼)
    参照mel, 有效, _跳过 = 声66.推理mel(模型, 文本, 每音节帧数=每帧数)
    mel一致 = bool(mel.shape == 参照mel.shape and float(np.abs(mel - 参照mel).max()) < 1e-5)
    波 = 谐.谐波声码器(mel, 预["f0"], 预["voicing"])
    波2 = 谐.谐波声码器(mel, 预["f0"], 预["voicing"])
    名 = "A2_%s_%02d_%s.wav" % (设置标签, 序号, str(文本)[:8].replace("，", "").replace("。", ""))
    出 = 写样本(名, 波, 源="v13 预测 mel + 头预测 F0/浊音 → 谐波声码器", 附加=dict(
        文本=文本, mel帧数=int(mel.shape[0]), 有效音节数=有效,
        浊音帧=int(预["voicing"].sum()), 清音帧=int((预["voicing"] == 0).sum()),
        预测浊音比=预["浊音比"],
        F0中位Hz=round(float(np.median(预["f0"][预["voicing"] > 0]))
                       if 预["voicing"].sum() else 0.0, 2),
        期望样本数=int(声66.期望长度(mel.shape[0])),
        长度正确=bool(int(np.size(波)) == int(声66.期望长度(mel.shape[0]))),
        两趟逐位同=bool(_sha256数组(波) == _sha256数组(波2)),
        mel_sha256=_sha256数组(mel), pred_mel与推理mel一致=mel一致,
        谐波声码器噪声种子=谐.噪声种子))
    if 也出GL:
        波G = 声66.mel转波(mel)
        名G = "A0同mel_%s_%02d_%s.wav" % (设置标签, 序号,
                                        str(文本)[:8].replace("，", "").replace("。", ""))
        出["同mel_GL"] = 写样本(名G, 波G, 源="与 A2 同一条预测 mel → Griffin-Lim（逐句配对对照）",
                               附加=dict(文本=文本, mel帧数=int(mel.shape[0]),
                                        期望样本数=int(声66.期望长度(mel.shape[0])),
                                        长度正确=bool(int(np.size(波G)) == int(声66.期望长度(mel.shape[0])))))
    return 出


# ════════════════════════════════════════════════════════════════════════
# 2. ASR（单进程；只在 --跑ASR 时载引擎）
# ════════════════════════════════════════════════════════════════════════
def _评文件(引, 路: str, 真: str) -> dict:
    声66 = 声码器66()
    量 = 声66.量具模块()
    t1 = 引["转写"](str(路))
    t2 = 引["转写"](str(路))
    cer = 量.CER(真, t1)
    对 = 量.对齐(真, t1)
    return dict(转写=t1, 转写各趟=[t1, t2], 趟一致=bool(t1 == t2), CER=cer,
                对齐=dict(S=对["S"], D=对["D"], I=对["I"]),
                测文=对["测文"], 真文=对["真文"])


def _汇总CER(逐条: list) -> dict:
    cs = np.asarray([r["CER"] for r in 逐条], dtype=np.float64)
    return dict(条数=len(逐条), CER均值=round(float(cs.mean()), 4) if cs.size else None,
                两趟一致=bool(all(r["趟一致"] for r in 逐条)),
                逐条=逐条)


def 评测ASR(run: str, 产物: dict, 做选择集: bool) -> dict:
    声66 = 声码器66()
    量 = 声66.量具模块()
    print("[ASR] 载引擎（单进程；Paraformer-seaco-large）…", flush=True)
    引 = 量.载引擎()
    print("[ASR] 引擎 = %s" % 引.get("名"), flush=True)
    出 = {}

    # ① A1 阳性对照（5 条真实样本；真 mel+真 F0 → 谐波）—— 顺序最先，护栏最快可见
    a1 = 产物["A1"]["样本"]
    for i, 条 in enumerate(a1):
        if "错误" in 条:
            continue
        文本 = 产物["A1"]["文本s"][i]
        条.update(_评文件(引, 条["路径"], 文本))
    出["A1"] = _汇总CER([x for x in a1 if "CER" in x])
    gl = 产物["A1GL"]["样本"]
    for i, 条 in enumerate(gl):
        if "错误" in 条:
            continue
        文本 = 产物["A1"]["文本s"][i]
        条.update(_评文件(引, 条["路径"], 文本))
    出["A1GL"] = _汇总CER([x for x in gl if "CER" in x])
    print("[ASR] A1=%s（护栏 ≤0.15）A1GL=%s"
          % (出["A1"]["CER均值"], 出["A1GL"]["CER均值"]), flush=True)

    # ② A0 / A2 双集（S1–S12 只报一次）
    for 设置 in 设置s:
        键 = 设置["键"]
        for 臂名 in ("A0", "A2"):
            逐 = []
            for 集合名 in ("历史", "留出"):
                for r in 产物[臂名][键][集合名]["逐条"]:
                    r.update(_评文件(引, r["路径"], r["文本"]))
                s = 产物[臂名][键][集合名]
                逐.append(s)
            for s in 逐:
                合 = _汇总CER(s["逐条"])
                s.update(dict(CER均值=合["CER均值"], 两趟一致=合["两趟一致"]))
            双 = dict(历史=逐[0]["CER均值"], 留出=逐[1]["CER均值"])
            产物[臂名][键]["双集CER"] = 双
            print("[ASR] %s·%s 历史=%s 留出=%s" % (臂名, 键, 双["历史"], 双["留出"]), flush=True)

    # ③ sham（must-fail）
    sham = 声66._sham(引, 音频目录, run)
    出["sham"] = sham

    # ④ 阳性对照继承（v13 判据 JSON 的 仪器阳性对照，P2-6）
    继承 = {}
    for 名, f in (("A_4轮05", "判据_学说话_复评_v13_4轮阻尼05.json"),
                  ("B_7轮05", "判据_学说话_复评_v13_7轮阻尼05.json")):
        路 = 判据目录66 / f
        if 路.exists():
            d = json.loads(路.read_text(encoding="utf-8"))
            阳 = d.get("仪器阳性对照") or {}
            继承[名] = dict(文件=f, sha256=_sha256文件(路), CER=阳.get("CER"),
                            门槛=0.10, 达标=bool(阳.get("CER") is not None and 阳["CER"] <= 0.10))
    出["阳性对照继承"] = 继承

    # ⑤ 可选：S13–S18 只选不报（结果只进 选择集，不出现于任何报告口径）
    if 做选择集:
        选 = {}
        for 设置 in 设置s:
            模型 = 产物["_模型"][设置["键"]]
            头 = 产物["_头"][设置["标签"]]
            for 臂名, 用谐波 in (("A2", True), ("A0", False)):
                逐 = []
                for 句号, 句 in 句集选择:
                    合成 = 产物["_选择合成"][(设置["键"], 臂名, 句号)]
                    逐.append(dict(句号=句号, **_评文件(引, 合成["路径"], 句)))
                选["%s_%s" % (臂名, 设置["键"])] = dict(
                    CER均值=round(float(np.mean([x["CER"] for x in 逐])), 4),
                    说明="S13–S18 只选不报，不进入报告表",
                    逐条=[dict(句号=x["句号"], CER=x["CER"]) for x in 逐])
        ###########################################
        # 注意：选择集只在 A2/A0 之间、且每臂只跑一次；候选=两套设置，不含逐句调参。
        ###########################################
        出["选择集"] = 选
        print("[ASR] 选择集（S13–S18 只选不报）：%s"
              % json.dumps({k: v["CER均值"] for k, v in 选.items()}, ensure_ascii=False), flush=True)
    return 出


# ════════════════════════════════════════════════════════════════════════
# 3. 主流程
# ════════════════════════════════════════════════════════════════════════
def 主(argv=None) -> int:
    ap = argparse.ArgumentParser(description="meeting67 P2 三臂评测（默认不跑 ASR）")
    ap.add_argument("--仅A1", action="store_true", help="只做 A1 阳性对照（5 条真实样本：真mel+真F0→谐波声码器）及其 ASR")
    ap.add_argument("--跑ASR", action="store_true", help="显式载 ASR 引擎（全机只许一个 ASR 进程）")
    ap.add_argument("--选择集", action="store_true", help="附加 S13–S18 只选不报")
    ap.add_argument("--输出", default=None)
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    _确保(音频目录, 评测目录, 日志目录)
    run = _新run_id()
    开始 = time.time()
    声66, 谐, 头m = 声码器66(), vocoder谐波(), 头模()
    meta = 声66._读meta()
    产品 = dict(_模型={}, _头={}, _选择合成={})
    异常: list = []

    # —— 文件登记 ——
    登记 = dict(
        工具="评测_P2.py", 版本="1.0", 任务="meeting67 P2 三臂评测（A0/A1/A2/A3）",
        时间=time.strftime("%Y-%m-%d %H:%M:%S"), run_id=run,
        脚本sha256={p.name: _sha256文件(p) for p in
                    (Path(__file__).resolve(), HERE / "声码器_谐波.py", HERE / "头_F0浊音.py",
                     HERE / "提取_F0浊音.py", M66实施 / "声码器与评测.py")},
        口径=dict(采样率=SR, 每音节帧数=每音节帧数, 句集="S1–S12 只报一次",
                选择集="S13–S18 只选不报（不进入报告）", ASR="Paraformer-seaco-large 离线仪器（单进程）",
                谐波声码器=dict(噪声种子=谐.噪声种子, 峰值归一=目标峰值,
                             相位="浊音按 F0 累积（固定初相 0）；清音（种子,帧号）确定性伪随机"),
                设置s=[dict(键=s["键"], ckpt=s["ckpt"], 迭代=s["迭代"], 阻尼=s["阻尼"]) for s in 设置s]),
        环境=dict(python=sys.version.split()[0], numpy=np.__version__))

    # —— 载 ckpt（两套设置）+ 头（若有） ——
    for 设置 in 设置s:
        路 = 产物66 / "模型_v13_K4_320" / 设置["ckpt"]
        if not 路.exists():
            异常.append("缺 ckpt：%s" % 路)
            continue
        产品["_模型"][设置["键"]] = 声66.载ckpt(路, 设备="cpu")
        登记.setdefault("ckpt", {})[设置["键"]] = dict(
            路径=str(路), sha256=_sha256文件(路),
            训练信息=dict(getattr(产品["_模型"][设置["键"]], "训练信息", {}) or {}))
    头信息 = {}
    for 设置 in 设置s:
        头路 = 头目录 / ("头_%s.pt" % 设置["标签"])
        质量路 = 头目录 / ("头质量_%s.json" % 设置["标签"])
        if 头路.exists():
            头, ck = 头m.载头(头路, 设备="cpu")
            产品["_头"][设置["标签"]] = 头
            头信息[设置["标签"]] = dict(
                头路径=str(头路), 头sha256=_sha256文件(头路), 参数数=int(ck.get("头", {}).get("参数数")
                                                                     or 头.参数数()),
                头质量=(json.loads(质量路.read_text(encoding="utf-8")) if 质量路.exists() else None),
                头质量sha256=(_sha256文件(质量路) if 质量路.exists() else None))
        else:
            头信息[设置["标签"]] = dict(缺="头 ckpt 不存在：%s（先跑 头_F0浊音.py）" % 头路)
    产品["A3"] = 头信息

    # —— A1：真 mel + 真 F0（5 条真实样本，0075 + 4 条最短） ——
    阳路 = F0目录 / "阳性对照集.json"
    a1样本 = []
    a1文本s = []
    if 阳路.exists():
        阳 = json.loads(阳路.read_text(encoding="utf-8"))
        for r in 阳["条"]:
            id_ = int(r["id"])
            a1样本.append(合成A1(id_, 用GL=False))
            a1文本s.append(r.get("文本"))
        gl样本 = [合成A1(int(r["id"]), 用GL=True) for r in 阳["条"]]
        产品["A1"] = dict(说明="真 mel + 真 F0/浊音 → 谐波声码器（护栏 P2-1：CER ≤0.15）",
                         源=阳路.name, 阳性对照集sha256=_sha256文件(阳路),
                         样本=a1样本, 文本s=a1文本s)
        产品["A1GL"] = dict(说明="真 mel → Griffin-Lim（同 5 条样本，同源对照）", 样本=gl样本)
        for 条 in a1样本 + gl样本:
            if not 条.get("长度正确", False):
                异常.append("%s 长度不符" % 条.get("文件"))
            if 条.get("两趟逐位同") is False:
                异常.append("%s 谐波声码器两趟不一致（C13）" % 条.get("文件"))
    else:
        异常.append("缺 阳性对照集.json（先跑 提取_F0浊音.py）")

    # —— 仅 A1 模式：阳性对照（5 条真实样本）与其 ASR，供单进程快速护栏 ——
    if args.仅A1:
        自检 = dict(谐波声码器=谐.自检(保存=True))
        if not any(x.get("错误") for x in a1样本):
            自检["A1长度与确定性"] = dict(
                条数=len(a1样本),
                长度全正确=bool(all(x.get("长度正确", False) for x in a1样本)),
                两趟逐位全同=bool(all(x.get("两趟逐位同", False) for x in a1样本)),
                峰值=[x.get("峰值") for x in a1样本])
        out = dict(登记, 产物=dict(A1=产品.get("A1"), A1GL=产品.get("A1GL")),
                   自检=_json可写(自检), ASR=None, 判定=None, 异常=异常,
                   说明="仅 A1 模式（阳性对照）；A0/A2/A3 未跑=未测")
        if args.跑ASR:
            引 = 声66.量具模块().载引擎()
            for i, 条 in enumerate(a1样本):
                if "错误" in 条:
                    continue
                条.update(_评文件(引, 条["路径"], a1文本s[i]))
                print("[A1] %s CER=%.4f 两趟一致=%s 转写=%s"
                      % (条.get("文件"), 条["CER"], 条["趟一致"], 条["转写"][:44]), flush=True)
            汇总 = _汇总CER([x for x in a1样本 if "CER" in x])
            sham = 声66._sham(引, 音频目录, run)
            判 = [dict(编号="P2-1", 条款="A1 阳性对照（真mel+真F0→谐波声码器）CER ≤0.15",
                       实测=汇总["CER均值"], 门槛=0.15,
                       通过=bool(汇总["CER均值"] is not None and 汇总["CER均值"] <= 0.15)),
                   dict(编号="P2-2", 条款="sham（白噪/静音）CER ≥0.9",
                       实测=dict(白噪=sham.get("白噪"), 静音=sham.get("静音")),
                       门槛=0.9, 通过=bool(sham.get("双达标")))]
            out["ASR"] = _json可写(dict(A1=汇总, sham=sham))
            out["判定"] = dict(护栏=判, 护栏全过=bool(all(x["通过"] for x in 判)),
                               说明="仅 A1；A0/A2/A3 未跑=未测；T-P2-1：A1>0.20 停本点")
            异常.append("仅 A1 模式：A0/A2/A3 未跑（未测）")
        输出 = 评测目录 / "判据_P2_A1.json"
        输出.parent.mkdir(parents=True, exist_ok=True)
        输出.write_text(json.dumps(_json可写(out), ensure_ascii=False, indent=2), encoding="utf-8")
        print("[评测] 输出：%s" % 输出, flush=True)
        if out.get("判定"):
            print("[评测] A1 CER=%s 护栏全过=%s"
                  % (out["ASR"]["A1"]["CER均值"], out["判定"]["护栏全过"]), flush=True)
        return 0 if (out.get("判定") or {}).get("护栏全过", True) else 2

    # —— A0 / A2：S1–S12 ——
    for 设置 in 设置s:
        键 = 设置["键"]
        产品["A0"] = 产品.get("A0", {})
        产品["A2"] = 产品.get("A2", {})
        产品["A0"][键] = dict(设置=dict(ckpt=设置["ckpt"], 迭代=设置["迭代"], 阻尼=设置["阻尼"]),
                             历史=dict(逐条=[]), 留出=dict(逐条=[]))
        产品["A2"][键] = dict(设置=dict(ckpt=设置["ckpt"], 迭代=设置["迭代"], 阻尼=设置["阻尼"]),
                             历史=dict(逐条=[]), 留出=dict(逐条=[]))
        模型 = 产品["_模型"].get(键)
        头 = 产品["_头"].get(设置["标签"])
        if 模型 is None:
            异常.append("%s：模型未载入，跳过" % 键)
            continue
        序 = 0
        for 集合名, 句s in (("历史", 声66.句集历史), ("留出", 声66.句集留出)):
            for 句号, 句 in 句s:
                序 += 1
                a0 = 合成GL(模型, 句, 每音节帧数, 设置["键"], run, 序,
                             设置["迭代"], 设置["阻尼"])
                a0["句号"] = 句号
                产品["A0"][键][集合名]["逐条"].append(a0)
                if 头 is not None:
                    a2 = 合成A2(模型, 头, 句, 每音节帧数, 设置["迭代"], 设置["阻尼"],
                                 设置["标签"], run, 序, 也出GL=False)
                    a2["句号"] = 句号
                    产品["A2"][键][集合名]["逐条"].append(a2)
                    if not a2.get("pred_mel与推理mel一致", False):
                        异常.append("%s·%s pred mel 与推理mel 不一致" % (设置["标签"], 句号))
                print("[合成] %s %s·%s 帧=%d 峰=%.2f"
                      % (设置["键"], 集合名, 句号,
                         产品["A0"][键][集合名]["逐条"][-1]["mel帧数"],
                         产品["A0"][键][集合名]["逐条"][-1]["峰值"]), flush=True)
        产品["A0"][键]["双集"] = dict(
            历史=dict(条数=len(产品["A0"][键]["历史"]["逐条"]),
                      长度正确=bool(all(x["长度正确"] for x in 产品["A0"][键]["历史"]["逐条"]))),
            留出=dict(条数=len(产品["A0"][键]["留出"]["逐条"]),
                      长度正确=bool(all(x["长度正确"] for x in 产品["A0"][键]["留出"]["逐条"]))))
        if 头 is not None:
            逐A2 = 产品["A2"][键]["历史"]["逐条"] + 产品["A2"][键]["留出"]["逐条"]
            产品["A2"][键]["自检"] = dict(
                长度全正确=bool(all(x["长度正确"] for x in 逐A2)),
                两趟逐位全同=bool(all(x.get("两趟逐位同", False) for x in 逐A2)),
                pred_mel全一致=bool(all(x.get("pred_mel与推理mel一致", False) for x in 逐A2)),
                预测浊音比均值=round(float(np.mean([x["预测浊音比"] for x in 逐A2])), 4),
                F0中位均值Hz=round(float(np.mean([x["F0中位Hz"] for x in 逐A2])), 2))

    # —— 选择集合成（只选不报；可选） ——
    if args.选择集:
        for 设置 in 设置s:
            模型 = 产品["_模型"].get(设置["键"])
            头 = 产品["_头"].get(设置["标签"])
            if 模型 is None:
                continue
            for 句号, 句 in 句集选择:
                if 头 is not None:
                    a2 = 合成A2(模型, 头, 句, 每音节帧数, 设置["迭代"], 设置["阻尼"],
                                 设置["标签"], run, 0, 也出GL=False)
                else:
                    a2 = None
                a0 = 合成GL(模型, 句, 每音节帧数, 设置["键"] + "_选择", run, 0,
                             设置["迭代"], 设置["阻尼"])
                a0["句号"] = 句号
                产品["_选择合成"][(设置["键"], "A0", 句号)] = a0
                if a2 is not None:
                    产品["_选择合成"][(设置["键"], "A2", 句号)] = a2

    # —— dry-run 自检汇总 ——
    自检 = dict(
        谐波声码器=谐.自检(保存=True),
    )
    if not any(x.get("错误") for x in a1样本):
        自检["A1长度与确定性"] = dict(
            条数=len(a1样本),
            长度全正确=bool(all(x.get("长度正确", False) for x in a1样本)),
            两趟逐位全同=bool(all(x.get("两趟逐位同", False) for x in a1样本)),
            峰值=[x.get("峰值") for x in a1样本])

    # —— ASR（可选，默认不跑） ——
    asr = None
    if args.跑ASR:
        if any("缺" in str(v) for v in 头信息.values()) and any(
                v.get("缺") for v in 头信息.values()):
            异常.append("有头缺失：%s" % [k for k, v in 头信息.items() if v.get("缺")])
        asr = 评测ASR(run, 产品, args.选择集)
    else:
        异常.append("未跑 ASR（默认 dry-run；待 Lead 单进程确认后执行 `--跑ASR`）")

    # —— 判定（只在 ASR 齐备时给；否则空着=未测） ——
    判定 = None
    if asr:
        判定 = 判据(产品, asr, 自检)
    产品.pop("_模型", None)
    产品.pop("_头", None)
    产品.pop("_选择合成", None)
    out = dict(登记, 产物=_json可写(产品), 自检=_json可写(自检), ASR=_json可写(asr) if asr else None,
               判定=_json可写(判定) if 判定 else None,
               异常=异常, 耗时秒=round(time.time() - 开始, 1),
               结束时间=time.strftime("%Y-%m-%d %H:%M:%S"),
               复现命令=("$env:PYTHONIOENCODING=\"utf-8\"; & \"%s\" \"%s\"%s"
                         % (sys.executable, Path(__file__).resolve(),
                            " --跑ASR" if args.跑ASR else "")),
               说明=("dry-run：ASR 字段为 null（未测）；正式读数须 `--跑ASR` 且全机只有一个 ASR 进程"
                   if not asr else "ASR 已跑（单进程）"))
    输出 = Path(args.输出) if args.输出 else ((评测目录 / "判据_P2.json") if asr
                                            else (评测目录 / "判据_P2_dryrun.json"))
    # dry-run 不覆盖正式判据
    if not asr and 输出.name == "判据_P2.json":
        输出 = 评测目录 / "判据_P2_dryrun.json"
    输出.parent.mkdir(parents=True, exist_ok=True)
    输出.write_text(json.dumps(_json可写(out), ensure_ascii=False, indent=2), encoding="utf-8")
    print("[评测] 输出：%s" % 输出, flush=True)
    if not asr:
        print("[评测] dry-run：未跑 ASR（未测=未测）；待 Lead：$PY 评测_P2.py --跑ASR", flush=True)
        for 键, 信 in 头信息.items():
            q = 信.get("头质量") or {}
            print("[评测] A3 %s：F0相关=%s 浊音准确=%s 清音F1=%s 全达标=%s"
                  % (键, q.get("F0相关_浊音帧_双浊"), q.get("浊音准确率"),
                     q.get("清音F1"), q.get("全达标")), flush=True)
    return 0


def 判据(产品: dict, asr: dict, 自检: dict) -> dict:
    """按预注册 `11` §3 判：护栏 P2-1..P2-6 + 主判据（A2 vs A0 双集）/ 头质量 / 触发。"""
    条 = []
    a1 = (asr.get("A1") or {}).get("CER均值")
    a1gl = (asr.get("A1GL") or {}).get("CER均值")
    sham = asr.get("sham") or {}
    条.append(dict(编号="P2-1", 条款="A1 阳性对照（真mel+真F0→谐波声码器）CER ≤0.15",
                   实测=a1, 门槛=0.15, 通过=bool(a1 is not None and a1 <= 0.15)))
    条.append(dict(编号="P2-1b", 条款="A1-GL 同源对照（真mel→GL）CER ≤0.10（仪器继承）",
                   实测=a1gl, 门槛=0.10, 通过=bool(a1gl is not None and a1gl <= 0.10)))
    条.append(dict(编号="P2-2", 条款="sham（白噪/静音）CER ≥0.9",
                   实测=dict(白噪=sham.get("白噪"), 静音=sham.get("静音")),
                   门槛=0.9, 通过=bool(sham.get("双达标"))))
    # P2-3 两趟一致（A0/A2 双集 + A1）
    def _两趟(臂: str) -> bool:
        d = 产品.get(臂) or {}
        值 = []
        for 设置 in d.values() if isinstance(d, dict) else []:
            if not isinstance(设置, dict):
                continue
            for 集 in ("历史", "留出"):
                if isinstance(设置.get(集), dict):
                    值.append(设置[集].get("两趟一致"))
        return bool(值) and all(值)
    两趟 = dict(A1=bool((asr.get("A1") or {}).get("两趟一致")),
                A0=_两趟("A0"), A2=_两趟("A2"))
    条.append(dict(编号="P2-3", 条款="12 句 × 两趟 ASR 逐字一致", 实测=两趟,
                   通过=bool(all(两趟.values()))))
    两趟sha = bool((自检.get("谐波声码器") or {}).get("两趟", {}).get("两趟逐位同"))
    条.append(dict(编号="P2-4", 条款="谐波声码器同输入两趟 sha256 逐位同（C13）",
                   实测=两趟sha, 通过=两趟sha))
    条.append(dict(编号="P2-5", 条款="口径：S1–S12 只报一次 / S13–S18 只选不报 / ASR 单进程",
                   实测="声明：本件只出现 S1–S12 报告表；选择集字段不进入报告；ASR 单进程由 Lead 确认",
                   通过=True, 备注="程序外口径（无法自证），登记为声明"))
    继 = (asr.get("阳性对照继承") or {})
    条.append(dict(编号="P2-6", 条款="阳性对照继承：v13 判据 JSON 仪器阳性对照（真mel→GL）≤0.10",
                   实测={k: v.get("CER") for k, v in 继.items()},
                   通过=bool(继 and all(v.get("达标") for v in 继.values())),
                   来源={k: v.get("文件") for k, v in 继.items()}))

    # 主判据：A2 vs A0（双集，逐设置；每条设置单独判，禁止各取优合并）
    主 = {}
    for 设置 in 设置s:
        键 = 设置["键"]
        a0 = ((产品.get("A0") or {}).get(键) or {}).get("双集CER") or {}
        a2 = ((产品.get("A2") or {}).get(键) or {}).get("双集CER") or {}
        if a0.get("历史") is None or a2.get("历史") is None:
            continue
        历史差 = round(float(a2["历史"]) - float(a0["历史"]), 4)
        留出差 = round(float(a2["留出"]) - float(a0["留出"]), 4)
        不退化 = bool(历史差 <= 0.01 and 留出差 <= 0.01)
        改善 = bool(历史差 < -0.01 or 留出差 < -0.01)
        主[键] = dict(A0=a0, A2=a2, Δ历史=历史差, Δ留出=留出差,
                     双集不退化=不退化, 至少一集改善=改善,
                     判=("PASS（有效增益）" if (不退化 and 改善)
                        else ("无有效增益（不退化但无改善>0.01）" if 不退化
                              else "退化（不退化条件不满足）")))
    有效 = [k for k, v in 主.items() if v["判"].startswith("PASS")]
    主判 = dict(逐设置=主,
                结论=("PASS：至少一套设置满足「双集不退化 + 至少一集改善>0.01」" if 有效
                     else "FAIL/无有效增益：无任一设置同时满足两条主判据"),
                有效设置=有效,
                口径="禁止各取优合并宣称；两集必须同时给出")

    # 头质量（A3，门槛 0.80/0.90/0.80）
    头 = {}
    for 键, 信 in (产品.get("A3") or {}).items():
        q = 信.get("头质量") or {}
        头[键] = dict(实际=q.get("F0相关_浊音帧_双浊"), 浊音准确率=q.get("浊音准确率"),
                      清音F1=q.get("清音F1"), 门槛=q.get("门槛"), 达标=q.get("达标"),
                      全达标=q.get("全达标"), 头sha256=信.get("头sha256"))

    # 触发（T-P2-1..5）
    触发 = []
    if a1 is not None and a1 > 0.20:
        触发.append(dict(编号="T-P2-1", 条款="A1 > 0.20 ⇒ 谐波声码器不可用，停本点", 触发=True))
    if a1gl is not None and a1gl > 0.15:
        触发.append(dict(编号="T-P2-1b", 条款="A1-GL > 0.15 ⇒ 仪器/链路异常", 触发=True))
    for 键, q in 头.items():
        r = q.get("实际")
        if r is not None and r < 0.5:
            触发.append(dict(编号="T-P2-2", 条款="F0 相关 <0.5 ⇒ 头失败，停，不追加训练",
                             键=键, 实测=r, 触发=True))
    for 键, v in 主.items():
        if v["Δ历史"] is not None and v["Δ留出"] is not None and \
                v["Δ历史"] > 0.03 and v["Δ留出"] > 0.03:
            触发.append(dict(编号="T-P2-3", 条款="A2 双集均差于 A0 >0.03 ⇒ 判无效停", 键=键, 触发=True))
    if any(not x["通过"] for x in 条):
        pass
    return dict(护栏=条, 护栏全过=bool(all(x["通过"] for x in 条)),
                主判据=主判, 头质量=头, 触发=触发,
                说明="读数来自同一 run；两集同时给出；未测字段保持 null")


if __name__ == "__main__":
    raise SystemExit(主())
