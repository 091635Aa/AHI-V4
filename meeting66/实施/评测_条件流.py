# -*- coding: utf-8 -*-
"""meeting66 / 评测_条件流.py —— v18 条件流（CFM）声学核心的双集 CER 评测。

定位：`模型_条件流.py`（条件 Flow Matching，非 Transformer / 无 attention）的
**只读评测驱动**。本件不训练、不改任何既有件（C14）；结构对齐 `评测_自回归.py`：
  句集 = `声码器与评测.py` 的 S1–S12（历史 S1–S6 / 留出 S7–S12，与 v8.1 基线同口径）；
  声码器 = 同 `V.mel转波`（Griffin-Lim 固定 64 迭代 / 固定种子 / 峰值归一）；
  写盘 = 同 `V.写wav`（22.05kHz / 16bit / 单声道）；
  ASR = 同 `V.量具模块().载引擎()`，每句**两趟转写**（C13），CER 取第一趟；
  协议 = `07_预注册_独立验证集.md`：S13–S18 只选不报，本报告只用 S1–S12。

用法：
  python 评测_条件流.py --ckpt <ckpt> [--每音节帧数 24] [--迭代 4] [--阻尼 0.5]
                        [--步数 10] [--采样 零起点|噪声] [--出 产物/音频_条件流]
                        [--不ASR] [--判据 产物/判据/判据_条件流.json] [--设备 auto|cpu|cuda]
  # 调试（不载 ASR，CER 记 None）：
  python 评测_条件流.py --ckpt 产物/_smoke_flow2/ckpt.pt --不ASR \
                        --出 产物/_smoke_flow_eval --判据 产物/_smoke_flow_eval/判据_条件流.json

输出：`<M66>/产物/判据/判据_条件流.json`（可用 --判据 覆盖，便于调试不污染正式判据）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
M66 = HERE.parent
AHI4 = M66.parent
for _p in (str(AHI4 / "meeting64" / "实施"), str(AHI4 / "meeting61" / "实施"),
           str(AHI4 / "meeting58" / "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 模型_条件流 as F  # noqa: E402
import 声码器与评测 as V  # noqa: E402

默认出 = M66 / "产物" / "音频_条件流"
默认判据 = M66 / "产物" / "判据" / "判据_条件流.json"

# 对照读数（v13 并行回归 + 阻尼迭代；只作参考，不参与判定）
对照v13 = dict(
    历史=0.8327, 留出=0.7533,
    说明="v13（Conv1D 并行回归 + 阻尼迭代）：历史 0.8327（4轮/0.5，探索性最好）"
         "｜留出 0.7533（7轮/0.5，预注册最好）；见 08_当前最佳与复现.md",
)

协议 = ("07_预注册_独立验证集.md：S13–S18 只用于选模型、不进入报告；"
       "对外报告只用历史 S1–S6 + 留出 S7–S12。")


def _sha256文件(路径) -> str:
    h = hashlib.sha256()
    with open(路径, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


def _检查API() -> None:
    """快速失败：核心 API 与任务契约不符时给出明确报错（本件不改核心）。"""
    缺 = [名 for 名 in ("加载条件流", "合成mel_条件流") if not hasattr(F, 名)]
    if 缺:
        raise RuntimeError("模型_条件流.py 缺少 API：%s（契约：加载条件流 / 合成mel_条件流）"
                           % "、".join(缺))
    for 名 in ("句集历史", "句集留出", "mel转波", "写wav", "量具模块", "期望长度"):
        if not hasattr(V, 名):
            raise RuntimeError("声码器与评测.py 缺少 API：%s" % 名)


def _json可写(o):
    """numpy / tuple → JSON 可写（递归）。"""
    if isinstance(o, dict):
        return {str(k): _json可写(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json可写(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _json可写(o.tolist())
    return o


def _mel统计(mel: np.ndarray) -> dict:
    m = np.asarray(mel, dtype=np.float64)
    有穷 = bool(np.isfinite(m).all())
    return dict(
        帧数=int(m.shape[0]), 带数=int(m.shape[1]),
        均值=round(float(np.nanmean(m)), 4) if 有穷 else None,
        标准差=round(float(np.nanstd(m)), 4) if 有穷 else None,
        含NaN或Inf=not 有穷,
    )


def _合成一句(模型, 文本: str, 每: int, 迭代: int, 阻尼: float, 步数: int,
            采样: str, 种子: int, 终值: str = "末轮") -> np.ndarray:
    """F.合成mel_条件流 → 未标准化 log-mel [T,80]（float32）。"""
    mel = F.合成mel_条件流(模型, str(文本), 每音节帧数=int(每), 迭代=int(迭代),
                          阻尼=float(阻尼), 步数=int(步数), 采样=str(采样),
                          终值=str(终值), 种子_=int(种子))
    return np.asarray(mel, dtype=np.float32)


def 主() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="meeting66 v18 条件流（CFM）双集 CER 评测")
    ap.add_argument("--ckpt", required=True, help="条件流 ckpt（模型_条件流.加载条件流 可读）")
    ap.add_argument("--每音节帧数", type=int, default=24)
    ap.add_argument("--迭代", type=int, default=4, help="自条件轮数（与 v13 的 3–7 轮同义）")
    ap.add_argument("--阻尼", type=float, default=0.5)
    ap.add_argument("--步数", type=int, default=10, help="Euler 积分步数")
    ap.add_argument("--采样", default="零起点", choices=["零起点", "噪声"])
    ap.add_argument("--终值", default="末轮", choices=["末轮", "平滑"],
                    help="末轮=最后一轮输出（同 v13）；平滑=阻尼后的估计（更稳）")
    ap.add_argument("--出", default=str(默认出), help="逐句 wav 输出目录")
    ap.add_argument("--不ASR", action="store_true",
                    help="调试：只合成 + 写 wav，不载 ASR（CER 记 None）")
    ap.add_argument("--判据", default=str(默认判据), help="判据 JSON 路径（调试时建议改到临时目录）")
    ap.add_argument("--种子", type=int, default=300, help="采样种子（零起点时仅占位）")
    ap.add_argument("--设备", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto=核心默认（有 CUDA 用 GPU）；GPU 在训练时可用 cpu 避开争用")
    a = ap.parse_args()

    _检查API()
    开始 = time.time()
    ckpt路 = Path(a.ckpt).resolve()
    if not ckpt路.exists():
        raise FileNotFoundError("ckpt 不存在：%s" % ckpt路)
    设备 = None if a.设备 == "auto" else ("cpu" if a.设备 == "cpu" else "cuda")
    if 设备 == "cuda":
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            raise RuntimeError("--设备 cuda 但 torch.cuda.is_available()=False")
    模型 = F.加载条件流(str(ckpt路), 设备=设备) if 设备 else F.加载条件流(str(ckpt路))
    训练信息 = dict(getattr(模型, "训练信息", {}) or {})
    参数数 = getattr(模型, "参数数", None)

    出目录 = Path(a.出)
    出目录.mkdir(parents=True, exist_ok=True)
    判据路 = Path(a.判据)

    推理设置 = dict(
        每音节帧数=int(a.每音节帧数), 迭代=int(a.迭代), 阻尼=float(a.阻尼),
        步数=int(a.步数), 采样=str(a.采样), 终值=str(a.终值), 种子=int(a.种子),
        设备=None,
        声码器="Griffin-Lim（V.mel转波：n_iter=64、random_state=0、峰值归一 0.9）",
        长度约定="T 帧 →（T−1）×220 采样点（center=True ISTFT）",
        不ASR=bool(a.不ASR),
    )
    推理设置["设备"] = str(next(模型.parameters()).device)
    print("[模型] 实际推理设备=%s（--设备=%s）"
          % (推理设置["设备"], a.设备), flush=True)

    print("[模型] ckpt=%s" % ckpt路, flush=True)
    print("[模型] 训练信息：%s" % json.dumps(
        {k: 训练信息.get(k) for k in ("轮", "训练损失", "验证损失", "完成", "范式")},
        ensure_ascii=False), flush=True)
    print("[模型] 参数数=%s ｜ 推理设置=%s"
          % (f"{参数数:,}" if 参数数 else "?", json.dumps(推理设置, ensure_ascii=False)),
          flush=True)

    # —— ① 确定性自检（C13 延伸）：同 ckpt / 同文本两趟 mel 逐位同 ——
    确定 = dict(两趟mel逐位同=None, 说明="同 ckpt、同文本、同设置两趟推理；C13 延伸")
    确定句 = "今天的风很轻，适合慢慢地说一会儿话。"
    try:
        m1 = _合成一句(模型, 确定句, a.每音节帧数, a.迭代, a.阻尼, a.步数, a.采样, a.种子,
                      终值=a.终值)
        m2 = _合成一句(模型, 确定句, a.每音节帧数, a.迭代, a.阻尼, a.步数, a.采样, a.种子,
                      终值=a.终值)
        确定["两趟mel逐位同"] = bool(np.array_equal(m1, m2))
        确定["形状"] = list(m1.shape)
    except Exception as e:  # noqa: BLE001
        确定["错误"] = "%s: %s" % (type(e).__name__, str(e)[:160])
    print("[确定性] 两趟 mel 逐位同=%s" % 确定["两趟mel逐位同"], flush=True)

    # —— ② 12 句合成（历史 S1–S6 / 留出 S7–S12）—— 
    记录: list = []
    异常: list = []
    for 集合名, 句s in (("历史", V.句集历史), ("留出", V.句集留出)):
        for 句号, 句 in 句s:
            t0 = time.time()
            mel = _合成一句(模型, 句, a.每音节帧数, a.迭代, a.阻尼, a.步数, a.采样, a.种子,
                          终值=a.终值)
            if mel.ndim != 2 or mel.shape[1] != 80 or mel.shape[0] < 1:
                raise RuntimeError("%s·%s 合成 mel 形状非法：%s"
                                   % (集合名, 句号, tuple(mel.shape)))
            统计 = _mel统计(mel)
            if 统计["含NaN或Inf"]:
                异常.append("%s·%s 合成 mel 含 NaN/Inf" % (集合名, 句号))
            波形 = V.mel转波(mel)
            p = 出目录 / ("条件流_%s_%s.wav" % (集合名, 句号))
            V.写wav(p, 波形)
            期 = int(V.期望长度(int(mel.shape[0])))
            长度正确 = bool(abs(int(波形.size) - 期) <= int(V.跳长))
            if not 长度正确:
                异常.append("%s·%s 波形长度不符：实际 %d，期望 %d（±%d）"
                            % (集合名, 句号, 波形.size, 期, V.跳长))
            if not p.exists() or p.stat().st_size <= 0:
                异常.append("%s·%s wav 未落盘或为空" % (集合名, 句号))
            条 = dict(
                集合=集合名, 句号=句号, 文=句, 路径=str(p),
                mel形状=list(mel.shape), 帧数=统计["帧数"],
                mel均值=统计["均值"], mel标准差=统计["标准差"],
                波形样本数=int(波形.size), 期望样本数=期, 长度正确=长度正确,
                时长秒=round(float(波形.size) / float(V.SR), 3),
                峰值=round(float(np.abs(波形).max()) if 波形.size else 0.0, 4),
                sha256=_sha256文件(p),
                合成秒=round(time.time() - t0, 3),
                CER=None, 两趟一致=None, 转写=None,
            )
            记录.append(条)
            print("[合成] %-2s %-4s 帧=%4d 样本=%6d 期望=%6d 长度=%s 时长=%.2fs 峰=%.2f melσ=%.2f (%.2fs)"
                  % (集合名, 句号, 条["帧数"], 条["波形样本数"], 期,
                     "OK" if 长度正确 else "BAD", 条["时长秒"], 条["峰值"],
                     统计["标准差"] or 0.0, 条["合成秒"]), flush=True)

    # —— ③ ASR 两趟转写（除非 --不ASR）——
    量 = None
    if not a.不ASR:
        print("[ASR] 载引擎（Paraformer-seaco-large，CPU；全机只许一个 ASR 进程）…", flush=True)
        量 = V.量具模块()
        引 = 量.载引擎()
        print("[ASR] 引擎 = %s" % 引.get("名"), flush=True)
        for 条 in 记录:
            t1 = 引["转写"](条["路径"])
            t2 = 引["转写"](条["路径"])
            cer = float(量.CER(条["文"], t1))
            条.update(转写=t1, 转写各趟=[t1, t2], 两趟一致=bool(t1 == t2), CER=cer)
            if not 条["两趟一致"]:
                异常.append("%s·%s 两趟转写不一致" % (条["集合"], 条["句号"]))
            print("  [ASR] %-2s %-4s CER=%.4f 两趟一致=%s 转写=%s"
                  % (条["集合"], 条["句号"], cer, 条["两趟一致"], t1[:44]), flush=True)
    else:
        print("[ASR] --不ASR：跳过（CER 记 None；不载引擎）", flush=True)

    # —— ④ 汇总（历史 / 留出均值）—— 
    汇总: dict = {}
    for 名 in ("历史", "留出"):
        组 = [r for r in 记录 if r["集合"] == 名]
        cers = [r["CER"] for r in 组 if r["CER"] is not None]
        汇总[名] = dict(
            句数=len(组),
            CER均值=(round(float(np.mean(cers)), 4) if cers else None),
            两趟一致=(bool(all(r["两趟一致"] for r in 组))
                     if (组 and all(r["两趟一致"] is not None for r in 组)) else None),
            长度全对=bool(all(r["长度正确"] for r in 组)),
            总时长秒=round(float(sum(r["时长秒"] for r in 组)), 3),
        )
    双集 = [汇总[名]["CER均值"] for 名 in ("历史", "留出")
            if 汇总[名]["CER均值"] is not None]
    汇总["双集均值"] = round(float(np.mean(双集)), 4) if len(双集) == 2 else None
    汇总["对照v13"] = dict(对照v13)
    汇总["Δvs_v13"] = (
        dict(历史=round(汇总["历史"]["CER均值"] - 对照v13["历史"], 4),
             留出=round(汇总["留出"]["CER均值"] - 对照v13["留出"], 4))
        if len(双集) == 2 else None
    )

    # —— 预注册判据（10_预实验_条件VAE_flow.md §5.5，先于 v18 读数写定）——
    判定 = None
    if len(双集) == 2:
        历, 留 = 汇总["历史"]["CER均值"], 汇总["留出"]["CER均值"]
        条件A = bool(历 <= 对照v13["历史"] and 留 <= 对照v13["留出"])
        条件B = bool((历 <= 对照v13["历史"] - 0.02 or 留 <= 对照v13["留出"] - 0.02)
                     and 历 <= 对照v13["历史"] + 0.02 and 留 <= 对照v13["留出"] + 0.02)
        判定 = dict(
            判据="10_预实验_条件VAE_flow.md §5.5（先于 v18 读数写定）",
            条件A_双集均不差=条件A,
            条件B_单集胜出且另一集不差=条件B,
            成功=bool(条件A or 条件B),
            结论=("成功（值得继续投入）" if (条件A or 条件B) else "无效（负结果）"),
            读数=dict(历史=历, 留出=留,
                      对照=dict(历史=对照v13["历史"], 留出=对照v13["留出"])))
        print("[判定] %s（A=%s B=%s）" % (判定["结论"], 条件A, 条件B), flush=True)

    print("=" * 78)
    for 名 in ("历史", "留出"):
        s = 汇总[名]
        print("%s集：%d 句 ｜ CER均值=%s ｜ 两趟一致=%s ｜ 长度全对=%s ｜ 总时长=%.2fs"
              % (名, s["句数"],
                 ("%.4f" % s["CER均值"]) if s["CER均值"] is not None else "None（--不ASR）",
                 s["两趟一致"], s["长度全对"], s["总时长秒"]))
    if not a.不ASR:
        print("双集均值 CER=%s" % (("%.4f" % 汇总["双集均值"]) if 汇总["双集均值"] is not None else "None"))
        print("对照 v13：历史 %.4f（Δ%+.4f）｜ 留出 %.4f（Δ%+.4f）"
              % (对照v13["历史"], 汇总["Δvs_v13"]["历史"],
                 对照v13["留出"], 汇总["Δvs_v13"]["留出"]))
    print("=" * 78)

    # —— ⑤ 判据 JSON 落盘 —— 
    判据路.parent.mkdir(parents=True, exist_ok=True)
    报 = dict(
        工具="评测_条件流.py", 版本="1.0",
        任务="meeting66 v18 条件流（CFM）双集 CER 评测",
        协议=协议,
        开始时间=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(开始)),
        结束时间=time.strftime("%Y-%m-%d %H:%M:%S"),
        ckpt=dict(
            路径=str(ckpt路),
            sha256=_sha256文件(ckpt路),
            字节=int(ckpt路.stat().st_size),
            修改时间=time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(ckpt路.stat().st_mtime)),
            参数数=参数数,
            训练信息=训练信息,
            训练完成=bool(训练信息.get("完成")),
        ),
        推理设置=推理设置,
        确定性=确定,
        对照v13=对照v13,
        判定=判定,
        逐句=记录,
        汇总=汇总,
        异常=异常,
        耗时秒=round(time.time() - 开始, 1),
        产物=dict(音频目录=str(出目录), 判据=str(判据路)),
        复现命令=('$env:PYTHONIOENCODING="utf-8"; & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" '
                 '"%s" --ckpt "%s" --每音节帧数 %d --迭代 %d --阻尼 %g --步数 %d --采样 %s'
                 ' --终值 %s --设备 %s'
                 % (Path(__file__).resolve(), ckpt路, 推理设置["每音节帧数"],
                    推理设置["迭代"], 推理设置["阻尼"], 推理设置["步数"], 推理设置["采样"],
                    推理设置["终值"], str(a.设备))),
    )
    判据路.write_text(json.dumps(_json可写(报), ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print("→ 判据 JSON：%s" % 判据路, flush=True)
    if 异常:
        print("异常 %d 条：" % len(异常))
        for 项 in 异常:
            print("  ! " + str(项))
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
