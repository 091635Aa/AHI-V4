# -*- coding: utf-8 -*-
"""meeting67/团队/声学_物理建模/探针_渲染上限_降级.py —— G0 器官渲染上限探针 · **v1.3 降级口径**。

★ 降级原因：`<path>\SpeakerA7代模型训练数据` 在 2026-10-21 10:50 掉线（Test-Path <path>\ = False，全盘不见），
  原始 wav 不可达。经 Lead 批准改走 v1.3（先于任何 CER 读数登记，不事后改判据）：

  数据源：`meeting66/产物/数据/样本/*.npz`（真 mel，SpeakerA，D: 盘，167 条；mel [T,80] 22.05k/10ms hop）
  取法  ：按 `meta.json` 的 `帧数` 升序、`文件` 字典序 tie-break，取前 N 条（= 最短的 N 条；确定性）
  链路  ：真 mel → Griffin-Lim（复用 声码器与评测.mel转波，64 迭代/种子 0）
          → `wav_GL` → 从 **wav_GL** 提取 oracle 控制（F0/清浊/LPC18 包络）→ 器官臂渲染 → ASR
  参考文本：`meta.json` 的 `文本`（**原始转写，不用 ASR 自转写**，防循环）
  护栏 G-1'：`真mel_GL` 臂 = wav_GL 直接 ASR vs 原始转写，须 ≤0.15（否则仪器/口径坏，本轮作废）
  判据  ：P2 均值 CER ≤0.20 PASS ／ (0.20,0.35] MARGINAL ／ >0.35 FAIL；P1/P3 按原角色
  口径声明：oracle 控制来自 **GL 重合成波形**而非真实录音；只判「器官类能否保住给定声学规格里的
          可懂度」，**不与 meeting66 整句读数跨口径比较**；绝对 CER 受 GL 与片段长度影响。

用法：
  & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" "…\\探针_渲染上限_降级.py" --条数 6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AHI4 = HERE.parents[2]
M66 = AHI4 / "meeting66"
样本目录 = M66 / "产物" / "数据" / "样本"
meta路 = M66 / "产物" / "数据" / "meta.json"
出目录 = HERE / "产物"
默认输出 = 出目录 / "探针_渲染上限_降级.json"
G1阈值 = 0.15
判据 = dict(PASS=0.20, MARGINAL=0.35)


def _按路径载(名: str, 路: Path):
    spec = importlib.util.spec_from_file_location(名, str(路))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def 器官():
    return _按路径载("探针_渲染上限_只读", HERE / "探针_渲染上限.py")


def 选片(条数: int) -> list:
    """确定性取法：meta 样本按 (帧数, 文件) 升序取前 N。"""
    meta = json.loads(meta路.read_text(encoding="utf-8"))
    样本 = meta["样本"]
    序 = sorted(样本, key=lambda r: (int(r.get("帧数", 10 ** 9)), str(r.get("文件", ""))))
    return 序[: int(条数)]


def 主(条数: int = 6, 输出: Path | None = None) -> dict:
    输出 = 输出 or 默认输出
    os.makedirs(str(出目录), exist_ok=True)
    P = 器官()
    V = P.量具()
    S = P.声码()
    片s = 选片(条数)
    引 = V.载引擎()
    print("[ASR] %s ｜ 片数=%d ｜ 总时长≈%.1fs"
          % (引["名"], len(片s), sum(r.get("帧数", 0) * 0.01 for r in 片s)), flush=True)
    开始 = time.time()
    逐条 = []
    for i, r in enumerate(片s):
        帧数 = int(r["帧数"])
        T_h = str(r.get("文本", ""))
        z = np.load(str(样本目录 / str(r["文件"])))
        mel = np.asarray(z["mel"], dtype=np.float64)
        wavGL = S.mel转波(mel)                        # 真 mel → GL（确定性）
        路GL = 出目录 / ("降级_GL_%s.wav" % Path(str(r["文件"])).stem)
        V.写wav(str(路GL), wavGL, fs=P.SR)
        条 = dict(文件=str(r["文件"]), 帧数=帧数, 时长秒=round(len(wavGL) / P.SR, 2),
                  音节数=r.get("音节数"), 原时长秒=r.get("原时长秒"), T_h=T_h, 条件={})
        # ① 护栏 G-1'
        T_gl = 引["转写"](str(路GL))
        条["条件"]["真mel_GL"] = dict(CER=V.CER(T_h, T_gl), 转写=T_gl)
        print("[%d/%d] %s %.1fs GL_CER=%.4f「%s」"
              % (i + 1, len(片s), r["文件"], 条["时长秒"], 条["条件"]["真mel_GL"]["CER"], T_gl[:26]),
              flush=True)
        # ② 从 wav_GL 提 oracle 控制
        控 = P.提取控制(np.asarray(wavGL, dtype=np.float64))
        n = len(wavGL)
        条["控制"] = dict(帧数=控["帧数"], 浊帧占比=round(float(控["浊"].mean()), 3),
                          F0中位=round(float(np.median(控["f0"][控["浊"]])) if 控["浊"].any() else 0.0, 2),
                          k绝对最大=round(float(np.abs(控["k"]).max()), 3))
        # ③ 三条器官臂
        for 臂 in ("P1_脉冲_LPC全极点", "P2_声门源_LPC全极点", "P3_声门源_KL波导"):
            t0 = time.perf_counter()
            y = P.合成_臂(控, 臂, n, 种子=2000 + i)
            合成秒 = time.perf_counter() - t0
            路 = 出目录 / ("降级_%s_%s.wav" % (臂.split("_")[0], Path(str(r["文件"])).stem))
            V.写wav(str(路), y, fs=P.SR)
            T_s = 引["转写"](str(路))
            条["条件"][臂] = dict(CER=V.CER(T_h, T_s), 转写=T_s,
                                  合成秒=round(合成秒, 3), 实时率=round(合成秒 / (n / P.SR), 3))
            print("        %-20s CER=%.4f RTF=%.3f「%s」"
                  % (臂, 条["条件"][臂]["CER"], 条["条件"][臂]["实时率"], T_s[:26]), flush=True)
        # 确定性（不重跑 ASR）
        y1 = P.合成_臂(控, "P2_声门源_LPC全极点", n, 种子=2000 + i)
        y2 = P.合成_臂(控, "P2_声门源_LPC全极点", n, 种子=2000 + i)
        条["确定性_两趟逐位同"] = bool(P._sha256(y1) == P._sha256(y2))
        逐条.append(条)
    汇总 = {}
    for 臂 in ("真mel_GL", "P1_脉冲_LPC全极点", "P2_声门源_LPC全极点", "P3_声门源_KL波导"):
        vs = [x["条件"][臂]["CER"] for x in 逐条 if 臂 in x["条件"]]
        if not vs:
            continue
        均 = round(float(np.mean(vs)), 4)
        汇 = dict(CER均值=均, CER中位=round(float(np.median(vs)), 4),
                  CER范围=[round(float(min(vs)), 4), round(float(max(vs)), 4)], n=len(vs))
        if 臂 == "真mel_GL":
            汇["护栏G1"] = ("PASS" if 均 <= G1阈值 else "FAIL（仪器/口径坏，本轮作废）")
        if 臂 == "P2_声门源_LPC全极点":
            汇["判据"] = ("PASS" if 均 <= 判据["PASS"] else
                        ("MARGINAL" if 均 <= 判据["MARGINAL"] else "FAIL"))
        汇总[臂] = 汇
    报 = dict(
        探针="meeting67 step0 器官渲染上限 · v1.3 降级口径（真 mel 源，GL 上提取 oracle 控制）",
        口径=dict(
            数据源="meeting66/产物/数据/样本/*.npz（真 mel，SpeakerA，D: 盘）",
            取法="meta 样本按 (帧数, 文件) 升序取前 N（最短 N 条；确定性）",
            链路="真mel → mel转波(GL 64迭代/种子0) → wav_GL → 提 oracle 控制 → 器官臂 → ASR",
            参考文本="meta.json 的 文本（原始转写，不用 ASR 自转写，防循环）",
            护栏="真mel_GL 臂 CER ≤ %.2f（G-1'）" % G1阈值,
            判据="P2 均值 CER ≤%.2f PASS / ≤%.2f MARGINAL / 否则 FAIL" % (判据["PASS"], 判据["MARGINAL"]),
            降级原因="<path>\\SpeakerA7代模型训练数据 于 2026-10-21 10:50 掉线（全盘不可达）；v1.3 先于 CER 读数登记",
            声明=("oracle 控制来自 GL 重合成波形而非真实录音；本读数只判『器官类能否保住给定声学规格里的可懂度』，"
                  "不与 meeting66 整句读数跨口径直接比较；绝对 CER 受 GL 与片段长度影响"),
            ASR=引["名"], 时间戳=time.strftime("%Y-%m-%d %H:%M:%S")),
        逐条=逐条, 汇总=汇总, 耗时秒=round(time.time() - 开始, 1))
    P._写json(输出, 报)
    print("\n=== 汇总（对 meta 原始转写算 CER）===", flush=True)
    for 名, v in 汇总.items():
        print("%-22s 均值=%.4f 中位=%.4f 范围=%s n=%d %s"
              % (名, v["CER均值"], v["CER中位"], v["CER范围"], v["n"],
                 v.get("判据", v.get("护栏G1", ""))), flush=True)
    print("[落盘] %s（%.1fs）" % (输出, 报["耗时秒"]), flush=True)
    return 报


if __name__ == "__main__":
    派 = argparse.ArgumentParser()
    派.add_argument("--条数", type=int, default=6)
    派.add_argument("--输出", default=None)
    a = 派.parse_args()
    主(a.条数, Path(a.输出) if a.输出 else None)
    sys.exit(0)
