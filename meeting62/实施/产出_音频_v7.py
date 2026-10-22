# -*- coding: utf-8 -*-
"""meeting62/实施/产出_音频_v7.py —— 交付音频：说话三角色 + 按谱唱歌 + 参考/v6 A/B 对照。

纯新增件（C14）：全部产物落在 `meeting62/产物/` 下，绝不写 meeting61。
只读 import meeting61/58 既有件 + meeting62 的 v7 件。
C1/C9/C10：纯 numpy/scipy + 既有件；无 TTS/RVC/神经声码器。C13：同种子两趟逐位同。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M62 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M62)
MEET61 = os.path.join(AHI4, "meeting61")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))
sys.path.insert(0, os.path.join(MEET61, "实施"))

import 契约_谱与亚版本 as 契  # noqa: E402
import 声线库 as 库  # noqa: E402
import 声带后端_v7 as 后7  # noqa: E402
import 声带后端_v6 as 后6  # noqa: E402
import 谱_亚版本_声库_v7 as 唱7  # noqa: E402
import 谱_亚版本_声库 as 唱6  # noqa: E402   （Task 3：v6 谱管线）
import 声线覆盖层_v7 as 覆  # noqa: E402   （N2 修复：三角色声线档案覆盖层）

声带V7 = 后7.声带V7
声带V6 = 后6.声带V6
出 = os.path.join(M62, "产物", "交付_v7")


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _f0中位(w):
    import 谱_亚版本_声库_v7 as P
    ts, f0 = P._f0轨(np.asarray(w, dtype=np.float64))
    v = f0[f0 > 0]
    return round(float(np.median(v)), 1) if v.size else 0.0


def 主() -> dict:
    os.makedirs(出, exist_ok=True)
    说目录 = os.path.join(出, "说话")
    唱目录 = os.path.join(出, "唱歌")
    参目录 = os.path.join(出, "对照_参考原声")
    os.makedirs(说目录, exist_ok=True)
    os.makedirs(唱目录, exist_ok=True)
    os.makedirs(参目录, exist_ok=True)

    句 = "今天的风很轻，适合慢慢地说一会儿话。"
    说读数 = {}
    for 名 in ("洛天依", "SpeakerA", "EmoCompanion"):
        # N2 修复：经 meeting62「声线档案覆盖层」派生（共振峰缩放 3 轴 + 谱倾斜，
        # 由 meeting61 已固化的参考录音统计确定性派生）⇒ 引擎真实合成路径生效。
        音 = 覆.取(名).转dict()
        b7 = 声带V7(seed=300, 目录=说目录, 台账路径=os.path.join(说目录, "台账.jsonl"))
        r7 = b7.发声(np.zeros(1024), 句, 音色=音)
        # 同名文件（后端自增编号）→ 重命名为角色名
        p7 = os.path.join(说目录, r7["wav"])
        p7n = os.path.join(说目录, "说话_v7_%s.wav" % 名)
        if os.path.exists(p7) and p7 != p7n:
            shutil.move(p7, p7n)
        说读数[名] = dict(v7_wav=p7n, v7_sha256=_sha(p7n), 时长秒=r7["时长秒"],
                       峰值=r7["自检"]["峰值"], 实时率=r7["实时率"], 通过=r7["通过"],
                       F0中位Hz=_f0中位(r7.get("波形")) if r7.get("波形") is not None else None)
        # v6 A/B 对照（仅EmoCompanion；用于「修①②」听感对照）
        if 名 == "EmoCompanion":
            b6 = 声带V6(seed=300, 目录=os.path.join(说目录, "对照_v6"),
                        台账路径=os.path.join(说目录, "对照_v6台账.jsonl"))
            r6 = b6.发声(np.zeros(1024), 句, 音色=音)
            p6 = os.path.join(说目录, "对照_v6", r6["wav"])
            p6n = os.path.join(说目录, "说话_v6对照_EmoCompanion.wav")
            if os.path.exists(p6):
                shutil.move(p6, p6n)
            说读数[名]["v6对照_wav"] = p6n
            说读数[名]["v6对照_sha256"] = _sha(p6n)
            说读数[名]["v6对照_F0中位Hz"] = _f0中位(r6.get("波形")) if r6.get("波形") is not None else None

    # —— 按谱唱歌（Task 3 谱管线 → 驱动 声带后端_v7）——
    唱读数 = {}
    谱路 = os.path.join(M62, "素材", "小谱_样本.json")
    谱对象 = 契.谱.从JSON文件(谱路)
    亚版本 = 契.建亚版本(谱对象)
    r7 = 唱7.谱声带V7(种子=300, 目录=唱目录).唱(亚版本, 名="唱歌_v7_小谱_样本")
    # v6 对照（Task 3 原管线）
    r6 = 唱6.谱声带(种子=300, 目录=os.path.join(唱目录, "对照_v6")).唱(亚版本, 名="唱歌_v6对照_小谱_样本")
    p7 = os.path.join(唱目录, "唱歌_v7_小谱_样本.wav")
    p6 = os.path.join(唱目录, "对照_v6", "唱歌_v6对照_小谱_样本.wav")
    唱读数["小谱样本"] = dict(v7_wav=p7, v7_sha256=_sha(p7), 时长秒=r7["时长秒"],
                           峰值=r7["峰值"], 实时率=r7["实时率"],
                           v6对照_wav=p6 if os.path.exists(p6) else None,
                           v6对照_sha256=_sha(p6) if os.path.exists(p6) else None,
                           f0实测=唱7.测音符(r7["波形"], 亚版本))
    # 长音变体（颤音）
    长谱路 = os.path.join(M62, "素材", "小谱_长音.json")
    if os.path.exists(长谱路):
        长亚 = 契.建亚版本(契.谱.从JSON文件(长谱路))
        rl = 唱7.谱声带V7(种子=300, 目录=唱目录).唱(长亚, 名="唱歌_v7_长音")
        pl = os.path.join(唱目录, "唱歌_v7_长音.wav")
        # v6 对照（谱_亚版本_声库 的 谱韵律 用 5.2Hz/0.35半音 幅度?)
        rl6 = 唱6.谱声带(种子=300, 目录=os.path.join(唱目录, "对照_v6")).唱(长亚, 名="唱歌_v6对照_长音")
        pl6 = os.path.join(唱目录, "对照_v6", "唱歌_v6对照_长音.wav")
        唱读数["长音"] = dict(v7_wav=pl, v7_sha256=_sha(pl), 时长秒=rl["时长秒"],
                            v6对照_wav=pl6 if os.path.exists(pl6) else None)

    # —— 参考 A/B（人类原声，只读拷贝 3 个 ref 作对照，不改原文件）——
    for 名, f in (("EmoCompanion", "ref44_EmoCompanion_01.wav"), ("SpeakerA", "ref44_SpeakerA_01.wav"),
                 ("洛天依", "ref44_洛天依_清唱_01.wav")):
        s = os.path.join(MEET61, "素材", 名, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(参目录, "参考_%s_%s" % (名, f)))

    读数 = dict(件="meeting62/产物/交付_v7", 目录=出, 说话=说读数, 唱歌=唱读数,
               参考=os.listdir(参目录),
               声线覆盖层=覆.说明(),
               说明="说话三角色（v7，声线经 meeting62「声线档案覆盖层」派生：共振峰缩放3轴+谱倾斜）"
                   "+ 按谱唱歌（Task3 谱管线驱动 声带后端_v7）+ v6 对照 + 人类参考原声拷贝；"
                   "口径＝参数化模仿，非克隆")
    with open(os.path.join(出, "读数_交付_v7.json"), "w", encoding="utf-8") as f:
        json.dump(读数, f, ensure_ascii=False, indent=1)
    return 读数


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    r = 主()
    print("=" * 88)
    for n, v in r["说话"].items():
        print("说话 v7 %-4s %s 时长%.2fs F0中位%.1f" % (n, v["v7_wav"], v["时长秒"], v["F0中位Hz"] or 0))
        if v.get("v6对照_wav"):
            print("    v6对照 %s F0中位%.1f" % (v["v6对照_wav"], v["v6对照_F0中位Hz"] or 0))
    for n, v in r["唱歌"].items():
        print("唱歌 v7 %-6s %s 时长%.2fs%s" % (n, v["v7_wav"], v["时长秒"],
              ("  v6对照 " + v["v6对照_wav"]) if v.get("v6对照_wav") else ""))
    print("参考拷贝:", r["参考"])
    print("读数: %s" % os.path.join(r["目录"], "读数_交付_v7.json"))
