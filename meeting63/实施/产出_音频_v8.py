# -*- coding: utf-8 -*-
"""meeting63/实施/产出_音频_v8.py —— 交付音频：说话三角色（v8）＋ v7 A/B ＋ 人类参考。

纯新增件（C14）：全部产物落在 `meeting63/产物/交付_v8/`，绝不写 meeting61/62 产物。
只读 import meeting61/62 既有件 + meeting63 的 v8 件。C1/C9/C10：纯 numpy/scipy。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M63 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M63)
MEET61 = os.path.join(AHI4, "meeting61")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting62", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 声带后端_v8 as 后8  # noqa: E402
import 声带后端_v7 as 后7  # noqa: E402
import 声线覆盖层_v7 as 覆  # noqa: E402

出 = os.path.join(M63, "产物", "交付_v8_1")
旧v8 = os.path.join(M63, "产物", "交付_v8", "说话")
句 = "今天的风很轻，适合慢慢地说一会儿话。"


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def 主() -> dict:
    os.makedirs(出, exist_ok=True)
    说目录 = os.path.join(出, "说话")
    参目录 = os.path.join(出, "对照_参考原声")
    os.makedirs(说目录, exist_ok=True)
    os.makedirs(参目录, exist_ok=True)
    读数 = {}
    for 名 in ("洛天依", "SpeakerA", "EmoCompanion"):
        音 = 覆.取(名).转dict()
        b8 = 后8.声带V8(seed=300, 目录=说目录, 台账路径=os.path.join(说目录, "台账_v8.jsonl"))
        r8 = b8.发声(np.zeros(1024), 句, 音色=音)
        p8 = os.path.join(说目录, r8["wav"])
        n8 = os.path.join(说目录, "说话_v8_1_%s.wav" % 名)
        if os.path.exists(p8) and p8 != n8:
            shutil.move(p8, n8)
        读数[名] = dict(v8_wav=n8, v8_sha256=_sha(n8), 时长秒=r8["时长秒"],
                     实时率=r8["实时率"], 通过=r8["通过"])
        if 名 == "EmoCompanion":
            b7 = 后7.声带V7(seed=300, 目录=os.path.join(说目录, "对照_v7"),
                            台账路径=os.path.join(说目录, "对照_v7台账.jsonl"))
            r7 = b7.发声(np.zeros(1024), 句, 音色=音)
            p7 = os.path.join(说目录, "对照_v7", r7["wav"])
            n7 = os.path.join(说目录, "说话_v7对照_EmoCompanion.wav")
            if os.path.exists(p7):
                shutil.move(p7, n7)
            读数[名]["v7对照_wav"] = n7
            读数[名]["v7对照_sha256"] = _sha(n7)
    # v8 对照（上一轮交付件，只拷贝，不重生成；旧目录不动）
    对v8 = os.path.join(说目录, "对照_v8")
    os.makedirs(对v8, exist_ok=True)
    for 名 in ("洛天依", "SpeakerA", "EmoCompanion"):
        s = os.path.join(旧v8, "说话_v8_%s.wav" % 名)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(对v8, os.path.basename(s)))
    for 名, f in (("EmoCompanion", "ref44_EmoCompanion_01.wav"), ("SpeakerA", "ref44_SpeakerA_01.wav"),
                 ("洛天依", "ref44_洛天依_清唱_01.wav")):
        s = os.path.join(MEET61, "素材", 名, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(参目录, "参考_%s_%s" % (名, f)))
    报 = dict(件="meeting63/产物/交付_v8_1", 目录=出, 句=句, 说话=读数,
             参考=os.listdir(参目录),
             声线覆盖层=覆.说明(),
             说明="说话三角色（v8.1：鼻尾全音节化 8/7 帧 + 持续包络）＋ v8/v7 A/B ＋ 人类参考拷贝；"
                 "音色参数零改动；口径＝参数化模仿，非克隆")
    with open(os.path.join(出, "读数_交付_v8_1.json"), "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    return 报


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    r = 主()
    print("=" * 80)
    for n, v in r["说话"].items():
        print("说话 v8.1 %-4s %s 时长%.2fs 实时率%.4f" % (n, v["v8_wav"], v["时长秒"], v["实时率"]))
    print("参考:", r["参考"])
    print("读数:", os.path.join(r["目录"], "读数_交付_v8_1.json"))
