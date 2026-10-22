# -*- coding: utf-8 -*-
"""meeting69/实施/收尾_精修v1.py —— 轻微移调测试（±1.5 半音）+ 听测包合成。"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
出 = M69 / "产物" / "精修v1"
S2A = M69 / "产物" / "S2A"
M64 = M69.parent / "meeting64" / "实施"
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


精 = 载("音频线_v1_精修", HERE / "音频线_v1_精修.py")
协 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")
量 = 载("量具_v9", M64 / "量具_v9.py")


def 读(路):
    with wave.open(str(路), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return x


def 写(名, x):
    路 = 出 / 名
    协.写wav(路, x)
    return 路


def 静(s):
    return np.zeros(int(SR * s), np.float32)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    报 = dict(工具="收尾_精修v1.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 轻移调=[])
    # ±1.5 半音（在 A4 无移调版本上做纯重采样）
    引 = 量.载引擎()
    文 = "今天天气很好我听见了自己的声音我想再说一遍"
    for 名, 基, 半音 in (("高兴_A4_P1.5", "高兴_A4_P2.wav", 1.5),
                        ("低落_A4_P-1.5", "低落_A4_P2.wav", -1.5)):
        x = 读(出 / 基)
        y = 精.移调_重采样(x, 半音)
        y = 精.收尾(y)
        路 = 写(名 + ".wav", y)
        转 = 引["转写"](str(路))
        报["轻移调"].append(dict(项=名, CER=量.CER(文, 转), 转写=转, 时长秒=round(len(y) / SR, 3)))
        print(名, "CER=%.3f" % 报["轻移调"][-1]["CER"], 转[:26], flush=True)

    # ── 听测包 ──
    真人 = 读(S2A / "自声" / "多句_0.wav")  # 占位不用
    真人 = 读(M69 / "产物" / "S1" / "真实" / "(64)@10.wav")
    平静 = 读(出 / "句0_A4.wav")
    高兴 = 读(出 / "高兴_A4_P2.wav")
    低落 = 读(出 / "低落_A4_P2.wav")
    三档 = np.concatenate([平静, 静(1.0), 高兴, 静(1.0), 低落, 静(1.0), 真人])
    写("听测_精修v1_三档.wav", 三档)
    v0平 = 读(S2A / "自声" / "平静.wav")
    AB电 = np.concatenate([v0平, 静(1.0), 平静])
    写("听测_AB_电音_平静_v0_vs_A4.wav", AB电)
    v0高 = 读(出 / "高兴_A0_P0.wav")
    AB沙 = np.concatenate([v0高, 静(1.0), 高兴])
    写("听测_AB_沙哑_高兴_v0移调_vs_A4不移调.wav", AB沙)
    多句 = np.concatenate([读(出 / f"句{i}_A4.wav") for i in (1, 2, 3)])
    # 句间 0.8 s
    多句 = np.concatenate([读(出 / "句1_A4.wav"), 静(0.8), 读(出 / "句2_A4.wav"), 静(0.8), 读(出 / "句3_A4.wav")])
    写("听测_精修v1_多句.wav", 多句)
    报["听测包"] = dict(三档=round(len(三档) / SR, 2), AB电音=round(len(AB电) / SR, 2),
                        AB沙哑=round(len(AB沙) / SR, 2), 多句=round(len(多句) / SR, 2))
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "读数_收尾.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报["听测包"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(主())
