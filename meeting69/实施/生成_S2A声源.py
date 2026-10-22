# -*- coding: utf-8 -*-
"""meeting69/实施/生成_S2A声源.py —— S2-A 声源（预注册 `24`）：AHI 自声 3 条 + 退化臂。

文本外采（宪法第四条：词音可外采）；声音由 `meeting68/实施/音频线_v0.合成` 产出
（v13 只读 ckpt → log-mel → Griffin-Lim，CPU，确定性）。只写 `meeting69/产物/S2A/`（C14）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
M68 = M69.parent / "meeting68" / "实施"
出 = M69 / "产物" / "S2A"
SR = 22050

文本 = "今天天气很好我听见了自己的声音我想再说一遍"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


音频线 = 载("音频线_v0", M68 / "音频线_v0.py")
协 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")


def 退化_加噪(x, 信噪比db, 种子=20261022):
    rng = np.random.RandomState(种子)
    n = rng.randn(len(x)).astype(np.float32)
    功_x = float(np.mean(x.astype(np.float64) ** 2)) + 1e-12
    功_n = float(np.mean(n.astype(np.float64) ** 2))
    目标_n = 功_x / (10.0 ** (信噪比db / 10.0))
    n = n * np.sqrt(目标_n / (功_n + 1e-12))
    y = (x + n).astype(np.float32)
    p = float(np.abs(y).max())
    return (y / p * 0.9).astype(np.float32) if p > 0 else y


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    (出 / "自声").mkdir(parents=True, exist_ok=True)
    记 = dict(文本=文本, 工具="生成_S2A声源.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              声源="meeting68/音频线_v0（v13 ckpt_e0150，4 轮/阻尼0.5，CPU，确定性）", 项=[])
    for 情绪 in ("平静", "高兴", "低落"):
        路 = 出 / "自声" / f"{情绪}.wav"
        t0 = time.time()
        r = 音频线.合成(文本, 情绪=情绪, 风格="正常", 迭代=4, 阻尼=0.5, 保存=str(路), 采样嵌入=False)
        r.pop("波形", None)
        记["项"].append(dict(情绪=情绪, 路径=str(路), 控制=r["控制"], 时长秒=r["时长秒"],
                            有效音节数=r["有效音节数"], sha256=r["sha256"], 耗时秒=round(time.time() - t0, 1)))
    # 退化臂：平稳 + 白噪 SNR=5 dB
    x = 音频线.合成(文本, 情绪="平静", 风格="正常", 迭代=4, 阻尼=0.5, 保存=None, 采样嵌入=False)["波形"]
    y = 退化_加噪(x, 5.0)
    路 = 出 / "自声" / "平静_噪5dB.wav"
    协.写wav(路, y)
    记["退化臂"] = dict(源="平静", 信噪比dB=5.0, 种子=20261022, 路径=str(路),
                       sha256=hashlib.sha256(路.read_bytes()).hexdigest())
    (出 / "声源清单.json").write_text(json.dumps(记, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in 记.items() if k != "项"}, ensure_ascii=False, indent=1))
    for r in 记["项"]:
        print(r["情绪"], r["时长秒"], "s", r["sha256"][:16])
    return 0


if __name__ == "__main__":
    sys.exit(主())
