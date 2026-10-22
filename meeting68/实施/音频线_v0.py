# -*- coding: utf-8 -*-
"""meeting68/实施/音频线_v0.py —— 最小化音频线：文本 → 声音（可调情绪/风格）。

最小实现（全部本项目件，无外部模型；C1/C10）：
  文本 → 音素 token → v13 声学模型（meeting66 只读 ckpt）→ log-mel → Griffin-Lim → wav
  控制层（对应所有者三词）：
    · 转动频率 = 音高（半音，pitch-shift）
    · 特殊振幅 = 能量（dB 增益）
    · 发生概率 = 语速（每音节帧数；越少越快）
    · 谱倾斜  = 明亮/暗（mel 带倾斜，dB）
情绪/风格只是这 4 个旋钮的预置；学习引擎优化的是同一组旋钮。

用法：
  & $py 音频线_v0.py 合成 "今天的风很轻，适合慢慢地说一会儿话。" --情绪 高兴 --风格 慢
  & $py 音频线_v0.py 列出样式
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SR = 22050
M66 = Path(r"<workspace>/03_AHI\AHI4\meeting66")
HERE = Path(__file__).resolve().parent
CKPT = M66 / "产物" / "模型_v13_K4_320" / "ckpt_e0150.pt"

# ── 情绪：4 个旋钮的预置（不是「情绪模型」，是参数化模仿；登记为参数级） ──
情绪表 = {
    "平静": dict(音高=0.0, 增益=0.0, 倾斜=0.0, 每音节帧数=24, 停顿毫秒=60),
    "高兴": dict(音高=+3.0, 增益=+2.0, 倾斜=+1.5, 每音节帧数=23, 停顿毫秒=40),
    "低落": dict(音高=-3.0, 增益=-2.5, 倾斜=-1.5, 每音节帧数=27, 停顿毫秒=180),
    "激动": dict(音高=+5.0, 增益=+3.5, 倾斜=+2.5, 每音节帧数=21, 停顿毫秒=0),
    "温柔": dict(音高=-1.0, 增益=-1.0, 倾斜=-0.5, 每音节帧数=26, 停顿毫秒=120),
}
风格表 = {"慢": dict(每音节帧数=28, 停顿毫秒=150), "正常": dict(每音节帧数=24, 停顿毫秒=60), "快": dict(每音节帧数=20, 停顿毫秒=0)}

_声 = None
_模型 = None


def _声码器():
    global _声
    if _声 is None:
        p = M66 / "实施" / "声码器与评测.py"
        s = importlib.util.spec_from_file_location("声码器_只读", p)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        _声 = m
    return _声


def 载模型():
    global _模型
    if _模型 is None:
        _模型 = _声码器().载ckpt(CKPT, 设备="cpu")
    return _模型


def 参数(情绪="平静", 风格="正常", 音高=None, 增益=None, 倾斜=None, 每音节帧数=None, 停顿毫秒=None):
    p = dict(情绪表.get(情绪, 情绪表["平静"]))
    p.update(风格表.get(风格, {}))
    if 音高 is not None: p["音高"] = float(音高)
    if 增益 is not None: p["增益"] = float(增益)
    if 倾斜 is not None: p["倾斜"] = float(倾斜)
    if 每音节帧数 is not None: p["每音节帧数"] = int(每音节帧数)
    if 停顿毫秒 is not None: p["停顿毫秒"] = float(停顿毫秒)
    return p


def _插停顿(波形: np.ndarray, 文本: str, 毫秒: float) -> np.ndarray:
    """在标点处插入静音（停顿/语速风格控制；按字符位置比例近似放置，确定性）。"""
    if 毫秒 <= 0 or 波形.size == 0:
        return 波形
    标 = "，。！？；、,.!?;：:"
    位 = [i for i, ch in enumerate(文本) if ch in 标]
    if not 位:
        位 = [max(1, len(文本) // 2)]
    n = max(1, len(文本))
    s = 波形.size
    段s = []
    上 = 0
    for i in 位:
        t = min(max(int(round(s * i / n)), 上), s)
        段s.append(波形[上:t])
        段s.append(np.zeros(int(SR * 毫秒 / 1000.0), dtype=np.float32))
        上 = t
    段s.append(波形[上:])
    return np.concatenate([a for a in 段s if a.size > 0]).astype(np.float32)


def 合成(文本: str, 情绪="平静", 风格="正常", 音高=None, 增益=None, 倾斜=None,
         每音节帧数=None, 停顿毫秒=None, 迭代=4, 阻尼=0.5, 保存=None, 采样嵌入=True):
    """文本 → wav（返回 dict：控制参数/路径/sha256/时长/峰值/耳嵌入）。"""
    p = 参数(情绪, 风格, 音高, 增益, 倾斜, 每音节帧数, 停顿毫秒)
    os.environ["M66_迭代推理"] = str(int(迭代))
    os.environ["M66_阻尼"] = str(float(阻尼))
    声 = _声码器()
    t0 = time.time()
    mel, 有效音节数, 跳过 = 声.推理mel(载模型(), 文本, int(p["每音节帧数"]))
    # 谱倾斜：在 mel 域加带倾斜（低带 −tilt、高带 +tilt），不增加额外模型
    if abs(float(p["倾斜"])) > 1e-9:
        tilt = np.linspace(-1.0, 1.0, mel.shape[1], dtype=np.float32) * float(p["倾斜"])
        mel = (mel + tilt[None, :]).astype(np.float32)
    波形 = 声.mel转波(mel)
    # 音高（转动频率）：确定性 pitch-shift
    if abs(float(p["音高"])) > 1e-9:
        import librosa
        波形 = librosa.effects.pitch_shift(波形, sr=SR, n_steps=float(p["音高"]))
    # 能量（特殊振幅）：dB 增益
    if abs(float(p["增益"])) > 1e-9:
        波形 = 波形 * (10.0 ** (float(p["增益"]) / 20.0))
    # 停顿（发生概率/节奏）：标点处插入静音
    if float(p.get("停顿毫秒", 0.0)) > 1e-6:
        波形 = _插停顿(波形, 文本, float(p["停顿毫秒"]))
    峰 = float(np.abs(波形).max()) if 波形.size else 0.0
    if 峰 > 1e-9:
        波形 = 波形 / 峰 * 0.9
    波形 = np.asarray(波形, dtype=np.float32)
    出 = dict(文本=文本, 情绪=情绪, 风格=风格, 控制=dict(p), 有效音节数=有效音节数,
             跳过音节=跳过, 时长秒=round(float(波形.size) / SR, 3),
             峰值=round(float(np.abs(波形).max()), 4), 合成耗时秒=round(time.time() - t0, 3))
    if 保存:
        Path(保存).parent.mkdir(parents=True, exist_ok=True)
        声.写wav(Path(保存), 波形)
        出["路径"] = str(保存)
        出["sha256"] = hashlib.sha256(Path(保存).read_bytes()).hexdigest()
    if 采样嵌入:
        import importlib.util as _u
        s = _u.spec_from_file_location("内生耳", HERE / "内生耳.py")
        m = _u.module_from_spec(s); s.loader.exec_module(m)
        出["耳嵌入"] = m.耳嵌入(出.get("路径")) if 出.get("路径") else None
    出["波形"] = 波形
    return 出


def 主(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="meeting68 最小化音频线 v0")
    子 = ap.add_subparsers(dest="命令", required=True)
    p = 子.add_parser("合成", help="文本 → wav")
    p.add_argument("文本")
    p.add_argument("--情绪", default="平静")
    p.add_argument("--风格", default="正常")
    p.add_argument("--音高", type=float, default=None)
    p.add_argument("--增益", type=float, default=None)
    p.add_argument("--倾斜", type=float, default=None)
    p.add_argument("--每音节帧数", type=int, default=None)
    p.add_argument("--出", default=None)
    p2 = 子.add_parser("列出样式", help="列出情绪/风格预置")
    参 = ap.parse_args(argv)
    if 参.命令 == "列出样式":
        print(json.dumps(dict(情绪=情绪表, 风格=风格表), ensure_ascii=False, indent=1))
        return 0
    出 = 参.出 or str(HERE.parent / "产物" / "音频线" / f"线_{参.情绪}_{参.风格}.wav")
    r = 合成(参.文本, 参.情绪, 参.风格, 参.音高, 参.增益, 参.倾斜, 参.每音节帧数, 保存=出)
    r.pop("波形", None)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
