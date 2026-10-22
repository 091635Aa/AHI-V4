# -*- coding: utf-8 -*-
"""meeting60/实施/合成_测试组.py —— v3 修复版多组测试音频总装（44.1kHz）。

组定义（详见 00_修复简报.md §4）：
  A 旧版 v2（由 matrix 从 meeting59 只读复制）
  B 新引擎裸声（无人类韵律/无 EQ）
  C 新引擎 + 人类韵律（无 EQ）
  D 新引擎 + 人类韵律 + 轻 EQ（≤±3dB）——**主修复版**
  E 组 D 但音高 ×0.85（-2.8st，自然说话音高变体）
  F 组 D 的同句三声
  G 组 D 与 44.1k 参考片段的 A/B 对照（含参考原声）

输出：产物/测试组/组X_…/角色_*.wav + 组X_三角色.wav（一组一条，方便连续试听）
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
MEET60 = os.path.dirname(HERE)
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import 韵律_44k as H44  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = H44.FS
产物组 = os.path.join(MEET60, "产物", "测试组")
产物档案 = os.path.join(MEET60, "产物", "声线档案")
os.makedirs(产物组, exist_ok=True)

顺序 = ("EmoCompanion", "SpeakerA", "洛天依")
种子 = {"EmoCompanion": 300, "SpeakerA": 317, "洛天依": 334}
角色文本 = {
    "EmoCompanion": ["你好，我是EmoCompanion。",
             "今天的风很轻，适合慢慢地说一会儿话。",
             "你最近，过得好吗？"],
    "SpeakerA": ["嘿，我是SpeakerA。",
             "别看我平时闹得欢，认真说话的时候，我也能安静下来。",
             "要不要，一起待一会儿？"],
    "洛天依": ["大家好，我是洛天依。",
               "谢谢你愿意听我说话。",
               "希望今天的声音，能让你心里暖一点。"],
}
同句 = "语言不只是声音，它把一个人的温度，送到另一个人的心里。"


def 读档案(名: str) -> 声线档案:
    p = os.path.join(产物档案, 名 + ".json")
    if not os.path.exists(p):
        raise SystemExit("缺少档案 %s（先跑 拟合_44k.py）" % p)
    with open(p, encoding="utf-8") as f:
        return 声线档案.从dict(json.load(f))


def 读EQ(名: str):
    p = os.path.join(产物档案, 名 + ".eq44.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return np.asarray(json.load(f)["增益dB"], dtype=np.float64)


def 静音(秒: float) -> np.ndarray:
    return np.zeros(int(round(秒 * FS)), dtype=np.float64)


def 拼(段s, 间隔=0.35) -> np.ndarray:
    out = []
    for i, w in enumerate(段s):
        out.append(np.asarray(w, dtype=np.float64))
        if i != len(段s) - 1:
            out.append(静音(间隔))
    return np.concatenate(out) if out else np.zeros(0)


def 响度(x: np.ndarray, 目标rms=0.06, 峰上限=0.92) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = x * (目标rms / max(float(np.sqrt(np.mean(x * x))), 1e-9))
    pk = float(np.max(np.abs(y))) if y.size else 0.0
    if pk > 峰上限:
        y = y * (峰上限 / pk)
    return y


def 写wav(路径: str, x: np.ndarray) -> str:
    os.makedirs(os.path.dirname(路径), exist_ok=True)
    y = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    wavfile.write(路径, FS, (y * 32767.0).round().astype("<i2"))
    return 路径


def 参考片段(名: str, 秒: float = 9.0) -> np.ndarray:
    目录 = os.path.join(MEET60, "素材", 名)
    ps = sorted(glob.glob(os.path.join(目录, "ref44_*.wav")))
    if not ps:
        return np.zeros(0)
    xs = []
    for p in ps[:2]:
        sr, x = wavfile.read(p)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if np.max(np.abs(x)) > 1.2:
            x = x / 32768.0
        xs.append(np.concatenate([x, 静音(0.2)]))
    y = np.concatenate(xs)[:int(秒 * FS)]
    return 响度(y, 0.06)


def 渲染角色(档案: 声线档案, 名: str, 模式: str, EQ=None, 语速倍=1.0) -> np.ndarray:
    """模式: '裸声' / '韵律'；EQ 为 None 或 7 段增益。"""
    ws = []
    for i, 句 in enumerate(角色文本[名]):
        种子i = 种子[名] + i * 11
        if 模式 == "裸声":
            x, _, _ = H44.裸声44(档案, 种子i).说(句)
        else:
            x, _, _ = H44.人类韵律44(档案, 种子i).说(句, EQ=EQ)
        ws.append(x)
    return 响度(拼(ws, 0.40), 0.06)


def 组目录(名: str) -> str:
    p = os.path.join(产物组, 名)
    os.makedirs(p, exist_ok=True)
    return p


def 构建() -> dict:
    档案 = {名: 读档案(名) for 名 in 顺序}
    EQ = {名: 读EQ(名) for 名 in 顺序}
    清单 = {"口径": "参数化模仿（非克隆）；44.1k 原生引擎；无 TTS/RVC/外部模型", "组": {}}
    角色音 = {}

    def 存组(组名: str, 说明: str, 音s: dict, 主文件=True):
        d = 组目录(组名)
        for 名, x in 音s.items():
            写wav(os.path.join(d, "角色_%s.wav" % 名), x)
        if 主文件:
            主 = 响度(拼([音s[名] for 名 in 顺序], 0.65), 0.06)
            写wav(os.path.join(产物组, "%s_三角色.wav" % 组名), 主)
        清单["组"][组名] = dict(说明=说明, 角色=list(音s), 主文件=("%s_三角色.wav" % 组名))
        print("[组]", 组名, 说明)

    # 组B：裸声
    b = {名: 渲染角色(档案[名], 名, "裸声") for 名 in 顺序}
    存组("组B_新引擎裸声", "44.1k 新引擎，无人类韵律/无EQ", b)

    # 组C：韵律无EQ
    c = {名: 渲染角色(档案[名], 名, "韵律") for 名 in 顺序}
    存组("组C_新引擎+韵律", "44.1k 新引擎 + 人类化韵律，无EQ", c)

    # 组D：韵律 + 轻度低频校正（主修复版）
    #     只调 250–2000Hz 三个低带、限 ±2dB；>2k 一律不动（保住 v3 的高频清晰度，避免 v2 的闷）
    轻EQ = {}
    for 名 in 顺序:
        g = None if EQ[名] is None else np.asarray(EQ[名], dtype=np.float64).copy()
        if g is not None:
            g = np.clip(g, -2.0, 2.0)
            g[3:] = 0.0
        轻EQ[名] = g
    dd = {名: 渲染角色(档案[名], 名, "韵律", 轻EQ[名]) for 名 in 顺序}
    存组("组D_新引擎+韵律+轻EQ", "主修复版：44.1k 新引擎 + 人类韵律 + 仅 250–2000Hz ≤±2dB（>2k 不动）", dd)
    角色音 = dd

    # 组E：音高 ×0.85
    e = {}
    for 名 in 顺序:
        档 = 档案[名].派生(基频=float(np.clip(档案[名].基频 * 0.85, 70.0, 400.0)))
        e[名] = 渲染角色(档, 名, "韵律", 轻EQ[名])
    存组("组E_自然音高", "组D 但音高 ×0.85（-2.8st，说话音高变体）", e)

    # 组F：同句三声（用组D）
    f = {}
    同句段 = []
    for 名 in 顺序:
        x, _, _ = H44.人类韵律44(档案[名], 种子[名] + 97).说(同句, EQ=轻EQ[名])
        f[名] = 响度(x, 0.06)
        同句段.append(f[名])
    存组("组F_同句三声", "组D 的同一句三声", f, 主文件=False)
    混 = 响度(拼(同句段, 0.65), 0.06)
    写wav(os.path.join(产物组, "组F_同句三声.wav"), 混)
    清单["组"]["组F_同句三声"]["主文件"] = "组F_同句三声.wav"

    # 组G：参考 ↔ 组D
    g段 = []
    for 名 in 顺序:
        r = 参考片段(名, 9.0)
        if r.size:
            g段.extend([r, 静音(0.45), 角色音[名][:int(9.5 * FS)], 静音(0.9)])
    if g段:
        写wav(os.path.join(产物组, "组G_参考A_B.wav"), 响度(np.concatenate(g段), 0.06))
        清单["组"]["组G_参考A_B"] = dict(说明="每角色：44.1k 参考片段 → 组D 合成片段（含参考原声）",
                                      主文件="组G_参考A_B.wav")

    with open(os.path.join(产物组, "组清单.json"), "w", encoding="utf-8") as fh:
        json.dump(清单, fh, ensure_ascii=False, indent=1)
    return 清单


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(构建(), ensure_ascii=False, indent=1))
