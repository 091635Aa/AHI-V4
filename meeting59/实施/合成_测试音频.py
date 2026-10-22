# -*- coding: utf-8 -*-
"""meeting59/实施/合成_测试音频.py —— 三角色测试音频总装（原生声带 + 人类化韵律 + 谱包络校正）。

输入：`产物/声线档案/{EmoCompanion,SpeakerA,洛天依}.json` 与同名 `.eq.json`（由 拟合_参考声线.py 从参考录音拟合）
输出（默认 `产物/测试音频/`，可用 `--输出 目录` 改到别处，写者自建）：
  · `测试音频_三角色.wav`  —— 主交付：EmoCompanion / SpeakerA / 洛天依 各说 3 句，再同句三声
  · `角色_<名>.wav`        —— 单角色
  · `同句三声.wav`         —— 同一句文本、三条声线（可比对）
  · `对照_参考vs合成.wav`  —— 参考片段 ↔ 同角色合成（A/B 对照；**含参考原声，不是纯合成**）
  · `清单.json` / `<输出>/指标/合成指标.json`

音高对中：拟合用探针句，交付用角色台词，两者声调分布不同 ⇒ 交付前按「交付台词实测 F0 中位
= 参考目标 F0」对 基频 做 1–3 轮校正（并保存最终档案）。目标 F0 取 `产物/指标/拟合_<名>.json`
的 `F0目标`（EmoCompanion/SpeakerA=参考全集中位；洛天依=说话组中位，清唱不参与音高）。

口径：参数化模仿（从参考录音估参数），不是真克隆；全部**合成**声音由原生声带 + 纯 numpy
7 段静态谱包络校正产生；参考音频仅本机用于估参、不再分发。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET59 = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import 声学工具 as T  # noqa: E402
import 韵律_人类 as 人类  # noqa: E402
import 谱包络_校正 as EQ模块  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

产物音频 = os.path.join(MEET59, "产物", "测试音频")
产物指标 = os.path.join(MEET59, "产物", "指标")
产物档案 = os.path.join(MEET59, "产物", "声线档案")

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
        raise SystemExit("缺少声线档案：%s（先跑 拟合_参考声线.py）" % p)
    with open(p, encoding="utf-8") as f:
        return 声线档案.从dict(json.load(f))


def 读目标F0(名: str) -> float | None:
    p = os.path.join(产物指标, "拟合_%s.json" % 名)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return float(d.get("F0目标")) if d.get("F0目标") else None


def 静音(秒: float) -> np.ndarray:
    return np.zeros(int(round(秒 * T.FS)), dtype=np.float64)


def 拼(段s, 间隔=0.35) -> np.ndarray:
    out = []
    for i, w in enumerate(段s):
        out.append(np.asarray(w, dtype=np.float64))
        if i != len(段s) - 1:
            out.append(静音(间隔))
    return np.concatenate(out) if out else np.zeros(0)


def 均一响度(x: np.ndarray, 目标rms: float = 0.055, 峰上限: float = 0.92) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    r = T.均方根(x)
    y = x * (目标rms / max(r, 1e-9))
    pk = float(np.max(np.abs(y))) if y.size else 0.0
    if pk > 峰上限:
        y = y * (峰上限 / pk)
    return y


def 中位F0(x: np.ndarray) -> float:
    f0, _, _ = T.基频轨迹(x)
    v = f0[np.isfinite(f0)]
    return float(np.median(v)) if v.size else 0.0


def 校准音高(档案: 声线档案, 句s, 目标F0: float | None, 种子基: int, EQ,
             上限轮: int = 3) -> tuple:
    """按交付台词「帧 F0 真值」中位对中目标（参考集 F0 中位）。返回 (档案, 波形, 交付帧F0)。"""
    w = np.zeros(0)
    帧F0 = 0.0
    for _ in range(上限轮):
        ws, fs = [], []
        for i, 句 in enumerate(句s):
            x, _, 帧 = 人类.说(句, 档案, 种子基 + i * 11, EQ=EQ)
            ws.append(x)
            fs.extend(帧["f0"])
        w = 拼(ws, 0.40)
        帧F0 = float(np.median(fs)) if fs else 0.0
        if 目标F0 is None or not np.isfinite(帧F0) or 帧F0 <= 1.0:
            break
        diff = float(12.0 * np.log2(目标F0 / 帧F0))
        if abs(diff) < 0.05:
            break
        档案 = 档案.派生(基频=float(np.clip(档案.基频 * (2.0 ** (diff / 12.0)), 70.0, 400.0)))
    return 档案, w, 帧F0


def 段指标(x: np.ndarray) -> dict:
    f0, hnr, 能 = T.基频轨迹(x)
    v = f0[np.isfinite(f0)]
    n = len(x) // 160
    env = np.array([T.均方根(x[i * 160:(i + 1) * 160]) for i in range(n)]) if n else np.zeros(0)
    停顿 = 0
    if n:
        门 = 0.10 * float(np.percentile(env, 90))
        低 = env < 门
        连 = 0
        for b in 低:
            连 = 连 + 1 if b else 0
            if 连 == 12:
                停顿 += 1
    return dict(
        时长秒=round(len(x) / T.FS, 3),
        均方根=round(T.均方根(x), 4),
        峰值=round(float(np.max(np.abs(x))) if x.size else 0.0, 3),
        F0中位=round(float(np.median(v)) if v.size else 0.0, 1),
        F0跨度半音=round(float(np.log2(np.percentile(v, 95) / np.percentile(v, 5)) * 12.0)
                     if v.size >= 10 else 0.0, 2),
        浊音比=round(T.浊音比(x), 3),
        停顿数=停顿,
        共振峰=[round(float(z), 1) for z in T.共振峰中位(x)],
        带包络=[round(float(z), 3) for z in T.带包络(x)],
    )


def 参考片段(名: str, 秒: float = 9.0) -> np.ndarray:
    目录 = os.path.join(MEET59, "素材", 名)
    ps = sorted(glob.glob(os.path.join(目录, "ref_*.wav")))
    if 名 == "洛天依":
        ps = [p for p in ps if "清唱" in os.path.basename(p)] or ps
    if not ps:
        return np.zeros(0)
    xs = [T.读wav16(p) for p in ps[:2]]
    x = np.concatenate([np.concatenate([v, 静音(0.2)]) for v in xs])
    return 均一响度(x[:int(秒 * T.FS)], 0.055)


def 构建(输出目录: str, 写: bool = True, 保存档案: bool | None = None) -> dict:
    os.makedirs(输出目录, exist_ok=True)
    默认目录 = os.path.abspath(输出目录) == os.path.abspath(产物音频)
    if 保存档案 is None:
        保存档案 = bool(写 and 默认目录)
    指标目录 = 产物指标 if 默认目录 else os.path.join(输出目录, "指标")
    os.makedirs(指标目录, exist_ok=True)
    档案, EQ, 目标F0 = {}, {}, {}
    for 名 in 顺序:
        档案[名] = 读档案(名)
        EQ[名] = EQ模块.取(os.path.join(产物档案, 名 + ".eq.json"))
        目标F0[名] = 读目标F0(名)
    清单 = {"口径": "参数化模仿（从参考录音估参数），不是真克隆；全部合成声音=原生声带+7段静态谱包络校正",
            "种子": 种子, "角色": {}, "声线档案": {}}
    角色音, 指标 = {}, {"角色": {}, "同句": {}, "总装": {}}
    for 名 in 顺序:
        档, w0, f0交付 = 校准音高(档案[名], 角色文本[名], 目标F0[名], 种子[名], EQ[名])
        档案[名] = 档
        # 保存对中后的最终档案（原始拟合档案保留在 产物/指标/拟合_<名>.json 的「声线档案」里）
        档 = 档.派生(备注=(档.备注 + "｜交付音高对中 F0=%.1fHz" % f0交付))
        档案[名] = 档
        if 写 and 保存档案:
            p = os.path.join(产物档案, 名 + ".json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(dict(档.转dict(), 指纹=档.指纹()), f, ensure_ascii=False, indent=1)
        w = 均一响度(w0, 0.055)
        角色音[名] = w
        p角色 = os.path.join(输出目录, "角色_%s.wav" % 名)
        if 写:
            T.写wav16(p角色, w)
        指标["角色"][名] = dict(段指标(w), 档案指纹=档.指纹(),
                              拟合目标F0=(round(目标F0[名], 2) if 目标F0[名] else None),
                              交付F0=round(f0交付, 2),
                              wav=os.path.basename(p角色))
        清单["角色"][名] = dict(文件=os.path.basename(p角色),
                              档案指纹=档.指纹(), 来源=档.来源,
                              拟合目标F0=(round(目标F0[名], 2) if 目标F0[名] else None),
                              交付F0=round(f0交付, 2), 备注=档.备注)
        清单["声线档案"][名] = dict(档.转dict(), 指纹=档.指纹(),
                                 EQ增益dB=(None if EQ[名] is None else [round(float(v), 2) for v in EQ[名]]))
        print("[角色] %-4s 时长=%.2fs 目标F0=%s 交付F0=%.1f 停顿=%d"
              % (名, len(w) / T.FS, 目标F0[名], f0交付, 指标["角色"][名]["停顿数"]))
    # 主交付：三角色 + 同句三声
    主段 = []
    for 名 in 顺序:
        主段.append(角色音[名])
        主段.append(静音(0.75))
    同句音 = {}
    for 名 in 顺序:
        w, _, _ = 人类.说(同句, 档案[名], 种子[名] + 97, EQ=EQ[名])
        同句音[名] = 均一响度(w, 0.055)
        指标["同句"][名] = dict(段指标(同句音[名]), 档案指纹=档案[名].指纹())
    同句混 = 拼([同句音[名] for 名 in 顺序], 0.65)
    p同 = os.path.join(输出目录, "同句三声.wav")
    主 = 均一响度(拼(主段 + [同句混]), 0.055, 0.90)
    p主 = os.path.join(输出目录, "测试音频_三角色.wav")
    if 写:
        T.写wav16(p同, 同句混)
        T.写wav16(p主, 主)
    指标["同句"]["句"] = 同句
    指标["同句"]["wav"] = os.path.basename(p同)
    指标["总装"] = dict(主文件=os.path.basename(p主), 总时长秒=round(len(主) / T.FS, 2),
                       均方根=round(T.均方根(主), 4))
    # 对照：参考片段 ↔ 同角色合成（含参考原声，非纯合成）
    对照段 = []
    for 名 in 顺序:
        r = 参考片段(名, 9.0)
        if r.size == 0:
            continue
        对照段.extend([r, 静音(0.45), 角色音[名][:int(9.5 * T.FS)], 静音(0.9)])
    if 对照段:
        p对 = os.path.join(输出目录, "对照_参考vs合成.wav")
        if 写:
            T.写wav16(p对, 均一响度(np.concatenate(对照段), 0.055, 0.90))
        指标["总装"]["对照"] = os.path.basename(p对)
    print("[主交付] %s 时长=%.2fs" % (p主, len(主) / T.FS))
    if 写:
        with open(os.path.join(输出目录, "清单.json"), "w", encoding="utf-8") as f:
            json.dump(清单, f, ensure_ascii=False, indent=1)
        with open(os.path.join(指标目录, "合成指标.json"), "w", encoding="utf-8") as f:
            json.dump(指标, f, ensure_ascii=False, indent=1)
    return dict(主=主, 指标=指标, 清单=清单, 输出目录=输出目录)


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    输出 = 产物音频
    for i, a in enumerate(argv):
        if a == "--输出" and i + 1 < len(argv):
            输出 = os.path.abspath(argv[i + 1])
    os.makedirs(输出, exist_ok=True)
    构建(输出, 写=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
