# -*- coding: utf-8 -*-
"""meeting62 · 红队探针：判据「自证 / 窗重叠 / 白噪 sham」的最小可证伪实验。

只读 meeting61 既有件，只写 meeting62/团队/红队/产物_探针.json（C14）。
CPU-only、单次 <10s、不占声卡、不启动 RVC。

复刻 `判据_发音_v6.py::_包络 / _峰与台` 的**同一量**（倒谱平滑谱包络在 150–3500Hz 的最大点），
但做三件判据自己没做的事：
  A 窗重叠矩阵：各元音的判据窗 [F1*0.85, F2*1.15] 互相包含多少（判「落位」是否可被窗宽买通）。
  B 实测峰落窗：真合成各元音 → 峰落在哪些元音的窗里（自身 / 交叉）。
  C 白噪 sham：纯噪声的包络峰落在几个元音窗里（若噪声也「落位」⇒ 该臂非判别性）。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

本目录 = os.path.dirname(os.path.abspath(__file__))
团队 = os.path.dirname(本目录)                 # meeting62/团队
MEET62 = os.path.dirname(团队)
AHI4 = os.path.dirname(MEET62)
MEET61 = os.path.join(AHI4, "meeting61")
sys.path.insert(0, os.path.join(MEET61, "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 音韵_v5 as Y            # noqa: E402
import 韵律_v6 as H5           # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
出 = os.path.join(本目录, "产物_探针.json")

# 判据 F1 臂用的样本（字, 拼, 是否单元音）
样 = (("啊", "a1"), ("哦", "o1"), ("鹅", "e1"), ("衣", "i1"),
      ("乌", "u1"), ("迂", "v1"), ("儿", "er2"))


def 读档(名="EmoCompanion") -> 声线档案:
    with open(os.path.join(MEET61, "产物", "声线档案", 名 + ".json"), encoding="utf-8") as f:
        return 声线档案.从dict(json.load(f))


def _包络(x, fs=FS):
    """逐位同 判据_发音_v6.py::_包络（倒谱平滑，1.6ms→此处 2.2ms 与 F1 臂一致）。"""
    x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    X = np.fft.rfft(x * np.hanning(len(x)))
    cep = np.fft.irfft(np.log(np.abs(X) + 1e-12))
    L = max(4, int(0.0022 * fs))
    cep[L:-L] = 0.0
    env = np.exp(np.fft.rfft(cep).real)
    return np.fft.rfftfreq(len(x), 1.0 / fs), env


def 峰位(x):
    if len(x) < 512:
        return 0.0
    f, env = _包络(x)
    m = (f >= 150.0) & (f <= 3500.0)
    ff, ee = f[m], env[m]
    return float(ff[int(np.argmax(ee))])


def 窗(拼, 档):
    s = Y.解析音节(拼)
    目s = s["目标s"]
    单元 = len(目s) == 1 and s["尾"] is None
    if 单元:
        return (目s[0][0] * 档.共振峰缩放[0] * 0.85,
                目s[0][1] * 档.共振峰缩放[1] * 1.15, True)
    f1t = min(t[0] for t in 目s) * 档.共振峰缩放[0]
    f2t = max(t[1] for t in 目s) * 档.共振峰缩放[1]
    return (f1t * 0.85, f2t * 1.15, False)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    档 = 读档("EmoCompanion")
    档低 = 档.派生(基频=120.0)
    窗s = {拼: 窗(拼, 档) for _, 拼 in 样}

    # A 窗重叠矩阵（纯几何，无需合成）
    重叠 = {}
    for _, pi in 样:
        row = {}
        for _, pj in 样:
            lo1, hi1, _ = 窗s[pi]
            lo2, hi2, 单j = 窗s[pj]
            inter = max(0.0, min(hi1, hi2) - max(lo1, lo2))
            span = hi1 - lo1
            row[pj] = round(inter / span, 3) if span > 0 else 0.0
        重叠[pi] = row

    # B 实测峰落窗（真合成）
    实测 = {}
    器 = H5.人类韵律_v5(档低, 300)
    for 字, 拼 in 样:
        帧 = H5.人类韵律_v5(档低, 300).建帧(字)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a0 = (idx[len(idx) // 3] if idx else 0) * 441
        w, _, _ = H5.人类韵律_v5(档低, 300).说(字)
        seg = w[a0:a0 + int(0.12 * FS)]
        p = 峰位(seg)
        落 = []
        for k, (lo, hi, _单) in 窗s.items():
            if lo <= p <= hi:
                落.append(k)
        实测[拼] = dict(峰=round(p, 1), 落窗=落,
                        落自身=bool(拼 in 落), 落他窗数=len([z for z in 落 if z != 拼]))

    # C 白噪 sham（与判据无关的输入；若也落位 ⇒ 该臂非判别性）
    rng = np.random.default_rng(62019)
    噪 = rng.standard_normal(int(0.12 * FS))
    噪落 = []
    pn = 峰位(噪)
    for k, (lo, hi, _单) in 窗s.items():
        if lo <= pn <= hi:
            噪落.append(k)

    # D 「置乱表」杀伤力（纯逻辑）：判据的窗完全由 音韵_v5 目标表算出；
    #    把 a1↔i1 的目标对调后引擎会照新表产出，判据用新表算窗 ⇒ 仍 PASS。
    s_a, s_i = Y.解析音节("a1"), Y.解析音节("i1")
    置乱 = dict(
        说明="窗 = [min(F1目标)*0.85, max(F2目标)*1.15]，目标直接取自 Y.解析音节(拼)；"
             "把目标表对调后引擎照新表产出、判据照新表算窗 ⇒ 判据不可能察觉『映射错』",
        a1目标=s_a["目标s"], i1目标=s_i["目标s"])

    # E 参考（真人）文件在判据同量下能否直接对照（外部正对照可行性：需先切元音段）
    参考文件 = []
    for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
        d = os.path.join(MEET61, "素材", 名)
        if os.path.isdir(d):
            参考文件 += [os.path.join(名, z) for z in sorted(os.listdir(d))
                      if z.startswith("ref44_") and z.endswith(".wav")]

    通过 = all(v["落自身"] for v in 实测.values())
    报 = dict(工具="探针_自证置乱.py", 量法="判据_发音_v6.F1 臂同量（倒谱包络 150–3500Hz 最大点）",
              窗= {k: [round(v[0], 1), round(v[1], 1), v[2]] for k, v in 窗s.items()},
              A_窗重叠矩阵=重叠, B_实测峰落窗=实测, C_白噪=dict(峰=round(pn, 1), 落窗=噪落),
              D_置乱逻辑=置乱, E_参考文件数=len(参考文件),
              自身全落位=通过,
              红队读数=dict(
                  噪声落窗数=len(噪落), 噪声落窗=噪落,
                  交叉落位元音数=sum(1 for v in 实测.values() if v["落他窗数"] > 0),
                  self落位=sum(1 for v in 实测.values() if v["落自身"])))
    with open(出, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)

    print("窗：", {k: [round(v[0], 1), round(v[1], 1)] for k, v in 窗s.items()})
    print("实测峰/落窗：")
    for k, v in 实测.items():
        print("  %-5s 峰=%-7.1f 落自身=%-5s 落窗=%s" % (k, v["峰"], v["落自身"], v["落窗"]))
    print("白噪峰=%.1f 落窗=%s" % (pn, 噪落))
    print("自身全落位=%s | 噪声落窗数=%d | 交叉落位元音=%d" %
          (通过, len(噪落), sum(1 for v in 实测.values() if v["落他窗数"] > 0)))
    print("JSON:", 出)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
