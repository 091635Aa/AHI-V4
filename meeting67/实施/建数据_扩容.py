# -*- coding: utf-8 -*-
"""meeting67/实施/建数据_扩容.py —— 新片段 → 与既有 npz 同格式的训练数据（C14：只 import meeting66 函数，不改既有件）。

对齐：
  1) 优先用 ASR token 时间戳（`标注_变长片段.py` 的 tokens，单位 ms）→ 每音节帧数；
  2) 时间戳不可用/与汉字序列不匹配 → 退回 `数据_SpeakerA.分配帧数`（VAD 段内均分），并记录 weak 口径；
  3) 词表（声母/韵母/调 id）沿用既有 `meeting66/产物/数据/meta.json` 的 vocab，保证合并训练 id 一致。

输入：标注.jsonl（每个片段一条）
输出：样本/xxxxx.npz + meta_扩容.json + 异常.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

M66 = Path(r"<workspace>/03_AHI\AHI4\meeting66")
既有META = M66 / "产物" / "数据" / "meta.json"


def 载模块():
    p = M66 / "实施" / "数据_SpeakerA.py"
    s = importlib.util.spec_from_file_location("数据_SpeakerA_只读", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def 主(argv=None):
    ap = argparse.ArgumentParser(description="新片段 → npz（同既有格式）")
    ap.add_argument("--标注", required=True)
    ap.add_argument("--出", required=True)
    参 = ap.parse_args(argv)
    出 = Path(参.出); (出 / "样本").mkdir(parents=True, exist_ok=True)
    数据 = 载模块()
    音韵 = 数据._载入音韵()
    meta既有 = json.loads(既有META.read_text(encoding="utf-8"))
    v = meta既有["vocab"]
    声母表, 韵母表, 调表 = v["声母"], v["韵母"], v["调"]
    声id = {s: i for i, s in enumerate(声母表)}
    韵id = {y: i for i, y in enumerate(韵母表)}
    调id = {t: i for i, t in enumerate(调表)}
    rows = [json.loads(x) for x in Path(参.标注).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("文本")]
    记录s, 异常 = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        文 = r["文本"]
        汉字 = 数据.取汉字(文)
        音节s, cn, 去数 = 数据.解析文本(文, 音韵)
        if not 音节s:
            异常.append({"文件": r.get("文件名"), "原因": "无音节"})
            continue
        # 二次筛（15 §9 第3条）：字数密度门（先于 npz 构建）
        时长 = float(r.get("时长秒") or 0.0)
        密 = (len(汉字) / 时长) if 时长 > 0 else 0.0
        if len(汉字) < 8 or not (1.5 <= 密 <= 8.0):
            异常.append({"文件": r.get("文件名"), "原因": f"字数密度淘汰(汉字{len(汉字)},密度{密:.2f})"})
            continue
        # 词表覆盖检查
        if any((s["声"] or "∅") not in 声id or s["韵"] not in 韵id or s["调"] not in 调id for s in 音节s):
            异常.append({"文件": r.get("文件名"), "原因": "词表未覆盖"})
            continue
        try:
            x, 源sr = 数据.读音频(Path(r["文件"]))
            起, 止, 异2 = 数据.找语音段(x)
            x段 = np.ascontiguousarray(x[起:止])
            mel = 数据.计算logmel(x段)
            帧数 = int(mel.shape[0])
            if 帧数 < len(音节s):
                异常.append({"文件": r.get("文件名"), "原因": f"帧数({帧数})<音节数({len(音节s)})"})
                continue
            对齐法 = "时间戳"
            帧s = None
            toks = r.get("tokens") or []
            # 从 tokens 取汉字 token（仅汉字，去空格/英文/数字）
            词s = 文.split()
            # ASR token 与文本词一一对应 → 逐汉字时间（多字 token 均分其时长；非汉字 token 丢弃）
            字时 = []
            if len(词s) == len(toks):
                for 词, tk in zip(词s, toks):
                    try:
                        a, b = int(tk[0]), int(tk[1])
                    except Exception:
                        continue
                    cs = [ch for ch in 词 if 数据.是汉字(ch)]
                    if not cs:
                        continue
                    L = len(cs)
                    for j, ch in enumerate(cs):
                        a2 = a + (b - a) * j / L
                        b2 = a + (b - a) * (j + 1) / L
                        字时.append((ch, int(a2), int(b2)))
            if 字时:
                import difflib
                我们 = list(汉字)
                matcher = difflib.SequenceMatcher(None, 我们, [c for c, _, _ in 字时], autojunk=False)
                匹配 = dict()
                for blk in matcher.get_matching_blocks():
                    for k2 in range(blk.size):
                        匹配[blk.a + k2] = (字时[blk.b + k2][1], 字时[blk.b + k2][2])
                if 匹配:
                    def _时(i2):
                        if i2 in 匹配:
                            return 匹配[i2]
                        left = [j for j in 匹配 if j < i2]
                        right = [j for j in 匹配 if j > i2]
                        if left and right:
                            l = max(left); r = min(right)
                            t = (i2 - l) / (r - l)
                            return (匹配[l][0] + (匹配[r][0] - 匹配[l][0]) * t,
                                    匹配[l][1] + (匹配[r][1] - 匹配[l][1]) * t)
                        if left:
                            return 匹配[max(left)]
                        return 匹配[min(right)]
                    起帧偏 = int(round(起 * 1000.0 / 数据.SR))
                    帧s = []
                    for i2 in range(len(我们)):
                        a, b = _时(i2)
                        f0 = max(0, int(round(a / 10.0)) - 起帧偏)
                        f1 = max(f0 + 1, int(round(b / 10.0)) - 起帧偏)
                        帧s.append(max(1, f1 - f0))
                    总 = sum(帧s)
                    if 总 > 0 and 总 != 帧数:
                        缩放 = 帧数 / 总
                        帧s = [max(1, int(round(q * 缩放))) for q in 帧s]
                        差 = 帧数 - sum(帧s)
                        k = len(帧s) - 1
                        while 差 != 0 and k >= 0:
                            if 差 > 0:
                                帧s[k] += 1; 差 -= 1
                            elif 帧s[k] > 1:
                                帧s[k] -= 1; 差 += 1
                            k = (k - 1) % len(帧s)
            if 帧s is None:
                帧s, 退回 = 数据.分配帧数(帧数, len(音节s))
                对齐法 = "均分(weak)" if not 退回 else "均分(floor兜底)"
            声帧s, 韵帧s = [], []
            for 帧 in 帧s:
                声帧, 韵帧 = 数据.声韵帧数(帧)
                声帧s.append(声帧); 韵帧s.append(韵帧)
            数组s = dict(
                mel=mel,
                声母=np.array([声id[s["声"] or "∅"] for s in 音节s], dtype=np.int32),
                韵母=np.array([韵id[s["韵"]] for s in 音节s], dtype=np.int32),
                调=np.array([调id[s["调"]] for s in 音节s], dtype=np.int32),
                音节帧数=np.array(帧s, dtype=np.int32),
                声母帧数=np.array(声帧s, dtype=np.int32),
                韵母帧数=np.array(韵帧s, dtype=np.int32),
            )
            名 = f"{i:04d}.npz"
            数据.写npz(出 / "样本" / 名, 数组s)
            记录s.append(dict(
                id=i, 文件=名, wav=r.get("文件名"), 源= r["文件"],
                时长秒=r.get("时长秒"), 对齐=对齐法,
                源采样率=源sr, 帧数=帧数, 音节数=len(音节s),
                文本=文, 汉字=汉字, 非汉字字数=去数,
                数组sha256=数据.数组sha(数组s), 异常=异2,
            ))
        except Exception as e:  # noqa: BLE001
            异常.append({"文件": r.get("文件名"), "原因": f"{type(e).__name__}: {e}"})
        if (i + 1) % 50 == 0:
            print(f"[建数据] {i+1}/{len(rows)} 条，已用 {time.time()-t0:.0f}s")
    # 词表来自既有 meta；新数据 id 与既有完全一致
    meta = dict(版本="数据_扩容 v1", 来源=dict(标注=str(Path(参.标注)), 既有meta=str(既有META)),
                vocab=dict(声母=声母表, 韵母=韵母表, 调=调表),
                记录=记录s, 异常=异常,
                统计=dict(条数=len(记录s), 总帧=int(sum(x["帧数"] for x in 记录s)),
                        总时长秒=round(float(sum(x["时长秒"] or 0 for x in 记录s)), 1),
                        对齐法={k: sum(1 for x in 记录s if x["对齐"] == k) for k in
                              sorted({x["对齐"] for x in 记录s})}),
                时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    (出 / "meta_扩容.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (出 / "异常.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in 异常) + ("\n" if 异常 else ""),
        encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("版本", "统计")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
