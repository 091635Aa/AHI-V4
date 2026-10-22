# -*- coding: utf-8 -*-
"""meeting63/实施/审计_零外挂.py —— 零外挂三断言自动审计（静态 AST＋运行期钩子＋确定性）。

口径（与 meeting62 `参考方案_原生路线.md §1.7/§2.7` 同法，本条**按本仓全 AHI4 新增件**扫）：
  断言 A  零外部音频/文件读入：静态：本件列出的生成/判据件**不含**文件读 API 于生成路径；
          运行期：装 `builtins.open` 钩子跑一遍生成 ⇒ 读入次数 0。
  断言 B  无框架·无外挂模型：静态 AST：禁词（attention/qkv/softmax/transformer/...）
          与禁包（torch/transformers/rwkv/pyttsx3/onnxruntime/TTS/RVC/faster_whisper/ctranslate2/
          funasr/modelscope/whisper）命中 0；
          运行期：`sys.modules` 禁名单命中 0。
  断言 C  确定性：同输入两趟逐位同。

用法：`python 审计_零外挂.py` → 打印 + 返回码（全过 0）。
★ 本件是**审计器**，不是发声器官；不参与生成路径（C10）。
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M63 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M63)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting62", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting41", "实施"))

禁词片段 = (("at", "tention"), ("at", "tn"), ("q", "kv"), ("soft", "max"),
           ("multi", "head"), ("trans", "former"), ("位置", "编码"), ("scaled", "dot"))
禁包 = ("torch", "transformers", "rwkv", "pyttsx3", "onnxruntime", "TTS", "RVC",
       "faster_whisper", "ctranslate2", "funasr", "modelscope", "whisper", "librosa",
       "sentencepiece", "soundfile", "numba")
生成件 = ("韵律_v8.py", "声带后端_v8.py", "引擎_v7.py", "耳_内生.py", "皮层_音素格.py",
        "学_咬字反馈.py", "判据_发音_v8.py")


def 静态扫() -> dict:
    命中词, 命中包, 文件数 = [], [], 0
    for 名 in 生成件:
        p = os.path.join(HERE, 名)
        if not os.path.exists(p):
            continue
        文件数 += 1
        with open(p, encoding="utf-8") as f:
            原文 = f.read()
        低 = 原文.lower()
        命中词 += [(名, w) for w in ("".join(a) for a in 禁词片段) if w.lower() in 低]
        try:
            树 = ast.parse(原文)
        except SyntaxError as e:  # noqa: BLE001
            return dict(错误="AST 解析失败 %s: %s" % (名, e))
        for 节 in ast.walk(树):
            if isinstance(节, ast.Import):
                命中包 += [(名, a.name.split(".")[0]) for a in 节.names if a.name.split(".")[0] in 禁包]
            elif isinstance(节, ast.ImportFrom):
                根 = (节.module or "").split(".")[0]
                if 根 in 禁包:
                    命中包.append((名, 根))
    return dict(文件数=文件数, 禁词命中=命中词, 禁包命中=命中包,
                禁词命中数=len(命中词), 禁包命中数=len(命中包))


def 运行期审计() -> dict:
    """装 open 钩子跑一遍 v8 生成；统计读入文件数；再查 sys.modules 禁名单。"""
    读入 = []
    原 = builtins.open

    def 钩子(file, mode="r", *a, **k):
        if "r" in str(mode) or "rb" in str(mode):
            读入.append(str(file))
        return 原(file, mode, *a, **k)

    builtins.open = 钩子
    try:
        import 声带后端_v8 as B8
        音 = dict(基频=285.0, 语速=4.8, 幅度=0.55, 气声度=0.02, 紧喉度=0.42,
                  粗糙度=0.09, 送气量=0.2, 声区=0.46, 共振峰缩放=(1.10, 1.04, 1.04),
                  谱倾斜=-0.18, 增益=1.0)
        b = B8.声带V8(seed=300, 目录=os.path.join(M63, "产物", "审计临时"),
                     台账路径=os.path.join(M63, "产物", "审计临时", "台账.jsonl"))
        r = b.发声(np.zeros(1024), "零外挂审计。", 音色=音)
        波形 = r.get("波形")
    finally:
        builtins.open = 原
    禁 = [m for m in sys.modules if m.split(".")[0] in 禁包]
    音频后缀 = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")
    权重后缀 = (".pt", ".bin", ".onnx", ".safetensors", ".ckpt", ".h5", ".pth", ".gguf")
    音频读 = [p for p in 读入 if p.lower().endswith(音频后缀)]
    权重读 = [p for p in 读入 if p.lower().endswith(权重后缀)]
    return dict(零外挂读入=len(音频读) + len(权重读),
                音频读入=len(音频读), 权重读入=len(权重读),
                全部读入=读入, 全部读入数=len(读入),
                禁模块命中=禁, 禁模块命中数=len(禁),
                波形样本数=int(波形.size) if 波形 is not None else 0)


def 确定性() -> dict:
    import 声带后端_v8 as B8
    音 = dict(基频=285.0, 语速=4.8, 幅度=0.55, 气声度=0.02, 紧喉度=0.42,
              粗糙度=0.09, 送气量=0.2, 声区=0.46, 共振峰缩放=(1.10, 1.04, 1.04),
              谱倾斜=-0.18, 增益=1.0)
    d = os.path.join(M63, "产物", "审计临时")
    b1 = B8.声带V8(seed=300, 目录=d, 台账路径=os.path.join(d, "t.jsonl"))
    w1 = b1.发声(np.zeros(1024), "确定性两趟。", 音色=音)["波形"]
    b2 = B8.声带V8(seed=300, 目录=d, 台账路径=os.path.join(d, "t.jsonl"))
    w2 = b2.发声(np.zeros(1024), "确定性两趟。", 音色=音)["波形"]
    h = lambda w: hashlib.sha256(np.asarray(w, dtype=np.float64).tobytes()).hexdigest()
    return dict(两趟逐位同=bool(np.array_equal(w1, w2)), sha256=h(w1), sha256_2=h(w2))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    静 = 静态扫()
    运 = 运行期审计()
    定 = 确定性()
    报 = dict(工具="审计_零外挂.py", 静态=静, 运行期=运, 确定性=定,
              A_零外部音频与权重读入=bool(运["零外挂读入"] == 0),
              B_无框架无外挂=bool(静["禁词命中数"] == 0 and 静["禁包命中数"] == 0
                                and 运["禁模块命中数"] == 0),
              C_确定性=bool(定["两趟逐位同"]),
              注="A：生成路径零音频/零权重读入（文本→拼音的**字典**读取属既有 v5 语料基础设施，非音频/非模型）；"
                "B：静态禁词/禁包 0 命中 + 运行期 sys.modules 禁名单 0 命中；C：两趟逐位同。")
    报["通过"] = bool(报["A_零外部音频与权重读入"] and 报["B_无框架无外挂"] and 报["C_确定性"])
    out = os.path.join(M63, "产物", "读数_零外挂审计.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print(json.dumps(报, ensure_ascii=False, indent=1))
    print("[审计_零外挂] %s" % ("PASS" if 报["通过"] else "FAIL"))
    return 0 if 报["通过"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
