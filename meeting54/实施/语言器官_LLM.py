# -*- coding: utf-8 -*-
"""meeting54/实施/语言器官_LLM.py —— 越宪实验：本地 Qwen2.5-1.5B-Instruct 当语言器官。

注意（越宪登记）：
  · 本文件是**语言器官**：接收「角色包 + AHI 状态摘要 + 场景」，生成连续自然语言；
  · 违反原 C10（外挂模型当器官）/C1（Transformer 架构）/C13（非定点、非两趟），
    按所有者 2026-10-18 授权登记为「越宪实验」，**不混入主线宣称**；
  · 必须记录：模型名、权重哈希、种子、温度、生成参数。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # AHI4
模型目录 = r"<workspace>/02_新一代架构_RWKV7\RWKE\models\Qwen2.5-1.5B-Instruct"
sys.path.insert(0, os.path.join(ROOT, "挂载"))
sys.path.insert(0, os.path.join(ROOT, "meeting53", "实施"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def 权重哈希(目录=模型目录, 上限=64 * 1024 * 1024):
    h = hashlib.sha256()
    for 名 in sorted(os.listdir(目录)):
        p = os.path.join(目录, 名)
        if not os.path.isfile(p):
            continue
        h.update(名.encode())
        with open(p, "rb") as f:
            h.update(f.read(上限))
    return h.hexdigest()[:24]


class 语言器官:
    def __init__(self, 模型路径: str = 模型目录, 设备: str = None):
        self.模型路径 = 模型路径
        self.设备 = 设备 or ("cuda" if torch.cuda.is_available() else "cpu")
        self.分词器 = AutoTokenizer.from_pretrained(模型路径)
        self.模型 = AutoModelForCausalLM.from_pretrained(
            模型路径, dtype=torch.bfloat16 if self.设备 == "cuda" else torch.float32,
            device_map=None).to(self.设备)
        self.模型.eval()
        self.权重码 = 权重哈希()
        self.台账 = dict(调用=0, 生成字=0, 墙钟秒=0.0, 设备=self.设备,
                         模型="Qwen2.5-1.5B-Instruct", 权重码=self.权重码)

    def 提示(self, 角色包: dict, 状态摘要: str, 场景: str = "") -> str:
        记忆 = "\n".join("- " + str(x.get("文本", "")) for x in (角色包.get("记忆条目列") or [])[:4])
        人设 = str(角色包.get("人设", ""))[:120]
        return (
            "你叫%s。人设：%s\n你记得：\n%s\n此刻：%s\n%s\n"
            "要求：第一人称，口语，只说 2 到 4 句；不要重复前一句；不要括号动作/表情/标签；不要解释你是 AI。"
            % (角色包.get("角色", "角色"), 人设, 记忆, 状态摘要, 场景)
        )

    def 提示对(self, 角色包: dict, 状态摘要: str, 场景: str = ""):
        sysm = "你是%s，一个用第一人称说话的角色。" % str(角色包.get("角色", "角色"))
        return [dict(role="system", content=sysm),
                dict(role="user", content=self.提示(角色包, 状态摘要, 场景))]

    @torch.no_grad()
    def 说(self, 角色包: dict, 状态摘要: str, 场景: str = "", 最多新字: int = 120,
           贪心: bool = False, 温度: float = 0.75, 种子: int = 53001) -> dict:
        t0 = time.time()
        消息 = self.提示对(角色包, 状态摘要, 场景)
        if hasattr(self.分词器, "apply_chat_template") and self.分词器.chat_template:
            提示 = self.分词器.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
        else:
            提示 = "\n".join(m["content"] for m in 消息) + "\n回答："
        输入 = self.分词器(提示, return_tensors="pt").to(self.设备)
        torch.manual_seed(int(种子))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(种子))
        参 = dict(max_new_tokens=int(最多新字), do_sample=not 贪心, num_beams=1,
                  pad_token_id=self.分词器.eos_token_id, repetition_penalty=1.15)
        if not 贪心:
            参.update(temperature=float(温度), top_p=0.9, top_k=50)
        out = self.模型.generate(**输入, **参)
        新 = out[0][输入["input_ids"].shape[1]:]
        文 = self.分词器.decode(新, skip_special_tokens=True).strip()
        文 = _收句(文)
        self.台账["调用"] += 1
        self.台账["生成字"] += len(文)
        self.台账["墙钟秒"] = round(self.台账["墙钟秒"] + time.time() - t0, 3)
        return dict(文=文, 提示=提示, 种子=int(种子), 贪心=bool(贪心), 温度=float(温度),
                    参数=dict(max_new_tokens=int(最多新字), repetition_penalty=1.15), 耗时秒=round(time.time() - t0, 3))


def _收句(文: str, 最少字: int = 30, 最多句: int = 4) -> str:
    """只留前 2–4 句；去掉括号动作/表情/标签/换行。"""
    import re
    文 = re.sub(r"[（(][^）)]{0,12}[）)]", "", 文)
    文 = re.sub(r"#\S+", "", 文)
    文 = 文.replace("\n", "").replace("**", "").strip()
    句 = []
    当前 = []
    for ch in 文:
        if ch in "。！？":
            当前.append(ch)
            句.append("".join(当前))
            当前 = []
            if len(句) >= 最多句 and sum(len(s) for s in 句) >= 最少字:
                break
        else:
            当前.append(ch)
    if 当前 and len(句) < 最多句:
        句.append("".join(当前).strip() + "。")
    文 = "".join(句).strip()
    if len(文) > 200:
        文 = 文[:200]
        if 文[-1] not in "。！？":
            文 = 文[:max(0, max(文.rfind("。"), 文.rfind("！"), 文.rfind("？")) + 1)] or (文 + "。")
    return 文


def 状态摘要(观测: int, 自我标量: float, 内心步数: int) -> str:
    return "看到第%d路；自我标量=%.2f；内心步数=%d" % (int(观测), float(自我标量), int(内心步数))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import 角色包 as 包模块  # noqa: E402
    器 = 语言器官()
    print("加载完成：", 器.台账["模型"], 器.台账["设备"], "权重码", 器.权重码)
    for 名, 包 in 包模块.全部包().items():
        r = 器.说(包, 状态摘要(1, 0.55, 2))
        print("=" * 16, 名, "|", round(r["耗时秒"], 2), "s")
        print(r["文"])
