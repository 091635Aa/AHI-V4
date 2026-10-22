# -*- coding: utf-8 -*-
"""meeting68/实施/跑_自模仿Demo.py —— 自模仿·恢复 Demo。

产出（全部 mel→Griffin-Lim，无外部模型）：
  · 恢复：自己的干净 5 s（目标）/ 退化 L4 / 恢复（听自己恢复自己）
  · 语气：两个自己的片段 → z_A/z_B → z 插值 0/0.25/0.5/0.75/1 解码 → wav；耳统计应随 α 单调
  · 确定性：同输入同 ckpt 两趟 wav sha256 逐位同

说明：这里的「自己的声音」= 同一说话人真人片段（所有者指定的数据）；恢复目标是统计上像自己。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
M68 = HERE.parent
M66 = Path(r"<workspace>/03_AHI\AHI4\meeting66")
出 = M68 / "产物" / "自模仿恢复" / "Demo"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def 主():
    复 = 载("自模仿_恢复", HERE / "自模仿_恢复.py")
    耳 = 载("内生耳", HERE / "内生耳.py")
    声 = 载("声码器", M66 / "实施" / "声码器与评测.py")
    出.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(复.种子)
    torch.use_deterministic_algorithms(True) if hasattr(torch, "use_deterministic_algorithms") else None
    模型 = 复.恢复器().to(dev)
    sd = torch.load(M68 / "产物" / "自模仿恢复" / "恢复器.pt", map_location=dev)
    模型.load_state_dict(sd["state_dict"]); 模型.eval()
    数据 = 复.载数据()
    验 = 数据["val"]
    T = 复.T

    def 写mel(mel, 名):
        wav = 声.mel转波(np.asarray(mel, dtype=np.float32))
        p = 出 / 名
        声.写wav(p, wav)
        return str(p), wav

    # 选两个自己的片段：语气差异最大
    stats = []
    for m in 验:
        if m.shape[0] >= T:
            x = torch.from_numpy(m[:T]).float().to(dev)[None]
            with torch.no_grad():
                stats.append(复.耳统计(x)[0].cpu().numpy())
    距离 = np.zeros((len(stats), len(stats)))
    for i in range(len(stats)):
        for j in range(len(stats)):
            距离[i, j] = np.abs(stats[i] - stats[j]).mean()
    i, j = np.unravel_index(np.argmax(距离), 距离.shape)
    A = 验[i][:T]; B = 验[j][:T]
    xA = torch.from_numpy(A).float().to(dev)[None]
    xB = torch.from_numpy(B).float().to(dev)[None]
    with torch.no_grad():
        zA = 模型(xA)[1][0]; zB = 模型(xB)[1][0]
    # 恢复（退化 L4）
    rs = np.random.RandomState(复.种子 + 7)
    degA = 复.退化(xA, 4, rs)
    with torch.no_grad():
        恢A = 模型(degA)[0]
    p真, _ = 写mel(A, "自己的声音_干净_5s.wav")
    p退, _ = 写mel(degA[0].cpu().numpy(), "自己的声音_退化L4.wav")
    p恢, _ = 写mel(恢A[0].cpu().numpy(), "自己的声音_恢复.wav")
    # 语气插值（用 A 的退化输入 + 插值 z 解码）
    插 = []
    for α in (0.0, 0.25, 0.5, 0.75, 1.0):
        z = (1 - α) * zA + α * zB
        with torch.no_grad():
            恢 = 模型(degA, z外部=z[None])[0]
        p, _ = 写mel(恢[0].cpu().numpy(), f"语气插值_{α:.2f}.wav")
        st = 复.耳统计(torch.from_numpy(恢[0].cpu().numpy()).float().to(dev)[None])[0].cpu().numpy()
        插.append(dict(α=α, 文件=p, 耳统计=[round(float(v), 4) for v in st]))
    单调 = dict(
        能量=all(插[k]["耳统计"][0] <= 插[k + 1]["耳统计"][0] + 1e-6 for k in range(len(插) - 1)) or
             all(插[k]["耳统计"][0] >= 插[k + 1]["耳统计"][0] - 1e-6 for k in range(len(插) - 1)),
        谱质心=all(插[k]["耳统计"][2] <= 插[k + 1]["耳统计"][2] + 1e-6 for k in range(len(插) - 1)) or
               all(插[k]["耳统计"][2] >= 插[k + 1]["耳统计"][2] - 1e-6 for k in range(len(插) - 1)),
    )
    # 恢复效果（耳统计距离：真实 vs 退化/恢复）
    with torch.no_grad():
        真s = 复.耳统计(xA)[0].cpu().numpy()
        退s = 复.耳统计(degA)[0].cpu().numpy()
        恢s = 复.耳统计(恢A)[0].cpu().numpy()
    d退 = float(((退s - 真s) ** 2).mean()); d恢 = float(((恢s - 真s) ** 2).mean())
    # 确定性：恢复两趟
    with torch.no_grad():
        恢2 = 模型(degA)[0]
    import hashlib
    h1 = hashlib.sha256(np.asarray(声.mel转波(恢A[0].cpu().numpy()), np.float32).tobytes()).hexdigest()
    h2 = hashlib.sha256(np.asarray(声.mel转波(恢2[0].cpu().numpy()), np.float32).tobytes()).hexdigest()
    报 = dict(工具="跑_自模仿Demo.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              选片=dict(A=int(i), B=int(j), 语气差=round(float(距离[i, j]), 4)),
              恢复=dict(文件=dict(干净=p真, 退化=p退, 恢复=p恢),
                        耳统计平方误差=dict(退化=round(d退, 6), 恢复=round(d恢, 6),
                                            改善=round((d退 - d恢) / max(d退, 1e-12), 4))),
              语气插值=插, 单调=单调,
              确定性同sha=(h1 == h2),
              边界="恢复=统计上像自己；z 自监督、无情绪标签；mel→GL；未做人耳")
    (出 / "Demo报告.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(恢复=报["恢复"], 单调=单调, 确定性同sha=报["确定性同sha"]),
                     ensure_ascii=False, indent=1))
    print("已写：", 出 / "Demo报告.json")
    return 0


if __name__ == "__main__":
    sys.exit(主())
