# -*- coding: utf-8 -*-
"""meeting68/实施/学习引擎_自听.py —— 自听学习：听自己 → 优化自己（闭合回路）。

与人类学习逻辑的对应（登记为机制类比，非等同）：
  · 目标：真人（同一说话人）一段语音 → 内生耳嵌入（「听到别人/目标的声音」）
  · 自产：最小音频线按控制参数 θ 生成 → 内生耳嵌入（「听到自己的声音」）
  · 误差：耳嵌入空间加权距离（不含外部 ASR/模型）
  · 学习：坐标下降（每维 ±步长，保优；无改进则缩步长），只优化 4 个旋钮：
        θ = [音高半音, 增益dB, 谱倾斜dB, 每音节帧数]
  · 输出：学习曲线、优化前后音频、参数、耳距离分项。

确定性：固定目标、固定文本、固定起点、固定步长顺序；同输入两趟同结果。
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
M68 = HERE.parent
M66 = Path(r"<workspace>/03_AHI\AHI4\meeting66")
数据根 = Path(r"<path>\SpeakerA7代模型训练数据")
出根 = M68 / "产物" / "自听学习"
文本 = "今天的风很轻，适合慢慢地说一会儿话。"
目标wav = 数据根 / "(1).wav"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def 主(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="meeting68 自听学习引擎")
    ap.add_argument("--轮", type=int, default=8, help="坐标下降轮数")
    ap.add_argument("--文本", default=文本)
    ap.add_argument("--目标", default=str(目标wav))
    ap.add_argument("--出", default=str(出根))
    ap.add_argument("--安静", action="store_true", help="只打印结果")
    参 = ap.parse_args(argv)
    出 = Path(参.出); 出.mkdir(parents=True, exist_ok=True)
    线 = 载("音频线", HERE / "音频线_v0.py")
    耳 = 载("内生耳", HERE / "内生耳.py")

    t0 = time.time()
    目标路 = Path(参.目标)
    if not 目标路.exists():
        raise FileNotFoundError(目标路)
    # 目标取前 20 s（同一说话人真实语音；不是同句）
    import soundfile as sf
    x, sr = sf.read(str(目标路), dtype="float32", always_2d=True)
    m = np.ascontiguousarray(x.mean(axis=1)[: 20 * sr])
    目标截 = 出 / "目标_真人20s.wav"
    sf.write(str(目标截), m, sr)
    目 = 耳.耳嵌入(目标截)
    print("[目标] 耳嵌入:", json.dumps({k: 目[k] for k in ("F0中位", "语速", "能量均值", "谱质心")}, ensure_ascii=False))

    # θ = [音高半音, 增益dB, 倾斜dB, 每音节帧数, 停顿毫秒]
    θ = np.array([0.0, 0.0, 0.0, 24.0, 60.0], dtype=np.float64)
    步 = np.array([1.5, 1.5, 1.5, 2.0, 40.0], dtype=np.float64)
    下界 = np.array([-6.0, -6.0, -6.0, 18.0, 0.0])
    上界 = np.array([8.0, 6.0, 6.0, 30.0, 300.0])

    缓存 = {}

    def 评(θv):
        键 = tuple(np.round(θv, 2).tolist())
        if 键 in 缓存:
            return 缓存[键]
        r = 线.合成(参.文本, 音高=θv[0], 增益=θv[1], 倾斜=θv[2],
                    每音节帧数=int(round(θv[3])), 停顿毫秒=float(θv[4]),
                    迭代=4, 阻尼=0.5, 保存=None, 采样嵌入=True)
        # 临时写 wav 以计算耳嵌入（合成函数未保存时无路径）
        tmp = 出 / "_临时.wav"
        线._声码器().写wav(tmp, r["波形"])
        emb = 耳.耳嵌入(tmp)
        d = 耳.距离(emb, 目)
        r["耳嵌入"] = emb; r["耳距离"] = d
        r.pop("波形", None)
        缓存[键] = r
        return r

    曲线 = []
    最佳 = 评(θ)
    曲线.append(dict(阶段="初始", θ=θ.tolist(), 损失=最佳["耳距离"]["总"]))
    最好损失 = 最佳["耳距离"]["总"]
    for 轮 in range(int(参.轮)):
        改进 = False
        for i in range(5):
            for 方向 in (+1.0, -1.0):
                试 = θ.copy()
                试[i] = float(np.clip(试[i] + 方向 * 步[i], 下界[i], 上界[i]))
                if np.allclose(试, θ):
                    continue
                r = 评(试)
                损 = r["耳距离"]["总"]
                曲线.append(dict(阶段=f"轮{轮+1}", 维=i, 方向=方向, θ=试.tolist(), 损失=损))
                if 损 < 最好损失 - 1e-6:
                    最好损失 = 损; θ = 试; 最佳 = r; 改进 = True
        if not 改进:
            步 = 步 * 0.5
        print(f"[轮{轮+1}] 最好损失 {最好损失:.6f} θ={np.round(θ,2).tolist()} 步={np.round(步,2).tolist()}")

    # 保存优化前后
    初 = 曲线[0]["θ"]
    前 = 线.合成(参.文本, 音高=初[0], 增益=初[1], 倾斜=初[2], 每音节帧数=int(round(初[3])),
                 停顿毫秒=float(初[4]), 保存=str(出 / "学习前_平静.wav"), 采样嵌入=False)
    后 = 线.合成(参.文本, 音高=θ[0], 增益=θ[1], 倾斜=θ[2], 每音节帧数=int(round(θ[3])),
                 停顿毫秒=float(θ[4]), 保存=str(出 / "学习后_自优化.wav"), 采样嵌入=True)
    前.pop("波形", None); 后.pop("波形", None)
    报 = dict(
        工具="学习引擎_自听.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
        目标=dict(路径=str(目标路), 截取秒=20.0, 耳嵌入=目),
        文本=参.文本, 轮数=参.轮,
        初始θ=初, 最优θ=θ.tolist(),
        损失=dict(初始=曲线[0]["损失"], 最终=最好损失,
                 改善=曲线[0]["损失"] - 最好损失,
                 相对改善=(曲线[0]["损失"] - 最好损失) / max(曲线[0]["损失"], 1e-12)),
        学习前=前, 学习后=后,
        曲线=曲线, 评估次数=len(缓存), 耗时秒=round(time.time() - t0, 1),
        边界="目标为同一说话人不同句（非同句）；耳为自研 DSP；不含外部模型；改善按耳距离，不代表可懂度提升",
    )
    (出 / "自听学习_报告.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    (出 / "自听学习_曲线.json").write_text(json.dumps(曲线, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: 报[k] for k in ("初始θ", "最优θ", "损失", "评估次数", "耗时秒")}, ensure_ascii=False, indent=1))
    print("已写：", 出 / "自听学习_报告.json")
    return 0


if __name__ == "__main__":
    sys.exit(主())
