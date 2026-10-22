# -*- coding: utf-8 -*-
"""meeting67 · P2-1（v2 修正）：F0/浊音离线提取（pyworld DIO + StoneMask，确定性）。

★ v1 → v2 修正（2026-10-21 13:1x，Lead 指出的接线/对齐问题）：
  `meeting66/产物/数据_对齐asr` 的 mel **不是整条 wav**，而是 `数据_SpeakerA_对齐.py` 用 ASR 字符时间戳
  **首末修剪后的语音段**（`起s = 首字起点 − 0.05 s`，`x段 = x22[int(起s*SR):int(止s*SR)]`）。
  v1 把整条 wav 的 F0 直接按帧数截断对齐 ⇒ **F0 与 mel 差一个起s 偏移**（实测样本 0 偏移 148 帧
  = 1.477 s），头训练目标整体错位。v2 修正为：
    1. 用与 `数据_SpeakerA_对齐.py` **逐参数相同**的管线重算**整条 wav** 的 log-mel（sf.read→resample_poly
       →melspectrogram hann/center/pad_constant/slaney）；
    2. 在整条 mel 上**搜索整数帧偏移** `off`，使 `mel_full[off:off+T]` 与 npz mel 最接近
       （粗搜步长 8 帧 + ±8 帧细搜），记录 MAE/max 作为对齐证据；
    3. 对整条 wav 做 pyworld DIO+StoneMask，取 `f0[off:off+T]` 作为该样本的帧级 F0；
       残余误差 ≤ 半帧 5 ms（mel 段起点的亚帧相位无法从 npz 完全恢复），逐条登记。
  定位不变（预注册 `11` §2.1，C10）：**离线分析/训练信号**，不 import 进生成路径。

帧率 10 ms、F0 60–500 Hz、清音 F0=0 + voicing=0；产物：
  - `meeting67/产物/P2/f0/<id:04d>.npz`（f0/logf0/voicing/帧数/id/偏移帧）
  - `meeting67/产物/P2/f0/索引.json`（参数/版本/逐文件 sha256/偏移/对齐 MAE/异常）
  - `meeting67/产物/P2/f0/阳性对照集.json`（A1 用：0075 + 4 条，含真值文本）
  - 自检：2 条样本重复提取两趟 → logf0/voicing/f0 逐位同。

用法：
  & <venv python> meeting67/实施/提取_F0浊音.py             # 全量 167（4 进程）
  & <venv python> meeting67/实施/提取_F0浊音.py --限 3       # 冒烟
  & <venv python> meeting67/实施/提取_F0浊音.py --进程 1     # 单进程（确定性复核）
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M67 = HERE.parent
AHI4 = M67.parent
M66 = AHI4 / "meeting66"
对齐目录 = M66 / "产物" / "数据_对齐asr"
样本目录 = 对齐目录 / "样本"
META = 对齐目录 / "meta.json"
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")
出目录 = M67 / "产物" / "P2" / "f0"
日志目录 = M67 / "团队" / "P2_实现"

SR = 22050
N_FFT = 1024
窗长 = 551
跳长 = 220                    # 10 ms @22.05k
MEL带数 = 80
FMIN = 0.0
FMAX = SR / 2.0
AMIN = 1e-6
F0下限 = 60.0
F0上限 = 500.0
帧周期ms = 10.0
允许范围 = 0.1
DIO速度 = 1
粗搜步 = 8
细搜半宽 = 8
对齐MAE门槛 = 1.0            # > 门槛 ⇒ 对齐存疑，记异常（不静默通过）
阳性对照必需 = 75
阳性对照另加 = 4

参数 = dict(
    工具="pyworld.dio + pyworld.stonemask（离线分析仪器，不入生成路径）",
    sr=SR, 帧移样本=跳长, 帧周期ms=帧周期ms,
    f0_floor=F0下限, f0_ceil=F0上限, allowed_range=允许范围, dio_speed=DIO速度,
    清音口径="F0=0 且 voicing=0；logF0 仅浊音帧非零（np.log(f0)）",
    帧对齐=("v2：重算整条 wav 的 log-mel（与 数据_SpeakerA_对齐.py 逐参数同）→ 搜索整数帧偏移 off "
        "使 mel_full[off:off+T] 逼近 npz mel → 取 f0_full[off:off+T]；残余 ≤ 半帧 5ms"),
    用途="仅训练/分析信号（头训练 + A1 阳性对照），不进入生成路径",
)
版本 = "2.0"


def _sha256数组(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _sha256文件(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for 块 in iter(lambda: f.read(1 << 20), b""):
            h.update(块)
    return h.hexdigest()


def _环境() -> dict:
    import pyworld
    import librosa
    import scipy
    import soundfile
    return dict(pyworld=getattr(pyworld, "__version__", "0.3.x"),
                numpy=np.__version__, scipy=scipy.__version__,
                librosa=librosa.__version__, soundfile=soundfile.__version__)


def 读音频22k(p: Path) -> np.ndarray:
    """与 `数据_SpeakerA_对齐.py.读音频22k` 逐参数同（sf.read → 均值单声道 → resample_poly）。"""
    import soundfile as sf
    from scipy import signal as sps
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1, dtype=np.float32)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = sps.resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x)


def 算mel(x: np.ndarray) -> np.ndarray:
    """与 `数据_SpeakerA_对齐.py.算mel` 逐参数同。"""
    import librosa
    S = librosa.feature.melspectrogram(
        y=x, sr=SR, n_fft=N_FFT, hop_length=跳长, win_length=窗长, window="hann",
        center=True, pad_mode="constant", power=2.0, n_mels=MEL带数,
        fmin=FMIN, fmax=FMAX, norm="slaney", htk=False)
    return np.log(S + AMIN).astype(np.float32).T


def 求mel偏移(mel_full: np.ndarray, mel: np.ndarray) -> tuple:
    """整条 mel 与 npz mel 的整数帧偏移（粗搜 step=8 + 细搜 ±8）。→ (off, MAE, 最大差)"""
    T = int(mel.shape[0])
    上限 = int(mel_full.shape[0]) - T
    if 上限 < 0:
        # npz mel 比整条还长（异常）：只给 0，标记
        off = 0
    else:
        采 = mel_full[::粗搜步]
        样 = mel[::粗搜步]
        步数 = (采.shape[0] - 样.shape[0])
        best = None
        for k in range(0, max(步数, 0) + 1):
            d = float(np.abs(采[k:k + 样.shape[0]] - 样).mean())
            if best is None or d < best[1]:
                best = (k * 粗搜步, d)
        off0 = int(best[0]) if best else 0
        best = None
        for off in range(max(0, off0 - 细搜半宽), min(上限, off0 + 细搜半宽) + 1):
            d = float(np.abs(mel_full[off:off + T] - mel).mean())
            if best is None or d < best[1]:
                best = (off, d)
        off = int(best[0]) if best else 0
    段 = mel_full[off:off + T]
    mae = float(np.abs(段 - mel).mean()) if 段.shape == mel.shape else 9.99
    最大 = float(np.abs(段 - mel).max()) if 段.shape == mel.shape else 9.99
    return off, mae, 最大


def 提取一条(id_: int, 写文件: bool = True) -> dict:
    """单条：offset 恢复 → 整条 wav F0 → 取 [off:off+T]。"""
    import pyworld as pw
    meta = json.loads(META.read_text(encoding="utf-8"))
    行 = meta["样本"][int(id_)]
    z = np.load(str(样本目录 / ("%04d.npz" % int(id_))))
    mel = np.asarray(z["mel"], dtype=np.float32)
    T = int(mel.shape[0])
    wav名 = str(行["wav"]).lstrip()
    wav路 = 音频根 / wav名
    if not wav路.exists():
        raise FileNotFoundError("wav 不存在：%s" % wav路)
    x = 读音频22k(wav路)
    mel_full = 算mel(x)
    off, mae, 最大 = 求mel偏移(mel_full, mel)

    f0, t = pw.dio(x.astype(np.float64), SR, f0_floor=F0下限, f0_ceil=F0上限,
                   frame_period=帧周期ms, allowed_range=允许范围, speed=DIO速度)
    f0 = np.asarray(pw.stonemask(x.astype(np.float64), f0, t, SR), dtype=np.float64)
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    f0 = np.where(np.isfinite(f0) & (f0 >= F0下限) & (f0 <= F0上限), f0, 0.0)
    if off + T > f0.size:
        f0 = np.pad(f0, (0, off + T - f0.size), mode="constant", constant_values=0.0)
    f0 = f0[off:off + T]
    voicing = (f0 > 0.0).astype(np.int8)
    logf0 = np.zeros_like(f0, dtype=np.float32)
    logf0[voicing == 1] = np.log(f0[voicing == 1]).astype(np.float32)
    f0 = f0.astype(np.float32)

    报 = dict(
        id=int(id_), npz="%04d.npz" % int(id_), wav=wav名,
        wav_sha256=_sha256文件(wav路),
        mel_sha256=_sha256数组(mel),
        帧数=T, 偏移帧=int(off), 偏移秒=round(off * 跳长 / SR, 4),
        mel对齐MAE=round(mae, 5), mel对齐最大差=round(最大, 5),
        对齐=("达标" if mae <= 对齐MAE门槛 else "存疑"),
        wav样本数=int(x.size), mel_full帧数=int(mel_full.shape[0]),
        浊音帧=int(voicing.sum()),
        浊音比=round(float(voicing.mean()) if T else 0.0, 4),
        f0中位=round(float(np.median(f0[voicing == 1])) if voicing.sum() else 0.0, 3),
        logf0_sha256=_sha256数组(logf0), voicing_sha256=_sha256数组(voicing),
    )
    if 写文件:
        np.savez(str(出目录 / ("%04d.npz" % int(id_))), f0=f0, logf0=logf0,
                 voicing=voicing, 帧数=np.int32(T), id=np.int32(id_), 偏移帧=np.int32(off))
        报["sha256"] = _sha256文件(出目录 / ("%04d.npz" % int(id_)))
    return 报


def _worker(args) -> tuple:
    id_, 写文件 = args
    try:
        return 提取一条(id_, 写文件)
    except Exception as e:  # noqa: BLE001
        return dict(id=int(id_), 错误="%s: %s" % (type(e).__name__, str(e)[:200]))


def 主(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    限 = None
    进程数 = 4
    if "--限" in argv:
        限 = int(argv[argv.index("--限") + 1])
    if "--进程" in argv:
        进程数 = max(1, int(argv[argv.index("--进程") + 1]))
    写文件 = "--不写" not in argv

    出目录.mkdir(parents=True, exist_ok=True)
    日志目录.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding="utf-8"))
    样本 = meta["样本"]
    ids = [int(r["id"]) for r in 样本]
    if 限:
        ids = ids[:限]
    print("[F0-v2] 版本=%s 环境：%s" % (版本, json.dumps(_环境(), ensure_ascii=False)), flush=True)
    print("[F0-v2] 样本 %d 条；进程=%d；出目录 %s" % (len(ids), 进程数, 出目录), flush=True)

    t0 = time.time()
    结果s = []
    if 进程数 <= 1:
        for id_ in ids:
            结果s.append(_worker((id_, 写文件)))
    else:
        with ProcessPoolExecutor(max_workers=进程数) as 池:
            for r in 池.map(_worker, [(i, 写文件) for i in ids], chunksize=4):
                结果s.append(r)
                好 = "错误" not in r
                print("[F0-v2] %3d/%3d %04d %s 偏移=%s帧 MAE=%s %s"
                      % (len(结果s), len(ids), r.get("id", -1),
                         ("帧=%d 浊音比=%.3f" % (r["帧数"], r["浊音比"])) if 好 else r["错误"][:80],
                         r.get("偏移帧"), r.get("mel对齐MAE"), r.get("对齐", "")), flush=True)

    逐条 = [r for r in 结果s if "错误" not in r]
    异常 = [r for r in 结果s if "错误" in r]
    存疑 = [r for r in 逐条 if r.get("对齐") != "达标"]
    逐条.sort(key=lambda r: r["id"])

    # 确定性自检：2 条重复提取（不写文件，比对数组 sha）
    自检 = []
    for id_ in ids[:2]:
        try:
            a = 提取一条(id_, 写文件=False)
            b = 提取一条(id_, 写文件=False)
            同 = bool(a["logf0_sha256"] == b["logf0_sha256"]
                      and a["voicing_sha256"] == b["voicing_sha256"])
            自检.append(dict(id=int(id_), 两趟逐位同=同, 偏移帧=a["偏移帧"],
                            mel对齐MAE=a["mel对齐MAE"]))
        except Exception as e:  # noqa: BLE001
            自检.append(dict(id=int(id_), 错误="%s: %s" % (type(e).__name__, str(e)[:120])))
    print("[F0-v2] 确定性自检：%s" % json.dumps(自检, ensure_ascii=False), flush=True)

    # 阳性对照集：0075 + 最短的 4 条（不含 0075）
    阳性 = [样本[阳性对照必需]]
    for r in sorted(样本, key=lambda r: int(r["帧数"])):
        if int(r["id"]) == 阳性对照必需:
            continue
        阳性.append(r)
        if len(阳性) >= 1 + 阳性对照另加:
            break
    阳性输出 = []
    for r in 阳性:
        id_ = int(r["id"])
        if not (出目录 / ("%04d.npz" % id_)).exists():
            continue
        阳性输出.append(dict(id=id_, npz="%04d.npz" % id_, wav=r["wav"], 文件=r["文件"],
                             帧数=int(r["帧数"]), 时长秒=r["时长秒"], 文本=r.get("文"),
                             mel_sha256=next((x["mel_sha256"] for x in 逐条 if x["id"] == id_), None)))
    阳性输出 = sorted(阳性输出, key=lambda x: x["id"])
    (出目录 / "阳性对照集.json").write_text(
        json.dumps(dict(说明="A1 阳性对照：真 mel + 真 F0/浊音 → 谐波声码器 → ASR（护栏 P2-1，CER≤0.15）",
                        必需=阳性对照必需, 条=阳性输出, 版本=版本), ensure_ascii=False, indent=2),
        encoding="utf-8")

    索引 = dict(
        工具="提取_F0浊音.py", 版本=版本, 任务="meeting67 P2-1 F0/浊音离线提取（v2 偏移修正）",
        时间=time.strftime("%Y-%m-%d %H:%M:%S"),
        脚本sha256=_sha256文件(Path(__file__).resolve()),
        参数=参数, 环境=_环境(), 音频根=str(音频根), mel源=str(样本目录),
        条数=len(逐条), 异常=异常, 异常数=len(异常),
        对齐存疑=存疑, 对齐存疑数=len(存疑),
        偏移帧统计=dict(最小=min((x["偏移帧"] for x in 逐条), default=None),
                    最大=max((x["偏移帧"] for x in 逐条), default=None),
                    非零条数=int(sum(1 for x in 逐条 if x["偏移帧"] != 0)),
                    MAE中位=round(float(np.median([x["mel对齐MAE"] for x in 逐条])), 5) if 逐条 else None),
        确定性自检=自检,
        自检PASS=bool(自检 and all(x.get("两趟逐位同") for x in 自检)),
        逐条=逐条, 耗时秒=round(time.time() - t0, 1),
    )
    (出目录 / "索引.json").write_text(json.dumps(索引, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print("[F0-v2] 完成：%d 条，异常 %d，对齐存疑 %d，耗时 %.1fs，自检PASS=%s"
          % (len(逐条), len(异常), len(存疑), time.time() - t0, 索引["自检PASS"]), flush=True)
    print("[F0-v2] 产物：%s" % (出目录 / "索引.json"), flush=True)
    return 0 if (逐条 and not 异常) else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(主())
