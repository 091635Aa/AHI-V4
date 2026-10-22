# -*- coding: utf-8 -*-
"""测试：Paraformer-seaco 是否支持字符级时间戳（供 TTS 对齐用）。只跑一次短音频。"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.io import wavfile  # noqa: E402


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    md = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "models",
                      "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    from funasr import AutoModel
    m = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                  disable_update=True, trust_remote_code=True, device="cpu",
                  cache_dir=os.path.dirname(md))
    # 取一段 12s
    src = r"<path>\SpeakerA7代模型训练数据\(1).wav"
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(axis=1)[: sr * 12]
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_12s.wav")
    wavfile.write(p, sr, (np.clip(x, -1, 1) * 32767).astype("<i2"))
    for 参数 in (dict(return_timestamp=True), dict()):
        try:
            r = m.generate(input=p, batch_size_s=300, **参数)
            it = r[0]
            print("参数", 参数, "→ keys:", list(it.keys()))
            print("  text:", str(it.get("text", ""))[:120])
            ts = it.get("timestamp")
            print("  timestamp:", (ts[:10] if ts else None), " 条数:", len(ts) if ts else 0)
        except Exception as e:  # noqa: BLE001
            print("参数", 参数, "→ 异常:", type(e).__name__, str(e)[:200])
    try:
        os.remove(p)
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(主())
