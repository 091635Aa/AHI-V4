# -*- coding: utf-8 -*-
"""meeting59 / refs-web：生成 素材/洛天依/manifest.json（来源+切段+声学统计+sha256）
用法: python 工具_写manifest.py
说明: 源元数据来自 B 站官方 API（view/playurl，2026-10-19 实测返回）；声学统计来自 核验_片段统计.json
"""
import json, os, hashlib, datetime

BASE = os.path.dirname(os.path.abspath(__file__))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def finfo(name):
    p = os.path.join(BASE, name)
    return {"path": "03_AHI/AHI4/meeting59/素材/洛天依/" + name,
            "size_bytes": os.path.getsize(p), "sha256": sha256(p)}


stats = {r["file"]: r for r in json.load(open(os.path.join(BASE, "核验_片段统计.json"), encoding="utf-8"))}

SOURCES = [
    {
        "source_id": "L-A",
        "kind": "第三方测试样本（含 BGM，不用于声线拟合）",
        "bvid": "BV1qKJb68ESn",
        "aid": None,
        "cid": 42792650838,
        "page": 1,
        "url": "https://www.bilibili.com/video/BV1qKJb68ESn",
        "title": "天依只想抱抱你，夸夸你，辛苦啦！",
        "author": "洛天依（官方认证账号）",
        "owner_mid": 36081646,
        "pubdate": "2026-10-16 19:32:15 +08:00",
        "duration_s": 72,
        "dimension": "1080x1920（竖屏）",
        "tags": ["洛天依", "虚拟歌手", "治愈", "Vsinger", "live2D", "轻哄", "Vsinger创作激励计划"],
        "desc": "最近也辛苦啦，要记得好好休息哦~",
        "view_at_dl": 186704,
        "audio_stream": "dash audio id=30280, aac 44.1kHz stereo",
        "raw_file": finfo("raw_30280.m4s"),
        "decode_cmd": "ffmpeg -i raw_30280.m4s -ac 1 -ar 16000 -c:a pcm_s16le ref_洛天依_16k.wav（由 lead 执行，本目录核验）",
        "native_16k_file": finfo("ref_洛天依_16k.wav"),
        "acoustic": stats["ref_洛天依_16k.wav"],
        "verdict": {
            "speech_or_song": "【推断·中】说话/轻哄式念白（非歌唱）：限带 280–700 Hz F0 跟踪显示浊音段中位 0.10 s、最长 0.67 s、帧间 |ΔF0| 中位 0.14 st，接近本目录干净“说话”样本（0.15 s / 1.35 s / 0.16 st），明显不同于干净“清唱”样本（0.29 s / 3.34 s / 0.06 st）",
            "bgm": "【实测】有：停顿段仍有约 −58 dBFS 稳态能量、句间未到数字静音；60–200 Hz 存在单一窄带峰（中位 110 Hz，p10–p90 90–160 Hz，峰/均 ≈171×）→ 低频伴奏/底噪；立体声侧信号在停顿帧 side≈mid（−0.55 dB）→ 停顿底噪为单声道相关噪声（非立体声伴奏）",
            "reverb": "【实测】包络衰减斜率中位 23 dB/s、env 自相关 0.05 s=0.82/0.1 s=0.58 → 有可测的混响/拖尾（非干声）",
            "f0_note": "【实测】全带 F0 跟踪不可靠（低频伴奏占优，NSDF 锁到 ~133 Hz）；手工谱峰点查显示人声谐波系列 F0 ≈ 375–562 Hz（t=2.75/5.0 s 等 5 帧）；限带 280–700 Hz 后 F0 中位 409 Hz",
            "use": "仅作第三方对照/测试样本；不参与声线拟合",
        },
    },
    {
        "source_id": "L-B",
        "kind": "干净人声·说话（拟合用）",
        "bvid": "BV1aJ4m1475U",
        "aid": 1250235372,
        "cid": 1428303502,
        "page": 1,
        "url": "https://www.bilibili.com/video/BV1aJ4m1475U",
        "title": "洛天依送给锦依卫的语音",
        "author": "口亨口亨（第三方 UP 主，非官方账号）",
        "owner_mid": 1758025780,
        "pubdate": "2024-02-03 02:38:51 +08:00",
        "duration_s": 77,
        "dimension": "720x720",
        "tags": None,
        "desc": "-",
        "audio_stream": "dash audio id=30232, aac 78.9 kbps",
        "raw_file": finfo("raw_语音_口亨口亨.m4s"),
        "decode_cmd": "ffmpeg -i raw_语音_口亨口亨.m4s -ac 1 -ar 16000 -c:a pcm_s16le 核验_语音_16k.wav",
        "native_16k_file": finfo("核验_语音_16k.wav"),
        "acoustic": stats["核验_语音_16k.wav"],
        "verdict": {
            "speech_or_song": "【实测】说话：短句 burst（median 浊音段 0.19s、最长 1.39s）+ 句间真静音；F0 中位 346.8 Hz",
            "bgm": "【实测】无：停顿段 RMS 约 −95 dBFS（近数字静音），quiet/active 带能量比 −70.6 dB；无稳态窄峰",
            "reverb": "【实测】包络衰减斜率中位 13 dB/s（p25 −60 dB/s → 大量硬切），无显著长拖尾",
            "identity_note": "【未测】第三方上传的语音/语音包，声音身份由 UP 主标题声明；未做声纹身份认证",
            "use": "干净人声参考（说话），供 F0/LTAS/共振峰拟合",
        },
    },
    {
        "source_id": "L-C",
        "kind": "干净人声·歌声/清唱干声（拟合用）",
        "bvid": "BV1Wf4y1F711",
        "aid": 380889227,
        "cid": 493504440,
        "page": 1,
        "url": "https://www.bilibili.com/video/BV1Wf4y1F711",
        "title": "【洛天依AI干声提取】一梦千宵",
        "author": "塔赴P（第三方 UP 主）",
        "owner_mid": 258080618,
        "pubdate": "2022-01-27 07:22:25 +08:00",
        "duration_s": 256,
        "dimension": "1920x1080",
        "tags": None,
        "desc": "本家 BV1QL4y147wM；歌手：洛天依；…干声提取（UP 自述：消得可能不是很干净）",
        "audio_stream": "dash audio id=30280, aac 321.6 kbps",
        "raw_file": finfo("raw_干声_一梦千宵.m4s"),
        "decode_cmd": "ffmpeg -i raw_干声_一梦千宵.m4s -ac 1 -ar 16000 -c:a pcm_s16le 核验_干声_16k.wav",
        "native_16k_file": finfo("核验_干声_16k.wav"),
        "acoustic": stats["核验_干声_16k.wav"],
        "verdict": {
            "speech_or_song": "【实测】歌声：浊音占 71%、最长连续浊音 7.26 s、句内 F0 半音级稳定（帧间 |ΔF0| 中位 0.07 st）、谱图见颤音",
            "bgm": "【实测】无人声外 BGM：停顿段为数字静音（−240 dBFS，样本恒为 0）；但为第三方分离提取，非官方干声，可能有残留分离伪影（UP 自述）",
            "reverb": "【实测】衰减斜率中位 40 dB/s（干声特征）",
            "identity_note": "【实测/来源声明】本家为洛天依演唱；分离提取为第三方处理，非官方 dry vocal",
            "use": "干净人声参考（歌声），供 F0/LTAS/共振峰拟合",
        },
    },
]

SEGMENTS = [
    # (file, source, t0, t1, use_tag, note)
    ("ref_洛天依_语音_01.wav", "L-B", 8.09, 12.09, "说话/干净人声",
     "选中依据: 4s 窗 activity 0.94、段内静音 9%"),
    ("ref_洛天依_语音_02.wav", "L-B", 12.83, 16.83, "说话/干净人声",
     "activity 0.92、段内静音 10%"),
    ("ref_洛天依_语音_03.wav", "L-B", 61.45, 65.45, "说话/干净人声",
     "activity 0.90、段内静音 12%"),
    ("ref_洛天依_清唱_01.wav", "L-C", 31.17, 38.17, "歌声/干净人声",
     "7s 纯浊音窗（段内静音 0%）"),
    ("ref_洛天依_清唱_02.wav", "L-C", 82.77, 89.77, "歌声/干净人声",
     "7s 纯浊音窗（段内静音 0%）"),
    ("ref_洛天依_清唱_03.wav", "L-C", 218.04, 225.04, "歌声/干净人声",
     "7s 纯浊音窗（段内静音 0%）"),
]

segs = []
for f, sid, t0, t1, tag, note in SEGMENTS:
    s = stats[f]
    segs.append({
        "file": "03_AHI/AHI4/meeting59/素材/洛天依/" + f,
        "source_id": sid,
        "t0_s_in_source": t0,
        "t1_s_in_source": t1,
        "duration_s": s["duration_s"],
        "sample_rate": s["sample_rate"],
        "channels": 1,
        "sample_format": "s16",
        "size_bytes": s["size_bytes"],
        "sha256": s["sha256"],
        "use_tag": tag,
        "acoustic": s,
        "processing": "scipy: 去直流均值 + 5ms 淡入淡出 + s16 量化(未归一化增益: 峰值均 <1.0，无削波)；无降噪/无增强/无TTS/无RVC",
        "note": note,
    })

total_clean = sum(s["duration_s"] for s in segs)
man = {
    "manifest": "meeting59 / 洛天依第三方参考音频 v1",
    "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "written_by": "refs-web (team meeting59)",
    "goal": "供 meeting59 声线拟合器估计 14 维音色档案的第三方参考；只做声学参数估计。",
    "authorization": {
        "scope": "仅用于本机声学参数估计（F0/LTAS/共振峰/倾斜/HNR/语速等统计）；参考音频不再分发、不再上传、不用于训练外部模型、不做真声克隆。",
        "source_terms": "B 站第三方公开视频；版权归原作者/权利方；本目录仅保存分析用小片段，未主张任何权利。",
        "prohibited": "不得把片段重新发布/商用；不得用于 RVC/TTS 等外挂模型训练或推理（本项目 C10）。",
    },
    "processing_chain": [
        "1) B 站 API: x/web-interface/view 取 aid/cid/标题/UP/日期；x/player/wbi/playurl(fnval=16) 取 dash audio（请求头 UA + Referer=https://www.bilibili.com，wbi 签名）",
        "2) 下载最高码率 dash audio 流为 raw_*.m4s（不转码、不改内容）",
        "3) ffmpeg -ac 1 -ar 16000 -c:a pcm_s16le 解码为 16k 单声道",
        "4) scipy 核验: NSDF(McLeod) F0 跟踪 / 帧能量 / 7 带 LTAS / 静音率 / 停顿段稳态谱 / 包络衰减 / 立体声侧信号(仅源A)",
        "5) 切段: 能量+静音率打分选窗 → 去直流 + 5ms 淡入淡出 → s16（不归一化、不降噪）",
        "6) sha256 登记 + 片段统计落盘",
    ],
    "sources": SOURCES,
    "segments": segs,
    "totals": {
        "clean_ref_files": len(segs),
        "clean_ref_duration_s": round(total_clean, 3),
        "with_bgm_test_sample_duration_s": stats["ref_洛天依_16k.wav"]["duration_s"],
        "note": "干净人声合计 %.1fs（要求 20–50s）；另有 1 条含 BGM 的 71.7s 第三方测试样本单独登记，不参与拟合。" % total_clean,
    },
    "unverified": [
        "未做人耳盲听 / MOS 主观评价",
        "未做声纹身份认证（声音身份依据官方账号或 UP 主声明 + 谱学特征一致性）",
        "源 A（含 BGM）的 F0 未能可靠估计（伴奏低频占优）；其说话/歌声判断为推断",
        "源 C 为第三方分离干声，可能存在分离伪影（UP 主自述），未做伪影量化",
        "未测其它平台（YouTube/网易云等）备选源",
    ],
}

out = os.path.join(BASE, "manifest.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(man, f, ensure_ascii=False, indent=1)
print("[manifest]", out)
print("clean segments: %d, %.2fs" % (len(segs), total_clean))
for s in segs:
    print("  %-28s %s  %.2fs  F0 med %.1f Hz" % (
        s["file"].rsplit("/", 1)[-1], s["sha256"][:12], s["duration_s"],
        s["acoustic"]["f0_hz_p5_p50_p95"][1]))
