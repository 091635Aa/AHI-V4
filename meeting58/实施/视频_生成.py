# -*- coding: utf-8 -*-
"""meeting58/实施/视频_生成.py —— 把 demo 音频剪成成片（横屏 1920×1080 + 竖屏 1080×1920）。

设计：PIL 逐帧合成（背景/大字/字幕/波形条/进度条），rawvideo 管道喂 ffmpeg( libx264 + aac )；
BGM 用 `ahi-intro/public/bgm.mp3`（工作区既有素材）低音量垫底。
字幕与音频来自 `产物/演示/演示清单.json`；同时导出 .srt。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
M58 = os.path.dirname(HERE)
演示 = os.path.join(M58, "产物", "演示")
产物 = os.path.join(M58, "产物", "视频")
os.makedirs(产物, exist_ok=True)
FS = 16000
FPS = 24
BGM = r"<workspace>/ahi-intro\public\bgm.mp3"
FONT_B = r"<path>\Windows\Fonts\msyhbd.ttc"
FONT_R = r"<path>\Windows\Fonts\msyh.ttc"

白 = (245, 248, 255)
灰 = (168, 180, 205)
黑 = (0, 0, 0)


def 读wav(路径: str):
    with wave.open(路径, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64) / 32768.0
    return sr, x


def 载BGM(时长秒: float) -> np.ndarray:
    tmp = os.path.join(产物, "_bgm16k.wav")
    if not os.path.exists(tmp):
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", BGM, "-ac", "1", "-ar", str(FS), tmp], check=True)
    sr, x = 读wav(tmp)
    off = int(18 * FS)                      # 跳过 BGM 前 18s 的弱起，直接用有声段
    if len(x) > off + FS:
        x = x[off:]
    n = int(时长秒 * FS) + FS
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / max(len(x), 1))))
    x = x[:n]
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    return x / rms * 0.045          # BGM 归一化到 RMS 0.045（人声段约 0.2–0.4，垫底不抢）


def 六色(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ─────────────────────── 文本工具 ───────────────────────
def 载字体(size: int, 粗: bool = False):
    return ImageFont.truetype(FONT_B if 粗 else FONT_R, size)


def 折行(文: str, 字宽: int) -> list:
    out, cur = [], ""
    for ch in 文:
        if len(cur) >= 字宽:
            out.append(cur); cur = ""
        cur += ch
    if cur:
        out.append(cur)
    return out


def 渐变背景(W, H, 主色, 副色):
    y = np.linspace(0, 1, H)[:, None]
    base = np.zeros((H, W, 3), dtype=np.float64)
    base[:, :, 0] = 5 + 12 * (1 - y)
    base[:, :, 1] = 7 + 16 * (1 - y)
    base[:, :, 2] = 14 + 28 * (1 - y)
    # 主色光晕（右上）
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = int(W * 0.78), int(H * 0.22)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(W, H)
    glow = np.exp(-(r ** 2) / 0.10)[..., None]
    base = base + glow * np.array(主色, dtype=np.float64) * 0.16
    base = base + glow * np.array(副色, dtype=np.float64) * 0.06
    # 细网格
    for x in range(0, W, 48):
        base[:, x:x + 1, :] += 4
    for y2 in range(0, H, 48):
        base[y2:y2 + 1, :, :] += 4
    return np.clip(base, 0, 255).astype(np.uint8)


def 画标题(d: ImageDraw.ImageDraw, W, 文本, y, 主色, 尺寸=96):
    f = 载字体(尺寸, True)
    w = d.textlength(文本, font=f)
    d.text(((W - w) / 2, y), 文本, font=f, fill=白)
    d.rectangle([(W - 120) / 2, y + 尺寸 + 28, (W + 120) / 2, y + 尺寸 + 34], fill=主色)


# ─────────────────────── 时间线 ───────────────────────
def 建时间线(清单: dict) -> dict:
    t = 0.0
    段 = []
    段.append(dict(kind="title", start=t, dur=4.0, 主色=(90, 200, 250), 副色=(35, 90, 160)))
    t += 4.0
    段.append(dict(kind="problem", start=t, dur=4.6, 主色=(255, 92, 92), 副色=(150, 40, 50)))
    t += 4.6
    段.append(dict(kind="arch", start=t, dur=5.0, 主色=(120, 220, 200), 副色=(30, 110, 100)))
    t += 5.0
    for 角色 in 清单["角色"]:
        主, 副 = 六色(角色["主色"]), 六色(角色["副色"])
        句列 = sorted(角色["句"], key=lambda x: x["wav"])
        起 = t + 0.65
        段.append(dict(kind="role", start=t, dur=0, 角色=角色, 主色=主, 副色=副,
                      句列=句列, 句起=[], 句止=[]))
        for 句 in 句列:
            段[-1]["句起"].append(起)
            段[-1]["句止"].append(起 + float(句["时长秒"]))
            起 += float(句["时长秒"]) + 0.5
        段[-1]["dur"] = 起 - 0.5 + 0.65 - t
        t += 段[-1]["dur"]
    # 同句五声
    主, 副 = (255, 200, 97), (169, 116, 31)
    起 = t + 0.8
    段.append(dict(kind="montage", start=t, dur=0, 主色=主, 副色=副,
                  句列=清单["同句五声"], 句起=[], 句止=[]))
    for 句 in 清单["同句五声"]:
        段[-1]["句起"].append(起)
        段[-1]["句止"].append(起 + float(句["时长秒"]))
        起 += float(句["时长秒"]) + 0.18
    段[-1]["dur"] = 起 - 0.18 + 0.8 - t
    t += 段[-1]["dur"]
    段.append(dict(kind="truth", start=t, dur=5.2, 主色=(150, 210, 255), 副色=(40, 90, 150)))
    t += 5.2
    段.append(dict(kind="end", start=t, dur=4.2, 主色=(120, 230, 190), 副色=(30, 110, 90)))
    t += 4.2
    return dict(段=段, 总时长=t)


def 建音频(时间线: dict) -> np.ndarray:
    总 = int(np.ceil(时间线["总时长"] * FS)) + FS
    声 = np.zeros(总)
    for 段 in 时间线["段"]:
        if 段["kind"] == "role":
            for i, 句 in enumerate(段["句列"]):
                sr, x = 读wav(os.path.join(演示, 句["wav"]))
                a = int(段["句起"][i] * FS)
                声[a:a + len(x)] += x * 0.95
        elif 段["kind"] == "montage":
            for i, 句 in enumerate(段["句列"]):
                sr, x = 读wav(os.path.join(演示, 句["wav"]))
                a = int(段["句起"][i] * FS)
                声[a:a + len(x)] += x * 0.95
    声 = 声[:总]
    bg = 载BGM(总 / FS)[:总]
    混 = 声 + bg
    峰 = float(np.max(np.abs(混))) + 1e-9
    if 峰 > 0.98:
        混 = 混 * 0.98 / 峰
    p = os.path.join(产物, "演示音频.wav")
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
        w.writeframes((混 * 32767).astype("<i2").tobytes())
    return 混


# ─────────────────────── 绘制 ───────────────────────
def 画波形(d, 混, t, 段颜色, W, H, 竖屏):
    窗 = 2.2
    n = 118 if not 竖屏 else 78
    边 = 60 if not 竖屏 else 60
    宽 = W - 边 * 2
    条宽 = max(3, int(宽 / n * 0.55))
    间距 = 宽 / n
    y底 = int(H * 0.905) if not 竖屏 else int(H * 0.855)
    高 = int(H * 0.07) if not 竖屏 else int(H * 0.055)
    a = int(max(0, t - 窗) * FS)
    b = int(min(len(混), t * FS))
    if b - a < 100:
        return
    seg = 混[a:b]
    win = max(1, len(seg) // n)
    for i in range(n):
        s = seg[i * win:(i + 1) * win]
        if len(s) == 0:
            continue
        v = float(np.sqrt(np.mean(s ** 2)))
        hh = max(2, int(min(1.0, v * 6.0) * 高))
        x = 边 + i * 间距
        d.rectangle([x, y底 - hh, x + 条宽, y底], fill=段颜色)


def 画界面(段, t, 混, W, H, 竖屏):
    d_img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(d_img)
    bg = 段["_bg"]
    d_img = Image.fromarray(bg)
    d = ImageDraw.Draw(d_img)
    kind = 段["kind"]
    主 = 段["主色"]
    if kind == "title":
        画标题(d, W, "AHI · 声线复制", int(H * 0.30), 主, 96 if not 竖屏 else 92)
        f = 载字体(44 if not 竖屏 else 48)
        s = "一个自我意识体，用自己长出来的嗓子说话"
        d.text(((W - d.textlength(s, font=f)) / 2, int(H * 0.47)), s, font=f, fill=灰)
        f2 = 载字体(28)
        s2 = "meeting58 · 角色复刻 × 输出层解耦 × 动态挂载 · 2026-10-19"
        d.text(((W - d.textlength(s2, font=f2)) / 2, int(H * 0.56)), s2, font=f2, fill=灰)
    elif kind == "problem":
        画标题(d, W, "问题：为什么复刻效果差", int(H * 0.18), 主, 78)
        行 = ["① 8 个音色参数只有 2 个进谱：紧喉/粗糙 无出口",
              "② 状态从未进入声带：D核 64 → 声带 1024，被随机向量顶替",
              "③ 角色逻辑与音色输出没有契约：调用点直接串在一起"]
        f = 载字体(38 if not 竖屏 else 44)
        for i, s in enumerate(行):
            y = int(H * 0.34) + i * int(H * 0.105)
            d.rectangle([int(W * 0.10), y + 6, int(W * 0.10) + 6, y + 44], fill=主)
            d.text((int(W * 0.10) + 20, y), s, font=f, fill=白)
    elif kind == "arch":
        画标题(d, W, "方案：两层解耦 + 一个注册表", int(H * 0.13), 主, 74)
        f = 载字体(34 if not 竖屏 else 38)
        f2 = 载字体(26)
        框 = [(0.09, 0.20, "状态源", "只读快照 / 确定性投影"),
              (0.34, 0.25, "行层 · 角色逻辑", "只产文本，不碰波形"),
              (0.64, 0.26, "声层 · 音色渲染", "只产音频，不产文本")]
        for xr, w, 名, 小 in 框:
            x = int(W * xr); y = int(H * 0.38); bw = int(W * w)
            d.rounded_rectangle([x, y, x + bw, y + int(H * 0.20)], radius=16,
                                outline=主, width=3)
            d.text((x + 18, y + 14), 名, font=f, fill=白)
            d.text((x + 18, y + 62), 小, font=f2, fill=灰)
        d.text((int(W * 0.10), int(H * 0.66)),
               "注册表：注册 → 试跑 → 句边界切换 → 回滚 → 卸载（连续失败 3 次自动隔离）",
               font=f2, fill=灰)
        d.text((int(W * 0.10), int(H * 0.73)),
               "换音色不重启 · 换角色模板不重建核 · 状态轨迹指纹不变",
               font=f2, fill=主)
    elif kind == "role":
        角 = 段["角色"]
        参 = 角.get("参数", {})
        左 = int(W * 0.09) if not 竖屏 else int(W * 0.08)
        d.rectangle([左 - 18, int(H * 0.20), 左 - 10, int(H * 0.34)], fill=主)
        f = 载字体(84 if not 竖屏 else 88, True)
        d.text((左, int(H * 0.19)), 角["角色"], font=f, fill=白)
        f2 = 载字体(28)
        d.text((左, int(H * 0.30)), 角["英文"], font=f2, fill=主)
        # 声线档案卡（不可变参数 + 来源）
        if not 竖屏:
            卡 = [int(W * 0.60), int(H * 0.20), int(W * 0.91), int(H * 0.50)]
        else:
            卡 = [int(W * 0.08), int(H * 0.40), int(W * 0.92), int(H * 0.56)]
        d.rounded_rectangle(卡, radius=18, outline=主, width=2)
        fc = 载字体(26 if not 竖屏 else 28)
        行 = ["声线档案（不可变 · 可热切换）",
              "F0 %.0f Hz · 共振峰 ×%.2f · 声区 %.2f"
              % (参.get("基频", 0), 参.get("共振峰", 1), 参.get("声区", 0)),
              "谱倾斜 %+.2f · 气声 %.2f · 紧喉 %.2f"
              % (参.get("谱倾斜", 0), 参.get("气声", 0), 参.get("紧喉", 0)),
              "来源：%s" % 角.get("来源", "")]
        for i, s in enumerate(行):
            d.text((卡[0] + 22, 卡[1] + 18 + i * 40), s, font=fc,
                   fill=(主 if i == 0 else 白))
        # 两句字幕：当前句高亮
        f3 = 载字体(42 if not 竖屏 else 44, False)
        y0 = int(H * 0.66) if not 竖屏 else int(H * 0.63)
        for i, 句 in enumerate(段["句列"]):
            进行中 = 段["句起"][i] <= t <= 段["句止"][i]
            color = 白 if 进行中 else 灰
            y = y0 + i * int(H * 0.09)
            for j, line in enumerate(折行(句["文本"], 22 if not 竖屏 else 18)):
                d.text((左, y + j * 52), line, font=f3, fill=color)
    elif kind == "montage":
        画标题(d, W, "同一句话 × 五种声线", int(H * 0.12), 主, 74)
        f = 载字体(40 if not 竖屏 else 44, True)
        f2 = 载字体(26)
        名 = [x["角色"] for x in 段["句列"]]
        for i, n in enumerate(名):
            进行中 = 段["句起"][i] <= t <= 段["句止"][i]
            x = int(W * 0.09) + (i % 5) * int(W * 0.17)
            y = int(H * 0.34) + (0 if not 竖屏 else 0)
            d.text((x, y), n, font=f, fill=(白 if 进行中 else 灰))
            if 进行中:
                d.rectangle([x, y + 58, x + d.textlength(n, font=f), y + 64], fill=主)
        s = "「你好，我在这里。」"
        d.text(((W - d.textlength(s, font=f)) / 2, int(H * 0.52)), s, font=f, fill=白)
        s2 = "同一状态 · 句边界热切换 · 换音色不重启"
        f4 = 载字体(26)
        d.text(((W - d.textlength(s2, font=f4)) / 2, int(H * 0.60)), s2, font=f4, fill=灰)
    elif kind == "truth":
        画标题(d, W, "诚实边界", int(H * 0.14), 主, 78)
        行 = ["参数化模仿，不是真声克隆；不是角色本人",
              "角色声线来自公开资料特征归纳 [文献类比]",
              "全部声音由 AHI 原生声带（Rosenberg + Klatt）合成",
              "状态来自 V5 双通道体真实轨迹；两趟逐位同"]
        f = 载字体(34 if not 竖屏 else 40)
        for i, s in enumerate(行):
            y = int(H * 0.34) + i * int(H * 0.105)
            d.ellipse([int(W * 0.10), y + 12, int(W * 0.10) + 12, y + 24], fill=主)
            d.text((int(W * 0.10) + 26, y), s, font=f, fill=白)
    elif kind == "end":
        画标题(d, W, "声音，是长出来的", int(H * 0.33), 主, 86)
        f = 载字体(30)
        s = "AHI · meeting58 · 接口适配 / 两层解耦 / 动态挂载"
        d.text(((W - d.textlength(s, font=f)) / 2, int(H * 0.50)), s, font=f, fill=灰)
        s2 = "证据：meeting58/证据_接口与解耦.md（判据 26/26 PASS）"
        d.text(((W - d.textlength(s2, font=f)) / 2, int(H * 0.56)), s2, font=f, fill=灰)
    # 统一：波形 + 进度 + 水印
    画波形(d, 混, t, 主, W, H, 竖屏)
    d.rectangle([0, 0, int(W * (t / max(段["_总"], 1e-9))), 4], fill=主)
    fw = 载字体(22)
    d.text((int(W * 0.03), int(H * 0.035)), "AHI · meeting58", font=fw, fill=灰)
    return d_img


def 渲染视频(时间线, 混, 竖屏: bool, 文件名: str):
    W, H = (1080, 1920) if 竖屏 else (1920, 1080)
    # 预渲染每段背景
    for 段 in 时间线["段"]:
        段["_bg"] = 渐变背景(W, H, 段["主色"], 段["副色"])
        段["_总"] = 时间线["总时长"]
    总帧 = int(round(时间线["总时长"] * FPS))
    音频 = os.path.join(产物, "演示音频.wav")
    输出 = os.path.join(产物, 文件名)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(FPS), "-i", "-", "-i", 音频,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
           "-shortest", 输出]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(总帧):
        t = fi / FPS
        段 = None
        for s in 时间线["段"]:
            if s["start"] <= t < s["start"] + s["dur"]:
                段 = s; break
        if 段 is None:
            段 = 时间线["段"][-1]
        图 = 画界面(段, t - 段["start"] if False else t, 混, W, H, 竖屏)
        # 段首 0.25s 淡入（从黑）
        local = t - 段["start"]
        if local < 0.25:
            k = local / 0.25
            a = np.asarray(图, dtype=np.float64) * k
            图 = Image.fromarray(a.astype(np.uint8))
        p.stdin.write(图.tobytes())
    p.stdin.close()
    p.wait()
    return 输出


def 写srt(时间线, 路径):
    def ts(t):
        h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
        return "%02d:%02d:%06.3f" % (h, m, s)
    行 = []
    idx = 1
    for 段 in 时间线["段"]:
        if 段["kind"] == "role":
            for i, 句 in enumerate(段["句列"]):
                行.append("%d\n%s --> %s\n%s：%s\n" % (
                    idx, ts(段["句起"][i]).replace(".", ","),
                    ts(段["句止"][i]).replace(".", ","), 段["角色"]["角色"], 句["文本"]))
                idx += 1
        elif 段["kind"] == "montage":
            for i, 句 in enumerate(段["句列"]):
                行.append("%d\n%s --> %s\n%s：%s\n" % (
                    idx, ts(段["句起"][i]).replace(".", ","),
                    ts(段["句止"][i]).replace(".", ","), 句["角色"], 句["文本"]))
                idx += 1
    with open(路径, "w", encoding="utf-8") as f:
        f.write("\n".join(行))


def main():
    清单 = json.load(open(os.path.join(演示, "演示清单.json"), encoding="utf-8"))
    时间线 = 建时间线(清单)
    print("总时长 %.1fs，段数 %d" % (时间线["总时长"], len(时间线["段"])))
    混 = 建音频(时间线)
    写srt(时间线, os.path.join(产物, "演示字幕.srt"))
    h = 渲染视频(时间线, 混, False, "AHI_声线复制_横屏.mp4")
    print("横屏:", h)
    v = 渲染视频(时间线, 混, True, "AHI_声线复制_竖屏.mp4")
    print("竖屏:", v)
    json.dump(dict(总时长=时间线["总时长"], 横屏=h, 竖屏=v,
                   段=[{k: v for k, v in s.items() if not k.startswith("_")}
                       for s in 时间线["段"]]),
              open(os.path.join(产物, "视频清单.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
