# 黄队 meeting54 R1 探针（≤30s）：语料规模 + GPU 能力 + （可选）Qwen 装载/生成延迟
import json, time, os, sys

RK = r"<workspace>/02_新一代架构_RWKV7\RWKE"
out = {}

# 1) 语料规模：句数 / 去重句数 / 字符数（用「。」「\n」切分）
p = os.path.join(RK, "训练数据", "EmoCompanion_日记_全量.txt")
t0 = time.perf_counter()
raw = open(p, encoding="utf-8", errors="ignore").read()
chars = len(raw)
segs = [s.strip() for s in raw.replace("\r", "").split("\n")]
sents = []
for line in segs:
    for s in line.split("。"):
        s = s.strip()
        if s:
            sents.append(s)
uniq = len(set(sents))
out["语料"] = dict(MB=round(os.path.getsize(p) / 1e6, 2), 字符=chars,
                  行数=len(segs), 句数=len(sents), 去重句数=uniq,
                  平均句长=round(chars / max(1, len(sents)), 1),
                  重复句占比=round(1 - uniq / max(1, len(sents)), 3),
                  读取秒=round(time.perf_counter() - t0, 2))

# 2) GPU 与显存
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out["device"] = dict(name=torch.cuda.get_device_name(0),
                             total_MiB=round(total / 2**20, 0), free_MiB=round(free / 2**20, 0))
        # 3) fp16 稠密算力小测（2048^3 × 20 次）
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 20
        for _ in range(N):
            a @ b
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        flops = 2 * 2048**3 * N
        out["fp16_TFLOPS"] = round(flops / dt / 1e12, 2)
        del a, b
        torch.cuda.empty_cache()
except Exception as e:
    out["torch_err"] = repr(e)[:200]

print(json.dumps(out, ensure_ascii=False, indent=1))
