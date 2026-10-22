import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import torch
torch.set_num_threads(1)
torch.cuda.init()
torch.cuda.synchronize()
print("READY pid=%d alloc=%d res=%d" % (
    __import__("os").getpid(),
    torch.cuda.memory_allocated(), torch.cuda.memory_reserved()), flush=True)
time.sleep(25)
