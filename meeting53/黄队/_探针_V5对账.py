# 黄队 R2 对账探针（≤30s）：量 V5 真实每拍代价 / 构造成本 / 3 脑代价
# 只读 import 主控的 V5_双通道；不写字节码；不改任何文件
import sys, json, time

sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting53\实施")

t_imp = time.perf_counter()
import V5_双通道 as V5
t_imp = time.perf_counter() - t_imp

t0 = time.perf_counter()
体 = V5.双通道体(D=64, 种子=53001)
t_build = time.perf_counter() - t0

t0 = time.perf_counter()
r1 = V5.跑单脑(拍数=200, 测拍=50, 种子=53001)
t1 = time.perf_counter() - t0

t0 = time.perf_counter()
r3 = V5.跑多脑(拍数=100, 测拍=50, 种子=53001)
t3 = time.perf_counter() - t0

t0 = time.perf_counter()
r192 = V5.跑多脑(拍数=100, 测拍=50, 种子=53001, 部分观测单脑=True, D=192)
t192 = time.perf_counter() - t0

print(json.dumps({
    "import_s": round(t_imp, 3),
    "build1_s": round(t_build, 4),
    "单脑_250拍_s": round(t1, 3),
    "单脑_ms每拍": round(t1 / 250 * 1000, 3),
    "多脑3_150拍_s": round(t3, 3),
    "多脑_ms每拍(3脑)": round(t3 / 150 * 1000, 3),
    "多脑D192_150拍_s": round(t192, 3),
    "多脑D192_ms每拍": round(t192 / 150 * 1000, 3),
    "单脑训练成功率": r1["训练成功率"],
    "多脑3测试": r3["测试成功率"],
    "单脑部分观测D192测试": r192["测试成功率"],
}, ensure_ascii=False))
