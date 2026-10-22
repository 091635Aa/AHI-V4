import json,sys
a=json.load(open(r"<workspace>/03_AHI\AHI4\meeting63\产物\判据\判据_发音_v8.json",encoding="utf-8"))
b=json.load(open(r"<workspace>/03_AHI\AHI4\meeting64\产物\判据\判据_发音_v9.json",encoding="utf-8"))
print("v8 通过/臂数 = %s/%s | v9 = %s/%s"%(a["通过"],a["臂数"],b["通过"],b["臂数"]))
print("v8 未过 =",a["未过"]); print("v9 未过 =",b["未过"])
差=[]
k8=set(a["臂"]); k9=set(b["臂"])
if k8!=k9: 差.append(("臂名集合",sorted(k8^k9)))
for k in sorted(k8&k9):
    if (a["臂"][k]["通过"],a["臂"][k]["读数"])!=(b["臂"][k]["通过"],b["臂"][k]["读数"]):
        差.append((k,a["臂"][k]["通过"],b["臂"][k]["通过"]))
print("逐臂（通过+读数）差异数 =",len(差))
for d in 差[:10]: print("  ",d)
print("must_fail v8/v9 =",a["must_fail"],b["must_fail"])
print("准入门 v8/v9 =",a["准入门"],b["准入门"])
print("句级 E2 差 v8=",a["臂"]["E2_清浊电平差"]["读数"]," v9=",b["臂"]["E2_清浊电平差"]["读数"])
