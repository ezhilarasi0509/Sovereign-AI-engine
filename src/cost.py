import json, math, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
GPU_HR=float(sys.argv[1]) if len(sys.argv)>1 else 0.50   # assumed $/hour for a T4-class cloud GPU
IN_P,OUT_P=0.25,1.50                                       # Gemini 3.1 Flash-Lite $/1M tokens (standard)
rows=[json.loads(l) for l in open("results/gemini-3.1-flash-lite.jsonl")]
ain=np.mean([r["in_tok"] for r in rows]); aout=np.mean([r["out_tok"] for r in rows])
api_per_1k=1000*(ain*IN_P+aout*OUT_P)/1e6
s=json.load(open("results/vllm_16bit_summary.json"))
HRS=730
fixed=GPU_HR*HRS
print(f"Gemini: avg_in {ain:.0f} avg_out {aout:.0f}  ->  ${api_per_1k:.3f} per 1,000 calls")
cap={}
for c in ["8","32"]:
    rps=s[c]["req_per_s"]; per_hr=rps*3600
    cap[c]=per_hr*HRS
    print(f"vLLM T4 conc {c}: {rps} req/s, p95 {s[c]['p95']}s, ${GPU_HR/per_hr*1000:.3f} per 1,000 calls at full load, capacity {cap[c]:,.0f} calls/month")
print(f"Self-host fixed cost: ${fixed:.0f}/month at ${GPU_HR}/hr")
cross=fixed/api_per_1k*1000
print(f"Crossover: {cross:,.0f} calls/month ({cross/30:,.0f}/day)")
x=np.logspace(3,7,200)
api=x/1000*api_per_1k
selfh=np.array([max(1,math.ceil(v/cap["32"]))*fixed for v in x])
plt.figure(figsize=(7,4))
plt.loglog(x,api,label="Gemini API"); plt.loglog(x,selfh,label=f"Self-hosted T4 (${GPU_HR}/hr)")
plt.axvline(cross,ls="--",c="gray")
plt.xlabel("calls per month"); plt.ylabel("USD per month"); plt.legend(); plt.grid(True,which="both",alpha=.3)
plt.savefig("results/crossover.png",dpi=150,bbox_inches="tight")
print("saved results/crossover.png")
