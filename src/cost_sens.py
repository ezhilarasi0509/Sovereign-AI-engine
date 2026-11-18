import json, numpy as np
IN_P,OUT_P=0.25,1.50; HRS=730
gr=[json.loads(l) for l in open("results/gemini-3.1-flash-lite.jsonl")]
ain=np.mean([r["in_tok"] for r in gr]); aout=np.mean([r["out_tok"] for r in gr])
api1k=1000*(ain*IN_P+aout*OUT_P)/1e6
rps=json.load(open("results/vllm_16bit_summary.json"))["32"]["req_per_s"]
md="### 4.8 Cost sensitivity\n\n"
md+=f"The GPU price and how busy the GPU is both change the answer. Self-hosted cost per 1,000 calls (using the measured {rps} requests/s at concurrency 32), against an API cost of ${api1k:.3f} per 1,000 calls:\n\n"
md+="| GPU $/hr | 100% busy | 50% busy | 25% busy | 10% busy | Crossover (calls/month) |\n|---|---|---|---|---|---|\n"
for p in [0.25,0.35,0.50,0.75,1.00]:
    row=[p/(rps*3600*u)*1000 for u in [1,.5,.25,.1]]
    md+=f"| ${p:.2f} | "+" | ".join(f"${x:.2f}" for x in row)+f" | {p*HRS/api1k*1000:,.0f} |\n"
md+=f"\n- A GPU that is only 10% busy costs more per call than the API at every price above. Self-hosting needs steady volume.\n- With the API's 50% batch discount (${api1k/2:.3f} per 1,000 calls) the crossover at $0.50/hr doubles to about {0.5*HRS/(api1k/2)*1000:,.0f} calls/month.\n- Prices are assumptions, not quotes. Check current GPU and API rates before deciding.\n\n"
open("results/cost_sens.md","w").write(md); print(md)
s=open("README.md").read()
if "### 4.8" not in s and "## 5. Which option should you pick?" in s:
    open("README.md","w").write(s.replace("## 5. Which option should you pick?",md+"## 5. Which option should you pick?")); print("README updated")
