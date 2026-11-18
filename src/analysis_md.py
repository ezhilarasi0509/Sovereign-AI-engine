import json, re, numpy as np
from rouge_score import rouge_scorer
S=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True)
def load(p): return [json.loads(l) for l in open(p)]
bills=[json.loads(l)["text"] for l in open("data/test.jsonl")][:200]
N={"base":"Base Qwen2.5-3B","gemini":"Gemini 3.1 Flash-Lite","ft16":"Fine-tuned 16-bit","q4":"Fine-tuned Q4_K_M"}
P={"base":"results/qwen2.5_3b.jsonl","gemini":"results/gemini-3.1-flash-lite.jsonl","ft16":"results/vllm_16bit_c8.jsonl","q4":"results/billsum-q4b.jsonl"}
D={k:load(p) for k,p in P.items()}
def per(rows,trunc=None):
    return np.array([[S.score(r["ref"]," ".join(r["pred"].split()[:trunc]) if trunc else r["pred"])[m].fmeasure
                      for m in ["rouge1","rouge2","rougeL"]] for r in rows])
rng=np.random.default_rng(0)
def ci(v,c):
    bs=[v[rng.integers(0,len(v),len(v)),c].mean() for _ in range(3000)]
    return v[:,c].mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
def pair(a,b,c=2):
    d=a[:,c]-b[:,c]; bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(5000)]
    lo,hi=np.percentile(bs,[2.5,97.5]); return d.mean(),lo,hi,("significant" if lo>0 or hi<0 else "not significant")
def wilson(k,n):
    p=k/n; z=1.96; d=1+z*z/n; c=p+z*z/(2*n); a=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-a)/d,(c+a)/d
full={k:per(v) for k,v in D.items()}
cut={k:per(v,200) for k,v in D.items()}
md="### 4.7 Statistical checks\n\n"
md+="**Bootstrap 95% confidence intervals** (200 bills, 3,000 resamples):\n\n| System | ROUGE-1 | ROUGE-2 | ROUGE-L |\n|---|---|---|---|\n"
for k in N:
    md+=f"| {N[k]} | "+" | ".join("%.3f [%.3f, %.3f]"%ci(full[k],c) for c in range(3))+" |\n"
md+="\n**Paired differences in ROUGE-L** (same bills, bootstrap):\n\n| Comparison | Mean difference | 95% CI | Verdict |\n|---|---|---|---|\n"
for a,b in [("ft16","gemini"),("ft16","base"),("ft16","q4")]:
    m,lo,hi,vd=pair(full[a],full[b]); md+=f"| {N[a]} minus {N[b]} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {vd} |\n"
md+="\nThe fine-tuned model's lead over Gemini and the base model is statistically supported. The 16-bit versus Q4_K_M gap is not distinguishable from noise on 200 bills.\n\n"
md+="**Length control.** Output caps differed: the 16-bit and Q4 runs were capped at 300 tokens, while base Qwen and Gemini had no cap. The share of outputs that hit 299 or more tokens, plus a rescoring with every output cut to its first 200 words (the length the prompt asked for):\n\n| System | Hit 300-token cap | Avg words (reference 151) | ROUGE-L, full output | ROUGE-L, first 200 words |\n|---|---|---|---|---|\n"
for k in N:
    md+=f"| {N[k]} | {np.mean([r['out_tok']>=299 for r in D[k]])*100:.1f}% | {np.mean([len(r['pred'].split()) for r in D[k]]):.0f} | {full[k][:,2].mean():.3f} | {cut[k][:,2].mean():.3f} |\n"
m,lo,hi,vd=pair(cut["ft16"],cut["gemini"])
md+=f"\nAfter the 200-word cut, fine-tuned minus Gemini is {m:+.3f} ROUGE-L [{lo:+.3f}, {hi:+.3f}], {vd}.\n\n"
md+="**Numbers not found in the bill.** A crude grounding check: every number in a summary (2 or more characters) is searched for in the source bill. Misses include legitimate rewrites such as totals or spelled-out numbers, so only compare systems with each other.\n\n| System | Numbers checked | Not in bill | Rate (95% Wilson CI) |\n|---|---|---|---|\n"
num=re.compile(r"\$?\d[\d,]*\.?\d*")
for k in N:
    tot=bad=0
    for r,b in zip(D[k],bills):
        bn=b.replace(",","")
        for m_ in num.findall(r["pred"]):
            t=m_.lstrip("$").replace(",","").rstrip(".")
            if len(t)<2: continue
            tot+=1; bad+=t not in bn
    lo,hi=wilson(bad,tot); md+=f"| {N[k]} | {tot} | {bad} | {bad/tot*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%] |\n"
md+="\nFine-tuning improved style and overlap but showed no evidence of better factual grounding on this check: the 16-bit model had the highest rate of unmatched numbers, though its interval overlaps the others, so this is suggestive rather than conclusive. Verify facts before using any summary in a legal setting.\n\n"
open("results/analysis.md","w").write(md); print(md)
s=open("README.md").read()
if "### 4.7" not in s and "## 5. Which option should you pick?" in s:
    s=s.replace("## 5. Which option should you pick?",md+"## 5. Which option should you pick?"); open("README.md","w").write(s); print("README updated")
