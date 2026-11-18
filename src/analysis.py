import json, re, numpy as np
from rouge_score import rouge_scorer
S=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True)
def load(p): return [json.loads(l) for l in open(p)]
bills=[json.loads(l)["text"] for l in open("data/test.jsonl")][:200]
SYS={"base":"results/qwen2.5_3b.jsonl","gemini":"results/gemini-3.1-flash-lite.jsonl",
     "ft16":"results/vllm_16bit_c8.jsonl","q4":"results/billsum-q4b.jsonl"}
D={k:load(p) for k,p in SYS.items()}
per={k:np.array([[S.score(r["ref"],r["pred"])[m].fmeasure for m in ["rouge1","rouge2","rougeL"]] for r in v]) for k,v in D.items()}
rng=np.random.default_rng(0)
def ci(a,b,col):
    d=a[:,col]-b[:,col]; n=len(d)
    bs=[d[rng.integers(0,n,n)].mean() for _ in range(5000)]
    return d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
print("Paired bootstrap, mean difference and 95% CI (ROUGE-L):")
for a,b in [("ft16","gemini"),("ft16","base"),("ft16","q4")]:
    m,lo,hi=ci(per[a],per[b],2); print(f"  {a} - {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]", "significant" if lo>0 or hi<0 else "NOT significant")
print("\nPer-system ROUGE-1/2/L with 95% bootstrap CI:")
for k,v in per.items():
    out=[]
    for c in range(3):
        bs=[v[rng.integers(0,200,200),c].mean() for _ in range(2000)]
        out.append(f"{v[:,c].mean():.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
    print(" ",k," | ".join(out))
print("\nTruncation (out_tok >= 299) and average words:")
for k,v in D.items():
    print(f"  {k}: truncated {np.mean([r['out_tok']>=299 for r in v])*100:.1f}%  avg words {np.mean([len(r['pred'].split()) for r in v]):.0f}  ref words {np.mean([len(r['ref'].split()) for r in v]):.0f}")
num=re.compile(r"\$?\d[\d,]*\.?\d*")
def norm(x): return x.replace(",","").rstrip(".")
print("\nNumbers in the summary that do NOT appear in the bill (lower is better):")
for k,v in D.items():
    tot=bad=0
    for r,b in zip(v,bills):
        bn=norm(b)
        for m in num.findall(r["pred"]):
            t=norm(m.lstrip("$"))
            if len(t)<2: continue
            tot+=1; bad+= t not in bn
    print(f"  {k}: {bad}/{tot} = {bad/max(tot,1)*100:.1f}%")
