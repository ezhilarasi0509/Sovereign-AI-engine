import json, numpy as np
from rouge_score import rouge_scorer
S=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True)
def load(p): return [json.loads(l) for l in open(p)]
bills=[json.loads(l)["text"] for l in open("data/test.jsonl")][:200]
P={"Base Qwen2.5-3B":"results/qwen2.5_3b.jsonl","Gemini zero-shot":"results/gemini-3.1-flash-lite.jsonl",
   "Gemini 2-shot":"results/gemini-3.1-flash-lite-2shot.jsonl","Fine-tuned 16-bit":"results/vllm_16bit_c8.jsonl",
   "Fine-tuned Q4_K_M":"results/billsum-q4b.jsonl"}
D={k:load(p) for k,p in P.items()}
rl={k:np.array([S.score(r["ref"],r["pred"])["rougeL"].fmeasure for r in v]) for k,v in D.items()}
L=np.array([len(b) for b in bills]); qs=np.quantile(L,[.25,.5,.75]); bins=np.digitize(L,qs)
names=["Shortest 25%","25-50%","50-75%","Longest 25%"]
heads=[f"{names[i]} ({int(L[bins==i].min()):,} to {int(L[bins==i].max()):,} chars)" for i in range(4)]
md="### 4.9 Where does it win or lose?\n\nROUGE-L by source bill length (quartiles of bill characters, about 50 bills each):\n\n"
md+="| System | "+" | ".join(heads)+" |\n|---|---|---|---|---|\n"
for k in P: md+=f"| {k} | "+" | ".join(f"{rl[k][bins==i].mean():.3f}" for i in range(4))+" |\n"
ft,g2=rl["Fine-tuned 16-bit"],rl["Gemini 2-shot"]
md+=f"\nPer-bill comparison, fine-tuned 16-bit against 2-shot Gemini: the fine-tuned model scores higher on {(ft>g2).mean()*100:.0f}% of bills, lower on {(ft<g2).mean()*100:.0f}%. "
md+="Averages hide variation, so see [docs/examples.md](docs/examples.md) for the three bills where the fine-tuned model gains most and the three where it loses most.\n\n"
d=ft-g2; order=np.argsort(d); sel=[("Largest gains for the fine-tuned model",order[::-1][:3]),("Largest losses for the fine-tuned model",order[:3])]
ex="# Example summaries (fine-tuned 16-bit versus 2-shot Gemini)\n\nTexts are cut to 700 characters. Chosen by the largest per-bill ROUGE-L difference, not hand-picked.\n\n"
for title,idx in sel:
    ex+=f"## {title}\n\n"
    for i in idx:
        i=int(i)
        ex+=f"### Test bill {i} ({len(bills[i]):,} characters), ROUGE-L fine-tuned {ft[i]:.3f}, 2-shot Gemini {g2[i]:.3f}\n\n"
        ex+="**Reference:** "+D["Fine-tuned 16-bit"][i]["ref"][:700].replace("\n"," ")+"\n\n"
        ex+="**Fine-tuned:** "+D["Fine-tuned 16-bit"][i]["pred"][:700].replace("\n"," ")+"\n\n"
        ex+="**2-shot Gemini:** "+D["Gemini 2-shot"][i]["pred"][:700].replace("\n"," ")+"\n\n"
open("docs/examples.md","w").write(ex); open("results/strat.md","w").write(md); print(md)
s=open("README.md").read()
if "### 4.9" not in s and "## 5. Which option should you pick?" in s:
    open("README.md","w").write(s.replace("## 5. Which option should you pick?",md+"## 5. Which option should you pick?")); print("README updated")
