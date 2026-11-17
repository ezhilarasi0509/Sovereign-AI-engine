import json, os, numpy as np
from rouge_score import rouge_scorer
GPU_HR=0.50; IN_P,OUT_P=0.25,1.50
S=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True)
SYS=[("Base Qwen2.5-3B (Ollama, Mac)","results/qwen2.5_3b.jsonl",0.813),
     ("Gemini 3.1 Flash-Lite (API)","results/gemini-3.1-flash-lite.jsonl",0.820),
     ("Fine-tuned 16-bit (vLLM, T4)","results/vllm_16bit_c8.jsonl",0.852),
     ("Fine-tuned Q4_K_M GGUF (Ollama, Mac)","results/billsum-q4b.jsonl",0.846)]
def load(p): return [json.loads(l) for l in open(p)]
def rouge(rows):
    sc=[S.score(r["ref"],r["pred"]) for r in rows]
    return [float(np.mean([x[k].fmeasure for x in sc])) for k in ["rouge1","rouge2","rougeL"]]
res={n:(rouge(load(p)),b,load(p)) for n,p,b in SYS}
ft=res["Fine-tuned 16-bit (vLLM, T4)"][0]; q4=res["Fine-tuned Q4_K_M GGUF (Ollama, Mac)"][0]
base=res["Base Qwen2.5-3B (Ollama, Mac)"][0]; gem=res["Gemini 3.1 Flash-Lite (API)"][0]
gr=load("results/gemini-3.1-flash-lite.jsonl")
ain=np.mean([r["in_tok"] for r in gr]); aout=np.mean([r["out_tok"] for r in gr])
api1k=1000*(ain*IN_P+aout*OUT_P)/1e6
v=json.load(open("results/vllm_16bit_summary.json")); fixed=GPU_HR*730
cross=fixed/api1k*1000
def judge_tbl():
    for f,label in [("results/judge_strong.json","strict rubric, gemini-3.7-flash judge"),("results/judge_flashlite.json","lenient rubric, flash-lite judge")]:
        if os.path.exists(f):
            j=json.load(open(f))
            t=f"LLM judge ({label}, 50 bills, scores 1-5):\n\n| System | Faithfulness | Coverage |\n|---|---|---|\n"
            for k,n in [("base_qwen","Base Qwen"),("gemini","Gemini Flash-Lite"),("finetuned_16bit","Fine-tuned 16-bit"),("q4_k_m","Fine-tuned Q4_K_M")]:
                t+=f"| {n} | {j[k]['faithfulness']} | {j[k]['coverage']} |\n"
            return t
    return ""
rows="".join(f"| {n} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | {b:.3f} |\n" for n,(r,b,_) in res.items())
md=f"""# Sovereign AI engine: summarizing US bills with a self-hosted open-weight model

Question: **should a team pay for a frontier API, or fine-tune and run a small open model on its own hardware?**
I answered it on BillSum (US congressional bills) with measured quality, latency, throughput and cost.

## Result
- Fine-tuning a 3B model (LoRA, 1,500 bills, one free T4 epoch) lifted ROUGE-2 from **{base[1]:.3f} to {ft[1]:.3f}** and beat the Gemini 3.1 Flash-Lite API baseline ({gem[1]:.3f}).
- 4-bit quantization (Q4_K_M) cut the model from 5.8 GB to 1.8 GB and cost **{ft[0]-q4[0]:.3f} ROUGE-1 / {ft[2]-q4[2]:.3f} ROUGE-L**, still above both baselines.
- Self-hosting beats the API on cost above about **{cross:,.0f} calls/month** (assumes a ${GPU_HR}/hr T4-class GPU running 24/7).

## Quality (200 BillSum test bills)
| System | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1* |
|---|---|---|---|---|
{rows}
*BERTScore used distilbert-base-uncased, so compare only within this table.

{judge_tbl()}
## Latency, throughput, cost (fine-tuned 16-bit on vLLM, Kaggle T4)
| Concurrency | P50 (s) | P95 (s) | Tokens/s | Requests/s |
|---|---|---|---|---|
""" + "".join(f"| {c} | {v[c]['p50']} | {v[c]['p95']} | {v[c]['tok_per_s']} | {v[c]['req_per_s']} |\n" for c in ["1","8","32"]) + f"""
- Gemini API: avg {ain:.0f} input + {aout:.0f} output tokens per call = **${api1k:.3f} per 1,000 calls** ($0.25 / $1.50 per 1M tokens, standard rate).
- Self-hosted T4 at full load: **${GPU_HR/(v['8']['req_per_s']*3600)*1000:.3f}** (concurrency 8) to **${GPU_HR/(v['32']['req_per_s']*3600)*1000:.3f}** (concurrency 32) per 1,000 calls, but the GPU costs ${fixed:.0f}/month whether used or not.
- Crossover: ~{cross:,.0f} calls/month (~{cross/30:,.0f}/day). See `results/crossover.png`.

![crossover](results/crossover.png)

## Why fine-tune at all
The base 3B model followed the task loosely (ROUGE-2 {base[1]:.3f}). Training on 1,500 expert summaries taught the legislative summary style and length, adding {ft[1]-base[1]:.3f} ROUGE-2.

## What quantization cost
| Format | Size | ROUGE-1 | ROUGE-L |
|---|---|---|---|
| fp16 | 5.8 GB | {ft[0]:.3f} | {ft[2]:.3f} |
| Q8_0 | 3.1 GB | not evaluated | not evaluated |
| Q4_K_M | 1.8 GB | {q4[0]:.3f} | {q4[2]:.3f} |

## Stack
Unsloth + PEFT (LoRA), vLLM (T4 serving), llama.cpp (GGUF), Ollama, FastAPI, Docker, Arize Phoenix tracing.

## Run it
```bash
pip install -r requirements.txt
python scripts/prepare_data.py          # BillSum subsets
python src/run_ollama.py 200 MODEL      # local model baseline
python src/run_gemini.py 200            # API baseline (GEMINI_API_KEY in .env)
python src/score.py results/*.jsonl
python src/cost.py 0.50                 # cost + crossover (arg = GPU $/hr)
docker build -t sovereign-ai-api api && docker run -p 8080:8080 --env-file .env sovereign-ai-api
```

## Limitations (read before trusting the numbers)
- One run per system on 200 bills, no confidence intervals or seeds comparison.
- ROUGE/BERTScore reward matching the reference style; the LLM judge is also noisy and Gemini graded a system that includes itself.
- Latencies are not comparable across systems: Q4 ran on a MacBook M4 via Ollama, 16-bit on a Kaggle T4 via vLLM with `--enforce-eager`, Gemini over the network. Use the vLLM table for serving claims.
- The GPU price (${GPU_HR}/hr) is an assumption; the API price is the list rate at time of writing and excludes batch discounts.
- Q8_0 was built but not evaluated. PubMedQA was not run.
"""
open("README.md","w").write(md); print("README.md written,",len(md),"chars")
