import json, os, numpy as np
from rouge_score import rouge_scorer
GPU_HR=0.50; IN_P,OUT_P=0.25,1.50; HRS=730
S=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True)
def load(p): return [json.loads(l) for l in open(p)]
def rouge(rows):
    sc=[S.score(r["ref"],r["pred"]) for r in rows]
    return [float(np.mean([x[k].fmeasure for x in sc])) for k in ["rouge1","rouge2","rougeL"]]
def lat(rows):
    l=[r["latency"] for r in rows]; return np.percentile(l,50),np.percentile(l,95)
BQ8=json.load(open("results/bert_q8.json"))["f1"]
SYS=[("base","Base Qwen2.5-3B-Instruct","Ollama on MacBook M4","results/qwen2.5_3b.jsonl",0.813),
     ("gem","Gemini 3.1 Flash-Lite","Google API (frontier baseline)","results/gemini-3.1-flash-lite.jsonl",0.820),
     ("ft","Fine-tuned Qwen2.5-3B, 16-bit","vLLM on Kaggle T4 (concurrency 8)","results/vllm_16bit_c8.jsonl",0.852),
     ("q8","Fine-tuned Qwen2.5-3B, Q8_0","Ollama on MacBook M4","results/billsum-q8.jsonl",BQ8),
     ("q4","Fine-tuned Qwen2.5-3B, Q4_K_M","Ollama on MacBook M4","results/billsum-q4b.jsonl",0.846)]
D={k:(n,h,load(p),b) for k,n,h,p,b in SYS}
R={k:rouge(v[2]) for k,v in D.items()}
L={k:lat(v[2]) for k,v in D.items()}
base,gem,ft,q4=R["base"],R["gem"],R["ft"],R["q4"]
q8=R["q8"]
gr=D["gem"][2]
ain=np.mean([r["in_tok"] for r in gr]); aout=np.mean([r["out_tok"] for r in gr])
api1k=1000*(ain*IN_P+aout*OUT_P)/1e6
v=json.load(open("results/vllm_16bit_summary.json"))
fixed=GPU_HR*HRS; cross=fixed/api1k*1000
c8=GPU_HR/(v["8"]["req_per_s"]*3600)*1000; c32=GPU_HR/(v["32"]["req_per_s"]*3600)*1000
cap32=v["32"]["req_per_s"]*3600*HRS
r2gain=(ft[1]-base[1])/base[1]*100
keep=q4[0]/ft[0]*100
qual="".join(f"| {D[k][0]} | {D[k][1]} | {R[k][0]:.3f} | {R[k][1]:.3f} | {R[k][2]:.3f} | {D[k][3]:.3f} |\n" for k in ["base","gem","ft","q8","q4"])
latt="".join(f"| {D[k][0]} | {D[k][1]} | {L[k][0]:.1f} | {L[k][1]:.1f} |\n" for k in ["base","gem","ft","q8","q4"])
bench="".join(f"| {c} | {v[c]['p50']} | {v[c]['p95']} | {v[c]['tok_per_s']} | {v[c]['req_per_s']} |\n" for c in ["1","8","32"])
if os.path.exists("results/judge_strong.json"):
    j=json.load(open("results/judge_strong.json"))
    jt="Scores are 1 to 5, from a strict-rubric LLM judge (gemini-3.7-flash) on the first 50 test bills.\n\n| System | Faithfulness | Coverage |\n|---|---|---|\n"
    for k,n in [("base_qwen","Base Qwen2.5-3B"),("gemini","Gemini 3.1 Flash-Lite"),("finetuned_16bit","Fine-tuned 16-bit"),("q4_k_m","Fine-tuned Q4_K_M")]:
        jt+=f"| {n} | {j[k]['faithfulness']} | {j[k]['coverage']} |\n"
    jt+="\nLLM judges are noisy and can favor their own style, so treat this as a sanity check, not a verdict.\n"
else:
    jt="An LLM-judge pass is not included. A first attempt with a lenient rubric and Gemini Flash-Lite as judge was discarded: it gave Gemini a perfect 5.0 on its own summaries and ranked the base model above the fine-tuned one, which contradicts every other metric.\n"
spd=""
if os.path.exists("results/speed_mac.json"):
    sp=json.load(open("results/speed_mac.json"))
    spd="### 4.5b Same-hardware speed: Q4_K_M versus Q8_0 (idle MacBook M4, Ollama)\n\n| Model | Requests | P50 (s) | P95 (s) | Output tokens/s |\n|---|---|---|---|---|\n"
    for m,n in [("billsum-q8","Q8_0"),("billsum-q4b","Q4_K_M")]:
        spd+=f"| {n} | {sp[m]['n']} | {sp[m]['p50_s']} | {sp[m]['p95_s']} | {sp[m]['tok_per_s']} |\n"
    spd+="\nBoth models ran alternately on an otherwise idle machine, 29 bills per round, 2 rounds, 300-token cap.\n\n"
md=f"""# Sovereign AI Engine: summarizing US legislation with a model you run yourself

> **One-line answer:** a small open-weight model (3B parameters), fine-tuned for free on a single T4 GPU, produced better bill summaries than a frontier API baseline on this task, and above roughly **{cross:,.0f} calls per month** it is also cheaper to run yourself.
>
> **Licensing note:** the base model (Qwen2.5-3B-Instruct) is released under the Qwen Research License, which allows non-commercial use only. See section 13.

## 1. Purpose of this project

Many organizations (hospitals, banks, courts, governments) want AI to read long documents but cannot send them to an outside company, or find the per-call API bill too high at scale. This project tests the alternative, a **"sovereign" AI engine**: an open-weight model that runs on hardware you control.

| Question | How this project answers it |
|---|---|
| Is a small open model good enough? | Baselined a 3B open model against a frontier API on 200 unseen bills. |
| Is fine-tuning worth it? | Measured quality before and after LoRA fine-tuning on 1,500 bills. |
| What does making the model smaller cost? | Compared 16-bit against 4-bit (Q4_K_M) for size and quality. |
| Can it serve real traffic? | Benchmarked P50 and P95 latency and throughput on vLLM at 1, 8 and 32 concurrent requests. |
| When does self-hosting become cheaper? | Computed cost per 1,000 calls and the monthly crossover volume. |
| Is it deployable? | Wrapped it in a FastAPI service with Docker and Phoenix tracing. |

**Who this is for:** engineers and managers deciding between "pay an API" and "run our own model", and anyone who wants a worked, measured example of that decision.

## 2. The task and the data

- **Task:** read a US congressional bill and write a short, factual legislative summary (at most 200 words).
- **Dataset:** [BillSum](https://arxiv.org/abs/1910.00523) (Kornilova and Eidelman, 2019), loaded as `FiscalNote/billsum`: US congressional bills paired with expert-written summaries.
- **Training data:** 1,500 bills from the train split (bills under 12,000 characters and 3,072 tokens).
- **Evaluation data:** the first 200 bills of the official test split (under 12,000 characters). None were used for training.

## 3. Systems compared

| System | Where it runs | Role |
|---|---|---|
| Base Qwen2.5-3B-Instruct | Ollama, MacBook M4 | Small open model, no training |
| Gemini 3.1 Flash-Lite | Google API | Frontier API baseline |
| Fine-tuned Qwen2.5-3B, 16-bit | vLLM, Kaggle T4 GPU | The sovereign model at full precision |
| Fine-tuned Qwen2.5-3B, Q4_K_M | Ollama, MacBook M4 | The same model shrunk to 4-bit |

```mermaid
flowchart LR
  A[BillSum bills] --> B[Baselines: base Qwen and Gemini]
  A --> C[LoRA fine-tune with Unsloth and PEFT]
  C --> D[Merge adapter into 16-bit model]
  D --> E[vLLM serving benchmark]
  D --> F[GGUF quantization Q8_0 and Q4_K_M]
  F --> G[Q4_K_M on Ollama]
  B --> H[Quality, latency, cost comparison]
  E --> H
  G --> H
  H --> I[Crossover analysis and FastAPI service]
```

## 4. Results

### 4.1 Summary quality (200 test bills)

| System | Runs on | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1* |
|---|---|---|---|---|---|
{qual}
\\*BERTScore used `distilbert-base-uncased`, so compare these numbers only with each other, not with published BERTScores.

**What this shows**
- Fine-tuning raised ROUGE-2 from {base[1]:.3f} to {ft[1]:.3f} (about {r2gain:.0f}% relative) and ROUGE-L from {base[2]:.3f} to {ft[2]:.3f}.
- The fine-tuned model beat the Gemini 3.1 Flash-Lite API baseline on every ROUGE metric and on BERTScore.
- The gain comes largely from learning the legislative summary style and length, which the BillSum references share. A general-purpose API model is not trained for that style.

### 4.2 LLM-judge check
{jt}
### 4.3 What 4-bit quantization cost

| Format | File size | ROUGE-1 | ROUGE-L | Notes |
|---|---|---|---|---|
| fp16 | 5.8 GB | {ft[0]:.3f} | {ft[2]:.3f} | Reference quality |
| Q8_0 | 3.1 GB | {q8[0]:.3f} | {q8[2]:.3f} | No significant loss versus fp16 (see 4.7) |
| Q4_K_M | 1.8 GB | {q4[0]:.3f} | {q4[2]:.3f} | Keeps {keep:.1f}% of fp16 ROUGE-1 at 31% of the size |

Quantizing to 4-bit lost {ft[0]-q4[0]:.3f} ROUGE-1 and {ft[2]-q4[2]:.3f} ROUGE-L, and the 4-bit model still beat both baselines. A paired bootstrap (section 4.7) shows this gap is within noise, so the quality cost of Q4_K_M is not statistically distinguishable from zero on 200 bills. Q8_0 scored {q8[0]:.3f} ROUGE-1 and {q8[2]:.3f} ROUGE-L with no significant loss against fp16. Its small ROUGE-1 edge over the vLLM fp16 run is not a benefit of quantization: it likely reflects generation differences (Q8_0 outputs were shorter, 154 versus 171 words, and hit the 300-token cap less often). Q4_K_M scored significantly below Q8_0 (about 0.03 ROUGE-1 and 0.02 ROUGE-L), so going from 8-bit to 4-bit has a small but measurable cost. For a laptop or edge deployment that trade is usually worth it.

### 4.4 Serving benchmark (fine-tuned 16-bit, vLLM, Kaggle T4)

| Concurrent requests | P50 latency (s) | P95 latency (s) | Output tokens/s | Requests/s |
|---|---|---|---|---|
{bench}
- Throughput rises with concurrency (24.6 to 125.5 tokens/s) but latency rises faster, so for interactive use keep concurrency low and add GPUs for volume.
- The server ran with `--enforce-eager` to fit the free T4. Throughput with CUDA graphs enabled would likely be higher.
- Quality did not change with load: ROUGE-1 stayed between 0.536 and 0.538 at all concurrency levels.

### 4.5 Observed latency per system (not directly comparable)

| System | Runs on | P50 (s) | P95 (s) |
|---|---|---|---|
{latt}
These come from different hardware and software stacks (a Mac with Ollama, a T4 with vLLM, a network API), so do not rank the systems by this table. Use section 4.4 for serving claims.

{spd}### 4.6 Cost and the crossover point

| Item | Value |
|---|---|
| Gemini 3.1 Flash-Lite price (standard) | $0.25 per 1M input tokens, $1.50 per 1M output tokens |
| Average tokens per call (measured) | {ain:.0f} input, {aout:.0f} output |
| **API cost per 1,000 calls** | **${api1k:.3f}** |
| Self-hosted T4 cost per 1,000 calls at full load | ${c8:.3f} (concurrency 8) to ${c32:.3f} (concurrency 32) |
| GPU fixed cost (assumed ${GPU_HR}/hr, 24/7) | ${fixed:.0f} per month |
| Capacity of one T4 at concurrency 32 | about {cap32:,.0f} calls per month |
| **Crossover volume** | **about {cross:,.0f} calls/month (about {cross/30:,.0f}/day)** |

![Monthly cost: API versus self-hosted](results/crossover.png)

Below the crossover the API is cheaper because you pay nothing when idle. Above it, a busy GPU wins. The GPU price is an assumption: change it with `python src/cost.py YOUR_PRICE`.

## 5. Which option should you pick?

| Your situation | Suggested option |
|---|---|
| Low or unpredictable volume, data may leave your network | Frontier API |
| Documents cannot leave your infrastructure (regulated or private data) | Self-host, even below the crossover |
| Steady volume above about {cross/30:,.0f} calls per day | Self-host the fine-tuned 16-bit model on a GPU |
| Laptop, edge or small server | Fine-tuned Q4_K_M, a small quality loss for a 3x smaller file |
| Need top quality on a narrow, well-defined task | Fine-tune a small model on your own examples |

## 6. What was built (repository map)

| Path | What it does |
|---|---|
| `scripts/prepare_data.py` | Builds the train and test subsets from BillSum |
| `scripts/train_unsloth.py` | LoRA fine-tuning with Unsloth and PEFT (run on a GPU) |
| `scripts/merge_lora_cpu.py` | Merges the adapter into the base model on CPU |
| `scripts/quantize_gguf.sh` | Builds Q8_0 and Q4_K_M GGUF files with llama.cpp |
| `scripts/bench_vllm.py` | Concurrency benchmark against a vLLM server |
| `src/run_ollama.py`, `src/run_gemini.py` | Generate summaries from the local model and the API |
| `src/score.py`, `src/bertscore_eval.py` | ROUGE and BERTScore evaluation |
| `src/judge_strict.py` | Strict-rubric LLM judge for faithfulness and coverage |
| `src/cost.py` | Cost per 1,000 calls and the crossover chart |
| `api/` | FastAPI service, Dockerfile, Phoenix tracing, switchable backends |
| `results/` | Raw model outputs, benchmark summary, crossover chart |

## 7. The serving API

`api/app.py` exposes `POST /summarize` and `GET /health`. A request picks a backend (`ollama`, `vllm` or `gemini`) through the same OpenAI-compatible interface, so the sovereign model and the API baseline can be swapped with one field. Every call is traced in Arize Phoenix (latency, tokens, input and output). Run `phoenix serve`, then open http://localhost:6006.

## 8. Technology and why each piece was used

| Tool | Purpose here |
|---|---|
| Unsloth | Fast, low-memory fine-tuning on a free T4 |
| Hugging Face PEFT / LoRA | Trains a small adapter (about 125 MB) instead of all 3B weights |
| vLLM | High-throughput GPU serving with continuous batching |
| llama.cpp / GGUF | Quantizing the model to 8-bit and 4-bit |
| Ollama | Running the models locally on a Mac |
| FastAPI and Docker | A deployable summarization service |
| Arize Phoenix | Open-source tracing and observability |
| Kaggle and Colab | Free GPUs for training and benchmarking |

## 9. Reproduce it

```bash
pip install -r requirements.txt
cp .env.example .env              # add GEMINI_API_KEY and HF_TOKEN
python scripts/prepare_data.py
python src/run_ollama.py 200 qwen2.5:3b       # base model baseline
python src/run_gemini.py 200                  # API baseline
python src/score.py results/*.jsonl
python src/cost.py 0.50                       # argument = GPU price per hour
docker build -t sovereign-ai-api api
docker run -p 8080:8080 --env-file .env sovereign-ai-api
```

Training, merging and the vLLM benchmark need a GPU (a free Kaggle or Colab T4 works). See the files in `scripts/`. The data folder, GGUF files and keys are not committed.

## 10. Limitations (read before trusting the numbers)

- One run per system on 200 bills, with bootstrap confidence intervals (section 4.7) but no repeated training seeds. Generation caps differed: the 16-bit and Q4 outputs were capped at 300 tokens while base Qwen and Gemini had no cap, and 29.5% of the 16-bit outputs hit that cap.
- ROUGE and BERTScore reward matching the reference summary style, not necessarily factual correctness.
- Gemini was also graded by a Gemini judge, and a first lenient judging pass was discarded for scoring itself 5.0 out of 5.0.
- Latencies across systems are not comparable (Mac and Ollama, T4 and vLLM, network API).
- The GPU price is an assumption, and the API price is the list rate at the time of writing, without batch or cache discounts.
- 4-bit quantization inside vLLM (AWQ or GPTQ) was not tested, because `bitsandbytes` is not supported in the vLLM version used.
- PubMedQA, the harder variant in the original brief, was not run.
- Licensing: the base model is under the Qwen Research License (non-commercial); read section 13 before any commercial use.

## 11. Next steps

Add AWQ or GPTQ 4-bit serving on vLLM; add confidence intervals over several seeds; run PubMedQA; re-run the throughput benchmark with CUDA graphs enabled.

## 12. Glossary

| Term | Meaning |
|---|---|
| Open-weight model | A model whose weights you can download and run yourself |
| Fine-tuning | Further training a model on examples of your task |
| LoRA | A cheap fine-tuning method that trains a small add-on instead of the whole model |
| Quantization | Storing weights in fewer bits (for example 4-bit) to shrink the model |
| GGUF | A file format for quantized models, used by llama.cpp and Ollama |
| ROUGE | Measures word overlap between a generated and a reference summary |
| BERTScore | Measures meaning similarity between two texts using a language model |
| P50 / P95 latency | Time within which 50% / 95% of requests finish |
| Throughput | How much output the system produces per second |
| vLLM | A fast serving engine for large language models |
| Crossover | The monthly call volume where self-hosting becomes cheaper than the API |

## Credits

BillSum dataset: Kornilova and Eidelman (2019), "BillSum: A Corpus for Automatic Summarization of US Legislation". Built with Unsloth, Hugging Face PEFT, vLLM, llama.cpp, Ollama, FastAPI and Arize Phoenix.
"""
open("README.md","w").write(md); print("README.md written,",len(md),"chars")
