# Sovereign AI Engine, explained step by step

This file explains the whole project from start to finish in plain language. You do not need to know machine learning. For every step it answers four questions:

- **What** happens
- **Why** it happens
- **Which tool** does it, and why that tool was chosen
- **What came out** of it

If you only want the results, read the [README](README.md). If you want to understand how every number in the README was produced, read this file from top to bottom.

---

## 1. The big picture

### The question

Companies that want an AI to read long documents have two choices:

1. **Pay for an API.** Send the document to a big company's AI (here, Google's Gemini) and pay per use.
2. **Run your own model.** Download a smaller open AI model and run it on hardware you control. This is called a **"sovereign"** setup, because the data never leaves your hands.

Which one is better? That depends on quality, speed, cost, and how many documents you process. This project measures all four on one real task.

### The task

**Read a US congressional bill and write a short, factual summary of it (at most 200 words).**

Bills are long and dense, and real experts have already written summaries for thousands of them. That gives us a way to check the AI's work.

### What we did, in six sentences

1. We took a small open model (Qwen2.5-3B) and tested how well it summarizes bills as it comes.
2. We also tested Google's Gemini API on the same bills, with no examples and then with two examples in the prompt.
3. We **fine-tuned** the small model on 1,500 bills with expert summaries, so it learns the style.
4. We **shrank** the model (quantization) and measured what that cost in quality and speed.
5. We served the model like a real product (an API in Docker) and measured speed under load, then worked out when running your own model costs less than paying the API.
6. We checked the answers in several ways, including a manual audit of invented facts, and wrote down everything that went wrong.

### The honest answer we found

- The fine-tuned model scored **higher than Gemini on overlap with the expert summaries** (ROUGE-L 0.409 against 0.375 for 2-shot Gemini).
- But in a small manual check, it also made **more claims that the bill does not support** than 2-shot Gemini did (1.00 per summary against 0.20).
- Running your own model only becomes cheaper above roughly **250,000 to 470,000 calls per month**, and only if the GPU stays busy.

So the right choice depends on what you value. The details are below.

---

## 2. Five words you need first

| Word | Plain meaning |
|---|---|
| **Model** | A big file of numbers (called weights) that has learned to predict text. You give it text, it writes more text. |
| **Fine-tuning** | Teaching an existing model a specific job by showing it many examples of that job. |
| **Quantization** | Storing the model's numbers with fewer bits to make the file smaller and faster, like saving a photo as a smaller JPEG. |
| **Inference** | Actually using the model to produce an answer (as opposed to training it). |
| **GPU** | A graphics chip that is very good at the math AI needs. Training and fast serving need one. |

A **token** is a small chunk of text, roughly three quarters of a word. Models read and write tokens, and APIs charge per token. An average bill in this project is about 1,700 tokens long, and a summary is about 190 tokens.

---

## 3. The tools used, and why each was chosen

| Tool | Its job here | Why it was chosen | Cost |
|---|---|---|---|
| **BillSum** (dataset) | The bills and the expert summaries | A public dataset made for exactly this task, and it has real human-written answers to compare against | Free (CC0 license) |
| **Qwen2.5-3B-Instruct** | The small open model we run ourselves | Small enough to fine-tune on one free GPU and to run on a laptop, and it follows instructions well | Free to download. Non-commercial license, see section 9 |
| **Gemini 3.1 Flash-Lite** | The API model we compare against | Cheap, fast, and has a free tier, so it is a fair "pay for an API" baseline. It is a low-cost model, not Google's biggest | Free tier, then pay per token |
| **Unsloth** | Runs the fine-tuning | Makes fine-tuning faster and use less memory, which is what lets it fit on a free 15 GB GPU | Free |
| **PEFT / LoRA** | The fine-tuning method | Trains a tiny add-on (about 125 MB) instead of changing all 3 billion numbers | Free |
| **Kaggle and Google Colab** | Free cloud GPUs (NVIDIA T4) | The project had to be completely free, and neither Unsloth nor vLLM can run on a Mac | Free (with weekly limits) |
| **vLLM** | Serves the model quickly to many users at once | An industry-standard serving engine. It handles many requests together, which is how real products serve AI | Free |
| **llama.cpp (GGUF)** | Shrinks the model to 8-bit and 4-bit | GGUF is the file format that runs on Apple laptops and in Ollama | Free |
| **Ollama** | Runs models on the MacBook | The easiest way to run a model on a laptop with one command | Free |
| **FastAPI** | The web API around the model | A simple, popular Python tool to turn code into a web service | Free |
| **Docker** | Packages the API so it runs the same everywhere | Avoids "it works on my machine" problems | Free |
| **Arize Phoenix** | Records every AI call for inspection | Open source and runs locally. You can see timing, tokens, and input and output for each call | Free |
| **ROUGE and BERTScore** | Score how close a summary is to the expert one | ROUGE counts shared words and phrases. BERTScore compares meaning. Together they are a standard pair for summarization | Free |
| **pytest + GitHub Actions** | Automatic tests on every push | Catches mistakes (such as results files with the wrong shape) before anyone sees them | Free |
| **Hugging Face Hub** | Stores the model files | The standard place to host models. Our repos are private | Free |

---

## 4. The pipeline, step by step

Here is the order of the project. Each step feeds the next.

```
Data -> Prompt -> Baselines -> Scoring -> Fine-tune -> Merge -> Serve and benchmark
     -> Shrink (quantize) -> Statistics -> Manual audit -> Cost model -> API -> Tests
```

### Step 1. Choose the task and the data

- **What:** We picked US bill summarization and the BillSum dataset. BillSum pairs each bill with a summary written by a person.
- **Why:** Summaries written by people give us an answer key. Without one, we could not say whether a model did well.
- **Tool:** BillSum from Hugging Face (`FiscalNote/billsum`).
- **What came out:** A task with a measurable goal: write a summary close to the expert one, without inventing facts.

### Step 2. Prepare the data

- **What:** `scripts/prepare_data.py` downloads BillSum and keeps only bills under 12,000 characters. It saves 2,000 bills for training and 200 for testing as simple text files.
- **Why we limit length:** Long bills need more memory and more time. A free 15 GB GPU can only handle so much, so we kept the bills moderate in size. The 200 test bills range from about 5,000 to 12,000 characters.
- **Why separate train and test:** The test bills are never used for training. If a model had seen the test bills, its scores would be inflated. This is called leakage, and it is the most common way to fool yourself.
- **What came out:** `train` (used for fine-tuning) and `test` (used only for scoring). Of the 2,000 training bills, the first 1,500 that fit in 3,072 tokens were used.

### Step 3. Fix one prompt

- **What:** Every system gets the same instruction: *"Summarize the following US congressional bill in a clear, factual legislative summary of at most 200 words. Output only the summary."* followed by the bill.
- **Why:** If each model got a different instruction, we could not tell whether differences came from the model or from the instruction. One shared prompt makes the comparison fair.
- **Where:** `src/prompt.py`.

### Step 4. Run the baselines (before any training)

- **What:** We ran two systems on the 200 test bills, with no training:
  - **Base Qwen2.5-3B** on the MacBook through Ollama.
  - **Gemini 3.1 Flash-Lite** through the API, with no examples (called **zero-shot**).
- **Why:** A baseline tells you where you start. If fine-tuning only matches the baseline, it was not worth doing.
- **Settings:** For local models we used temperature 0 (always the most likely word), a 300-token cap on the answer, and a large enough context window to fit the whole bill.
- **What came out:** Two files of 200 summaries each, plus the time each one took.

### Step 5. Score the summaries

- **What:** We compare each model summary to the expert summary with two kinds of scores:
  - **ROUGE-1, ROUGE-2, ROUGE-L.** ROUGE-1 counts shared single words. ROUGE-2 counts shared two-word phrases (a stricter test of matching wording). ROUGE-L looks at the longest shared sequence of words in order.
  - **BERTScore F1.** Compares meaning using a small language model (we used `distilbert-base-uncased`, so compare BERTScores only with each other).
- **Why two kinds:** A summary can use different words and mean the same thing, or use the same words and mean something wrong. No single score catches everything.
- **Important warning:** These scores measure similarity to the expert summary. They **do not measure whether the summary is true.** Section 4, Step 12 returns to this.
- **Starting scores:** Base Qwen ROUGE-L 0.319. Gemini zero-shot ROUGE-L 0.320.

### Step 6. Fine-tune the small model

- **What:** We trained the model on 1,500 bills with their expert summaries, so it learns what a legislative summary looks like: the length, the order, and the wording.
- **How (LoRA):** Changing all 3 billion numbers would need a huge GPU. LoRA trains a small add-on that adjusts how the model behaves. The add-on here is only about 125 MB.
- **Settings in simple terms:**
  - The base model was loaded in 4-bit form to save memory (this combination is often called QLoRA).
  - LoRA rank `r=16` sets the size of the add-on. A bigger number can learn more but uses more memory.
  - 1 pass over the data (1 epoch), 94 update steps, learning rate 0.0002.
  - It trained on a free Google Colab T4 GPU and took about an hour.
- **How we know it learned:** The training loss (a measure of error, lower is better) fell from about 1.22 to about 0.94.
- **Tools and why:** Unsloth for speed and low memory, PEFT for the LoRA method.
- **What came out:** A LoRA adapter file. The adapter alone is not a usable model yet. It must be combined with the base model.

### Step 7. Merge the adapter into the model

- **What:** We combined the base model and the adapter into one ordinary 16-bit model (about 5.8 GB) and uploaded it to a private Hugging Face repo.
- **Why:** Serving tools like vLLM and llama.cpp work with a single normal model, not a model plus an add-on.
- **Why on CPU:** Colab's free GPU time ran out, so the merge was done on a CPU session with a small script (`scripts/merge_lora_cpu.py`) that adds the adapter's changes layer by layer. It checked out: 252 layers were merged.
- **What came out:** `billsum-qwen2.5-3b-merged` (private on Hugging Face).

### Step 8. Serve it like a real product and measure the speed

- **What:** We ran the merged model on **vLLM** on a Kaggle T4 GPU and sent it many bills at once, with 1, 8, and 32 requests at the same time (this is called **concurrency**).
- **Why:** A model that scores well but answers too slowly is not useful. A product needs to handle many users at once.
- **Words you will see:**
  - **P50 latency:** half of requests finish faster than this.
  - **P95 latency:** 95% of requests finish faster than this (it shows how slow the unlucky ones are).
  - **Throughput:** how much text the system produces per second.
- **Results:**

  | Requests at once | P50 (s) | P95 (s) | Tokens/s | Requests/s |
  |---|---|---|---|---|
  | 1 | 7.8 | 12.5 | 24.6 | 0.13 |
  | 8 | 15.7 | 26.5 | 104.1 | 0.49 |
  | 32 | 53.1 | 86.5 | 125.5 | 0.59 |

- **What this means:** More requests at once means more total work per second, but each person waits longer. For an interactive product, keep concurrency low and add GPUs instead.
- **Fine print:** vLLM ran in a slower "eager" mode to fit the free GPU, so these numbers are probably lower than what a tuned setup would reach. The quality scores did not change with load.
- **What came out:** `results/vllm_16bit_*.jsonl` and `vllm_16bit_summary.json`.

### Step 9. Shrink the model (quantization)

- **What:** We converted the model to two smaller files with llama.cpp:

  | Format | File size | Meaning |
  |---|---|---|
  | 16-bit (original) | 5.8 GB | Full precision |
  | Q8_0 | 3.1 GB | 8 bits per number |
  | Q4_K_M | 1.8 GB | About 4 bits per number |

- **Why:** A smaller model needs less memory and usually runs faster. That is the difference between needing a server and running on a laptop.
- **Why GGUF and not vLLM's 4-bit:** The vLLM version we used no longer supported the `bitsandbytes` 4-bit option. GGUF runs on the MacBook, so we used it.
- **What it cost in quality:** Q4_K_M was **not significantly worse** than 16-bit on ROUGE-L. Q8_0 and Q4_K_M did differ: Q4_K_M was about 0.03 ROUGE-1 lower than Q8_0.
- **What it gained in speed (same idle MacBook):** Q4_K_M produced about 15.8 tokens per second against 12.3 for Q8_0, about 28% faster.
- **A trap we avoided:** Q8_0 scored slightly *higher* than the 16-bit model. That does not mean 8-bit is better. The Q8_0 summaries were shorter (about 154 words against 171) and ran into the 300-token cap less often, and the models were served by different software. Differences in how the answers were generated, not the compression, probably explain the gap.

### Step 10. Check the results with statistics

A single score can be luck. With only 200 bills, small differences can be noise. We added checks:

- **Bootstrap 95% confidence intervals.** We re-sample the 200 bills many times and see how much the score moves. If the range for a difference does not include zero, the difference is real (we call it "significant").
- **Paired comparison.** Because every system summarized the same bills, we compare them bill by bill, which is a stricter and fairer test.
- **Length control.** The local models were capped at 300 tokens and the API was not, and about 30% of the fine-tuned model's answers hit the cap. So we also re-scored everything after cutting each answer to its first 200 words, the length the prompt asked for. The ranking did not change.
- **Length buckets.** We split the test bills into four groups by length. The fine-tuned model came out ahead of 2-shot Gemini in every group, but on a per-bill basis it was higher on only 54% of bills. The average lead is real but modest.
- **Number check.** We looked for numbers in each summary that never appear in the bill. This is a crude check, because legitimate rewrites also trigger it. The fine-tuned model had the highest rate (5.4% against roughly 3% for the others, and 1.7% for 2-shot Gemini), but the ranges overlap, so treat it as a hint.

**Key results (200 test bills):**

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|---|---|---|---|---|
| Base Qwen2.5-3B | 0.492 | 0.237 | 0.319 | 0.813 |
| Gemini, zero-shot | 0.502 | 0.228 | 0.320 | 0.820 |
| Gemini, 2-shot | 0.539 | 0.285 | 0.375 | 0.841 |
| Fine-tuned, 16-bit | 0.536 | 0.342 | 0.409 | 0.852 |
| Fine-tuned, Q8_0 | 0.559 | 0.356 | 0.423 | 0.854 |
| Fine-tuned, Q4_K_M | 0.529 | 0.328 | 0.400 | 0.846 |

Fine-tuned 16-bit minus zero-shot Gemini was +0.089 ROUGE-L (confidence range +0.071 to +0.108). Fine-tuned 16-bit minus 2-shot Gemini was +0.034 (range +0.016 to +0.052).

### Step 11. A fairer API baseline: giving Gemini examples

- **What:** The first Gemini run had no examples. A fair question is: what if you simply show the API two examples of a good summary? That is called **2-shot** prompting.
- **Result:** Gemini's ROUGE-L rose from 0.320 to 0.375, closing about 62% of the gap to the fine-tuned model. On ROUGE-1 it tied the fine-tuned model (0.539 against 0.536). The fine-tuned model still led on ROUGE-2, ROUGE-L, and BERTScore.
- **The cost:** The two examples add about 2,800 tokens to every request, so the API cost rose from $0.77 to $1.44 per 1,000 calls.
- **Lesson:** Try better prompting before fine-tuning. It is cheaper and closes most of the gap.
- **A mistake we caught here:** The first 2-shot run silently contained **no examples**, because the code that picked the example bills matched none. The numbers looked plausible, but the token counts gave it away (only 12 more tokens than zero-shot, when two examples should add thousands). We flagged the wrong row on GitHub, fixed the code, re-ran it, and now the code refuses to continue if the examples are missing.

### Step 12. Check whether the summaries are true (manual audit)

- **Why this step exists:** ROUGE and BERTScore only reward similarity. A summary can score well and still contain an invented amount, date, or agency. For legal text, that matters most.
- **What we did:** We picked 15 test bills at random. For each, three summaries (from fine-tuned 16-bit, fine-tuned Q4_K_M, and 2-shot Gemini) were shown **anonymously**, labeled A, B, C in shuffled order, with the answer key hidden. Each was cut to 200 words so length would not give it away. For each summary, we counted the claims the bill does not support: a number, name, date, or requirement that is missing from the bill or says more than the bill does.
- **Results (average unsupported claims per summary, lower is better):**

  | System | Per summary | Summaries with at least one |
  |---|---|---|
  | Gemini, 2-shot | 0.20 | 3 of 15 |
  | Fine-tuned, Q4_K_M | 0.53 | 7 of 15 |
  | Fine-tuned, 16-bit | 1.00 | 10 of 15 |

  Fine-tuned 16-bit minus 2-shot Gemini was +0.80 (range +0.33 to +1.27), which is a clear gap.
- **What this means:** The fine-tuned model copies the style of the expert summaries well but invents more details than the prompted API model. Higher ROUGE did **not** mean more accurate.
- **How to read it fairly:**
  - It is an **AI-assisted audit**, not an independent human review. An AI model read each bill and the three summaries and proposed the counts, and the author entered them. Some counts were judgment calls.
  - The rule was: a claim counts if the bill does not say it or the summary says it more strongly than the bill. Leaving something out does not count.
  - Only 15 bills were checked, and the base model and zero-shot Gemini were not audited, so this cannot show that fine-tuning itself caused the extra errors.
  - It is a spot check, not a measurement of overall accuracy.
- **What was tried first and thrown away:** An automatic AI "judge" (Gemini Flash-Lite grading the summaries) gave Gemini a perfect 5.0 out of 5 on its own work and rated the base model above the fine-tuned model. That contradicts every other measure, so we discarded it. A stricter judge never finished because the API was overloaded.

### Step 13. Work out the cost and when self-hosting wins

- **What:** We compared the cost per 1,000 summaries for the API and for running your own GPU.
- **API cost:** Gemini 3.1 Flash-Lite charges $0.25 per million input tokens and $1.50 per million output tokens (standard rate). Using the measured token counts:
  - Zero-shot: about 1,800 input and 214 output tokens per call, which comes to **$0.77 per 1,000 calls**.
  - 2-shot: about 4,600 input tokens per call, which comes to **$1.44 per 1,000 calls**.
- **Self-hosted cost:** A GPU costs money by the hour whether it is busy or idle. We assumed **$0.50 per hour** for a T4-class GPU (an assumption, not a quote; Kaggle and Colab themselves were free). That is about **$365 per month**. When fully busy, the measured speed works out to **$0.24 to $0.28 per 1,000 calls**.
- **The crossover:** Self-hosting wins once you process enough documents that the fixed GPU cost is spread thin. That is **about 253,000 calls per month** (against the 2-shot API) to **about 473,000** (against the zero-shot API).
- **The catch:** This assumes the GPU is busy. A GPU that is only 10% busy costs more per call than the API. The README includes a table of how the answer changes with GPU price and how busy the GPU is.
- **Why this matters:** Self-hosting is not automatically cheaper. It pays off with steady, high volume, or when your data cannot leave your network.

### Step 14. Wrap the model in a real API

- **What:** `api/app.py` is a small web service (FastAPI) with two endpoints: `POST /summarize` (send bill text, get a summary back) and `GET /health` (check it is alive). A single field in the request chooses the engine: `ollama` (the local model), `vllm`, or `gemini`.
- **Why:** A model in a notebook cannot serve users. An API is how real products use models, and having switchable backends makes it easy to compare engines.
- **Docker:** The API is packaged in a Docker container so it runs the same on any machine.
- **Tracing (Arize Phoenix):** Every call is recorded: how long it took, how many tokens, what went in, and what came out. This helps you debug and watch costs.

### Step 15. Tests, automation, and the README itself

- **Tests:** `tests/` has automated checks that the results files have the right shape, that all main runs cover the same 200 bills, and that the prompt is intact.
- **GitHub Actions:** These tests run automatically on every push, so a broken change shows up as a red mark.
- **The README is generated:** A script builds the README tables straight from the saved result files, so the numbers in the README cannot drift from the real results. When we fixed the bad 2-shot run, every table and sentence updated from the data.

---

## 5. What went wrong, and what it taught us

Mistakes are part of the project. Each was caught, fixed, and documented.

| What happened | How it was caught | What we did |
|---|---|---|
| The AI judge gave Gemini a perfect score on its own work | The result contradicted every other measure | Discarded it and wrote it up |
| The 2-shot baseline had no examples | The token count rose by only 12 per request | Flagged the row on GitHub, fixed the code, re-ran, and added a check that stops the script if examples are missing |
| Q8_0 scored higher than the original model | It should not be possible from compression alone | Showed it was a generation difference and said so in the README |
| The vLLM 4-bit option was missing | The server refused to start | Switched to GGUF quantization |
| Free GPU time ran out twice (Colab and the Gemini daily quota) | Error messages | Moved to Kaggle, merged on CPU, and resumed runs from where they stopped |
| The base model's license is non-commercial | We checked it before publishing | Added a license section and a NOTICE file, and kept the model repos private |

---

## 6. What this project does **not** prove

- **It is not a production system.** It is a small model, one training run, and a free shared GPU.
- **One training run.** We did not repeat training with different random seeds, so we cannot say how much results vary between runs.
- **200 test bills** is a decent sample, but not huge.
- **ROUGE and BERTScore are imperfect.** They reward similar wording, not truth.
- **The faithfulness audit is small and AI-assisted** (15 bills, 3 systems).
- **Speed numbers are not comparable across machines.** A MacBook with Ollama, a T4 with vLLM, and a network API are different setups. Use the vLLM table for serving claims.
- **The GPU price is an assumption**, and the API price is the list price at the time of writing.
- **Not tested:** 4-bit serving on vLLM (AWQ or GPTQ), vLLM with CUDA graphs enabled, and the harder PubMedQA dataset from the original brief.

---

## 7. Licenses and attribution

- The code in this repo is MIT-licensed.
- **BillSum** is CC0 (no restrictions) according to its Hugging Face card.
- **Qwen2.5-3B-Instruct** is under the **Qwen Research License**, which allows non-commercial use only. The fine-tuned, merged, and GGUF files are derived from it and carry the same terms, which is why those repos stay private. A company wanting to deploy this commercially would need a license from Alibaba Cloud or a base model whose license allows it, and would need to re-run the evaluation with that model.
- See `NOTICE` for the required attribution. This is not legal advice.

---

## 8. Where everything lives in the repo

| Path | What is in it |
|---|---|
| `README.md` | The results summary, auto-built from the result files |
| `PROJECT_EXPLAINED.md` | This file |
| `scripts/` | Steps that need a GPU or heavy setup: prepare data, fine-tune, merge, benchmark vLLM, quantize |
| `src/` | Steps that run anywhere: run models, score, statistics, cost model, audit tools, README builder |
| `api/` | The FastAPI service and its Dockerfile |
| `tests/` | Automatic checks |
| `results/` | Every model's raw summaries (`.jsonl`), benchmarks, and the cost chart |
| `docs/` | Diagrams and the Phoenix screenshot |
| `.github/workflows/ci.yml` | Runs the tests on every push |
| `LICENSE`, `NOTICE` | License and attribution |

Not stored in the repo: the datasets (re-downloadable), the model files (private and large), and the raw audit sheets (kept private so the audit stays blind).

---

## 9. How to reproduce it

```bash
pip install -r requirements.txt
cp .env.example .env                      # add GEMINI_API_KEY and HF_TOKEN
python scripts/prepare_data.py            # make the train and test files
python src/run_ollama.py 200 qwen2.5:3b   # base model baseline (needs Ollama)
python src/run_gemini.py 200              # API baseline
python src/score.py results/*.jsonl       # ROUGE scores
python src/cost.py 0.50                   # cost and crossover (argument = GPU $/hour)
docker build -t sovereign-ai-api api
docker run -p 8080:8080 --env-file .env sovereign-ai-api
```

Fine-tuning, merging, and the vLLM benchmark need an NVIDIA GPU (a free Kaggle or Colab T4 works). Use the scripts in `scripts/`.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **API** | A way for one program to ask another for something over the internet |
| **Baseline** | A starting point to compare against |
| **Bootstrap (confidence interval)** | Re-sampling the data many times to see how much a result could change by luck |
| **BERTScore** | A score that compares the meaning of two texts |
| **Concurrency** | How many requests are being handled at the same moment |
| **Context window** | How much text a model can read at once |
| **Docker** | A box that packages software so it runs the same everywhere |
| **Epoch** | One full pass over the training data |
| **Fine-tuning** | Teaching an existing model a specific job with examples |
| **GGUF** | A file format for compressed models that run on laptops |
| **GPU** | A chip that does AI math quickly |
| **Hallucination** | When a model states something that is not true or not in the source |
| **Inference** | Using a trained model to produce answers |
| **Latency (P50, P95)** | How long a request takes. P95 shows how bad the slow ones are |
| **LoRA** | A cheap way to fine-tune by training a small add-on |
| **Loss** | A number measuring the model's error during training. Lower is better |
| **Open-weight model** | A model whose files you can download and run yourself |
| **Prompt** | The instruction and input you give a model |
| **Quantization** | Storing a model's numbers with fewer bits so it is smaller and faster |
| **ROUGE** | A score based on shared words and phrases between two texts |
| **Sovereign AI** | Running AI on hardware you control so your data stays with you |
| **Throughput** | How much output a system produces per second |
| **Token** | A small chunk of text, about three quarters of a word |
| **vLLM** | A fast engine for serving language models to many users |
| **Zero-shot / 2-shot** | Asking a model to do a task with no examples, or with two examples in the prompt |
