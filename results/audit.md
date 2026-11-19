### 4.10 Manual faithfulness audit

One reviewer, assisted by an AI model, checked 15 randomly chosen test bills. For each bill the three summaries were shown anonymously (labels A, B, C in shuffled order, each cut to its first 200 words) and the reviewer counted claims the bill does not support (amounts, dates, names, programs or requirements that are missing from or contradict the bill).

| System | Summaries | Avg unsupported claims | Summaries with at least one (95% Wilson CI) |
|---|---|---|---|
| Fine-tuned 16-bit | 15 | 1.00 | 10/15 = 67% [42%, 85%] |
| Gemini 3.1 Flash-Lite (2-shot) | 15 | 0.20 | 3/15 = 20% [7%, 45%] |
| Fine-tuned Q4_K_M | 15 | 0.53 | 7/15 = 47% [25%, 70%] |

Paired difference in average unsupported claims per summary (bootstrap over bills; a negative value means fewer unsupported claims):

| Comparison | Mean difference | 95% CI |
|---|---|---|
| Fine-tuned 16-bit minus Gemini 3.1 Flash-Lite (2-shot) | +0.80 | [+0.33, +1.27] |
| Fine-tuned 16-bit minus Fine-tuned Q4_K_M | +0.47 | [+0.07, +0.87] |

Limits: the counts were proposed by an AI model (Claude) after it read each bill and the three summaries, and the reviewer entered them, so this is an AI-assisted audit and not an independent human review. The counting rule was: a summary claim counts as unsupported if the bill does not say it or the summary states it more strongly than the bill does; claims that are weaker than the bill, and omissions, do not count. Some counts were judgment calls. The reviewer knew which project this was, only 15 bills were checked, and style cues such as length or run-on text can still hint at the system. Treat this as a spot check, not a measurement of overall factual accuracy.

