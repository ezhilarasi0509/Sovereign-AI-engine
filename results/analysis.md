### 4.7 Statistical checks

**Bootstrap 95% confidence intervals** (200 bills, 3,000 resamples):

| System | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Base Qwen2.5-3B | 0.492 [0.478, 0.505] | 0.237 [0.228, 0.248] | 0.319 [0.309, 0.329] |
| Gemini 3.1 Flash-Lite (zero-shot) | 0.502 [0.489, 0.515] | 0.228 [0.219, 0.237] | 0.320 [0.310, 0.329] |
| Gemini 3.1 Flash-Lite (2-shot) | 0.539 [0.525, 0.552] | 0.285 [0.273, 0.297] | 0.375 [0.363, 0.388] |
| Fine-tuned 16-bit | 0.536 [0.516, 0.558] | 0.342 [0.322, 0.362] | 0.409 [0.389, 0.429] |
| Fine-tuned Q8_0 | 0.559 [0.539, 0.579] | 0.356 [0.336, 0.377] | 0.423 [0.403, 0.443] |
| Fine-tuned Q4_K_M | 0.529 [0.509, 0.549] | 0.328 [0.308, 0.348] | 0.400 [0.381, 0.420] |

**Paired differences in ROUGE-L** (same bills, bootstrap):

| Comparison | Mean difference | 95% CI | Verdict |
|---|---|---|---|
| Fine-tuned 16-bit minus Gemini 3.1 Flash-Lite (zero-shot) | +0.089 | [+0.071, +0.108] | significant |
| Fine-tuned 16-bit minus Gemini 3.1 Flash-Lite (2-shot) | +0.034 | [+0.016, +0.052] | significant |
| Fine-tuned 16-bit minus Base Qwen2.5-3B | +0.090 | [+0.071, +0.109] | significant |
| Fine-tuned 16-bit minus Fine-tuned Q4_K_M | +0.009 | [-0.009, +0.027] | not significant |
| Fine-tuned Q8_0 minus Fine-tuned 16-bit | +0.014 | [-0.001, +0.029] | not significant |
| Fine-tuned Q8_0 minus Fine-tuned Q4_K_M | +0.023 | [+0.008, +0.037] | significant |

A confidence interval that includes zero means the difference is not distinguishable from noise on 200 bills. Read each row accordingly.

**Length control.** Output caps differed: the 16-bit and Q4 runs were capped at 300 tokens, while base Qwen and Gemini had no cap. The share of outputs that hit 299 or more tokens, plus a rescoring with every output cut to its first 200 words (the length the prompt asked for):

| System | Hit 300-token cap | Avg words (reference 151) | ROUGE-L, full output | ROUGE-L, first 200 words |
|---|---|---|---|---|
| Base Qwen2.5-3B | 9.5% | 152 | 0.319 | 0.318 |
| Gemini 3.1 Flash-Lite (zero-shot) | 0.0% | 167 | 0.320 | 0.320 |
| Gemini 3.1 Flash-Lite (2-shot) | 0.0% | 152 | 0.375 | 0.375 |
| Fine-tuned 16-bit | 29.5% | 171 | 0.409 | 0.411 |
| Fine-tuned Q8_0 | 24.5% | 154 | 0.423 | 0.423 |
| Fine-tuned Q4_K_M | 20.0% | 144 | 0.400 | 0.400 |

After the 200-word cut, fine-tuned minus zero-shot Gemini is +0.092 ROUGE-L [+0.074, +0.110], significant.

**Numbers not found in the bill.** A crude grounding check: every number in a summary (2 or more characters) is searched for in the source bill. Misses include legitimate rewrites such as totals or spelled-out numbers, so only compare systems with each other.

| System | Numbers checked | Not in bill | Rate (95% Wilson CI) |
|---|---|---|---|
| Base Qwen2.5-3B | 738 | 24 | 3.3% [2.2%, 4.8%] |
| Gemini 3.1 Flash-Lite (zero-shot) | 718 | 22 | 3.1% [2.0%, 4.6%] |
| Gemini 3.1 Flash-Lite (2-shot) | 647 | 11 | 1.7% [1.0%, 3.0%] |
| Fine-tuned 16-bit | 523 | 28 | 5.4% [3.7%, 7.6%] |
| Fine-tuned Q8_0 | 541 | 16 | 3.0% [1.8%, 4.7%] |
| Fine-tuned Q4_K_M | 486 | 16 | 3.3% [2.0%, 5.3%] |

Fine-tuning improved style and overlap but showed no evidence of better factual grounding on this check: the 16-bit model had the highest rate of unmatched numbers, though its interval overlaps the others, so this is suggestive rather than conclusive. The manual audit (section 4.10) points the same way. Verify facts before using any summary in a legal setting.

