### 4.7 Statistical checks

**Bootstrap 95% confidence intervals** (200 bills, 3,000 resamples):

| System | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Base Qwen2.5-3B | 0.492 [0.478, 0.505] | 0.237 [0.228, 0.248] | 0.319 [0.309, 0.329] |
| Gemini 3.1 Flash-Lite | 0.502 [0.489, 0.515] | 0.228 [0.219, 0.237] | 0.320 [0.310, 0.329] |
| Fine-tuned 16-bit | 0.536 [0.516, 0.556] | 0.342 [0.322, 0.363] | 0.409 [0.389, 0.429] |
| Fine-tuned Q4_K_M | 0.529 [0.510, 0.549] | 0.328 [0.309, 0.347] | 0.400 [0.381, 0.420] |

**Paired differences in ROUGE-L** (same bills, bootstrap):

| Comparison | Mean difference | 95% CI | Verdict |
|---|---|---|---|
| Fine-tuned 16-bit minus Gemini 3.1 Flash-Lite | +0.089 | [+0.071, +0.109] | significant |
| Fine-tuned 16-bit minus Base Qwen2.5-3B | +0.090 | [+0.071, +0.109] | significant |
| Fine-tuned 16-bit minus Fine-tuned Q4_K_M | +0.009 | [-0.009, +0.027] | not significant |

The fine-tuned model's lead over Gemini and the base model is statistically supported. The 16-bit versus Q4_K_M gap is not distinguishable from noise on 200 bills.

**Length control.** Output caps differed: the 16-bit and Q4 runs were capped at 300 tokens, while base Qwen and Gemini had no cap. The share of outputs that hit 299 or more tokens, plus a rescoring with every output cut to its first 200 words (the length the prompt asked for):

| System | Hit 300-token cap | Avg words (reference 151) | ROUGE-L, full output | ROUGE-L, first 200 words |
|---|---|---|---|---|
| Base Qwen2.5-3B | 9.5% | 152 | 0.319 | 0.318 |
| Gemini 3.1 Flash-Lite | 0.0% | 167 | 0.320 | 0.320 |
| Fine-tuned 16-bit | 29.5% | 171 | 0.409 | 0.411 |
| Fine-tuned Q4_K_M | 20.0% | 144 | 0.400 | 0.400 |

After the 200-word cut, fine-tuned minus Gemini is +0.092 ROUGE-L [+0.074, +0.110], significant.

**Numbers not found in the bill.** A crude grounding check: every number in a summary (2 or more characters) is searched for in the source bill. Misses include legitimate rewrites such as totals or spelled-out numbers, so only compare systems with each other.

| System | Numbers checked | Not in bill | Rate (95% Wilson CI) |
|---|---|---|---|
| Base Qwen2.5-3B | 738 | 24 | 3.3% [2.2%, 4.8%] |
| Gemini 3.1 Flash-Lite | 718 | 22 | 3.1% [2.0%, 4.6%] |
| Fine-tuned 16-bit | 523 | 28 | 5.4% [3.7%, 7.6%] |
| Fine-tuned Q4_K_M | 486 | 16 | 3.3% [2.0%, 5.3%] |

Fine-tuning improved style and overlap but showed no evidence of better factual grounding on this check: the 16-bit model had the highest rate of unmatched numbers, though its interval overlaps the others, so this is suggestive rather than conclusive. Verify facts before using any summary in a legal setting.

