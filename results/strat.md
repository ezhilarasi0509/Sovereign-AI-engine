### 4.9 Where does it win or lose?

ROUGE-L by source bill length (quartiles of bill characters, about 50 bills each):

| System | Shortest 25% (5,008 to 6,171 chars) | 25-50% (6,194 to 7,788 chars) | 50-75% (7,799 to 9,903 chars) | Longest 25% (9,923 to 11,978 chars) |
|---|---|---|---|---|
| Base Qwen2.5-3B | 0.327 | 0.335 | 0.317 | 0.296 |
| Gemini zero-shot | 0.325 | 0.329 | 0.331 | 0.294 |
| Gemini 2-shot | 0.386 | 0.399 | 0.374 | 0.342 |
| Fine-tuned 16-bit | 0.426 | 0.428 | 0.412 | 0.370 |
| Fine-tuned Q4_K_M | 0.425 | 0.415 | 0.388 | 0.372 |

Per-bill comparison, fine-tuned 16-bit against 2-shot Gemini: the fine-tuned model scores higher on 54% of bills, lower on 46%. Averages hide variation, so see [docs/examples.md](docs/examples.md) for the three bills where the fine-tuned model gains most and the three where it loses most.

