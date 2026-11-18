### 4.8 Cost sensitivity

The GPU price and how busy the GPU is both change the answer. Self-hosted cost per 1,000 calls (using the measured 0.59 requests/s at concurrency 32), against an API cost of $0.772 per 1,000 calls:

| GPU $/hr | 100% busy | 50% busy | 25% busy | 10% busy | Crossover (calls/month) |
|---|---|---|---|---|---|
| $0.25 | $0.12 | $0.24 | $0.47 | $1.18 | 236,352 |
| $0.35 | $0.16 | $0.33 | $0.66 | $1.65 | 330,893 |
| $0.50 | $0.24 | $0.47 | $0.94 | $2.35 | 472,705 |
| $0.75 | $0.35 | $0.71 | $1.41 | $3.53 | 709,057 |
| $1.00 | $0.47 | $0.94 | $1.88 | $4.71 | 945,409 |

- A GPU that is only 10% busy costs more per call than the API at every price above. Self-hosting needs steady volume.
- With the API's 50% batch discount ($0.386 per 1,000 calls) the crossover at $0.50/hr doubles to about 945,409 calls/month.
- Prices are assumptions, not quotes. Check current GPU and API rates before deciding.
- If the 2-shot prompt is used (more input tokens), the API cost is $0.761 per 1,000 calls and the crossover at $0.50/hr falls to about 479,405 calls/month.

