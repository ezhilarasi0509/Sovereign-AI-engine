# Start first: vllm serve <merged-model> --dtype half --max-model-len 4096 --max-num-seqs 32 --enforce-eager --served-model-name m
import asyncio, json, time, httpx, numpy as np
rows=[json.loads(l) for l in open("test.jsonl")][:200]
def P(t):
    return ("Summarize the following US congressional bill in a clear, factual legislative summary "
            "of at most 200 words. Output only the summary.\n\nBILL:\n"+t)
async def one(c,sem,r):
    async with sem:
        t=time.time()
        x=(await c.post("http://localhost:8000/v1/chat/completions",json={"model":"m",
            "messages":[{"role":"user","content":P(r["text"])}],"max_tokens":300,"temperature":0})).json()
        return {"pred":x["choices"][0]["message"]["content"],"ref":r["summary"],"latency":time.time()-t,
                "in_tok":x["usage"]["prompt_tokens"],"out_tok":x["usage"]["completion_tokens"]}
async def run(conc,n):
    sem=asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=900) as c:
        t=time.time(); res=await asyncio.gather(*[one(c,sem,r) for r in rows[:n]]); return res,time.time()-t
async def main():
    await run(1,2); summary={}
    for conc,n in [(1,50),(8,200),(32,200)]:
        res,wall=await run(conc,n); lat=[r["latency"] for r in res]
        summary[conc]={"n":n,"wall_s":round(wall,1),"p50":round(float(np.percentile(lat,50)),2),
            "p95":round(float(np.percentile(lat,95)),2),"tok_per_s":round(sum(r["out_tok"] for r in res)/wall,1),
            "req_per_s":round(n/wall,2)}
        print(conc,summary[conc],flush=True)
        with open(f"vllm_16bit_c{conc}.jsonl","w") as f:
            for r in res: f.write(json.dumps(r)+"\n")
    json.dump(summary,open("vllm_16bit_summary.json","w"))
asyncio.run(main())
