import json, time, sys, httpx
from prompt import load, prompt
N=int(sys.argv[1]) if len(sys.argv)>1 else 200
model=sys.argv[2] if len(sys.argv)>2 else "qwen2.5:3b"
out=open(f"results/{model.replace(':','_')}.jsonl","w")
for i,r in enumerate(load("data/test.jsonl",N)):
    t=time.time()
    x=httpx.post("http://localhost:11434/api/chat",timeout=600,json={
        "model":model,"stream":False,"options":{"num_ctx":8192,"temperature":0},
        "messages":[{"role":"user","content":prompt(r["text"])}]}).json()
    lat=time.time()-t
    out.write(json.dumps({"pred":x["message"]["content"],"ref":r["summary"],"latency":lat,
        "in_tok":x.get("prompt_eval_count",0),"out_tok":x.get("eval_count",0)})+"\n"); out.flush()
    print(i,round(lat,1))
