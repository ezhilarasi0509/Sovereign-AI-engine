import json, time, httpx, numpy as np
from prompt import load, prompt
bills=load("data/test.jsonl",30)
def run(m):
    lat=[];tok=[]
    for i,r in enumerate(bills):
        t=time.time()
        x=httpx.post("http://localhost:11434/api/chat",timeout=600,json={"model":m,"stream":False,
            "options":{"num_ctx":8192,"temperature":0,"num_predict":300},
            "messages":[{"role":"user","content":prompt(r["text"])}]}).json()
        dt=time.time()-t
        if i>0: lat.append(dt); tok.append(x.get("eval_count",0))
    return lat,tok
R={m:{"lat":[],"tok":[]} for m in ["billsum-q4b","billsum-q8"]}
for rnd in range(2):
    for m in R:
        l,t=run(m); R[m]["lat"]+=l; R[m]["tok"]+=t; print("round",rnd,m,flush=True)
out={}
for m,v in R.items():
    out[m]={"n":len(v["lat"]),"p50_s":round(float(np.percentile(v["lat"],50)),2),"p95_s":round(float(np.percentile(v["lat"],95)),2),
            "tok_per_s":round(sum(v["tok"])/sum(v["lat"]),1)}
    print(m,out[m])
json.dump(out,open("results/speed_mac.json","w"))
