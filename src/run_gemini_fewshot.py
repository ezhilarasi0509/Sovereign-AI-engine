import json, time, sys
from dotenv import load_dotenv; load_dotenv(".env")
from google import genai
from prompt import load
M="gemini-3.1-flash-lite"
train=[json.loads(l) for l in open("data/train.jsonl")]
shots=sorted(train,key=lambda r:len(r["text"]))[:2]
assert len(shots)==2 and all(len(s["text"])>200 for s in shots)
print("shot sizes",[(len(s["text"]),len(s["summary"])) for s in shots],flush=True)
HDR=("Summarize the following US congressional bill in a clear, factual legislative summary of at most 200 words. "
     "Output only the summary. Two examples of the expected style follow.\n\n")
EX="".join(f"BILL:\n{s['text']}\n\nSUMMARY:\n{s['summary']}\n\n" for s in shots)
c=genai.Client()
import os
P="results/gemini-3.1-flash-lite-2shot.jsonl"
done=sum(1 for _ in open(P)) if os.path.exists(P) else 0
out=open(P,"a")
for i,r in enumerate(load("data/test.jsonl",200)):
    if i<done: continue
    x=None
    for a in range(8):
        try:
            t=time.time(); x=c.models.generate_content(model=M,contents=HDR+EX+f"BILL:\n{r['text']}\n\nSUMMARY:\n"); lat=time.time()-t; break
        except Exception as e:
            print("ERR",i,a,str(e)[:120].replace(chr(10)," "),flush=True); time.sleep(90)
    if x is None: print("giving up",i); sys.exit(1)
    u=x.usage_metadata
    out.write(json.dumps({"pred":x.text or "","ref":r["summary"],"latency":lat,"in_tok":u.prompt_token_count or 0,"out_tok":u.candidates_token_count or 0})+"\n"); out.flush()
    print(i,round(lat,1),flush=True); time.sleep(4)
