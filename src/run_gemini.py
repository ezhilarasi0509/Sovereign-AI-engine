import json, time, sys
from dotenv import load_dotenv; load_dotenv(".env")
from google import genai
from prompt import load, prompt
N=int(sys.argv[1]) if len(sys.argv)>1 else 200
model=sys.argv[2] if len(sys.argv)>2 else "gemini-3.8-flash"
c=genai.Client()
out=open(f"results/{model}.jsonl","w")
for i,r in enumerate(load("data/test.jsonl",N)):
    x=None
    for attempt in range(4):
        try:
            t=time.time()
            x=c.models.generate_content(model=model,contents=prompt(r["text"]))
            lat=time.time()-t; break
        except Exception as e:
            print("ERR",i,attempt,str(e)[:400],flush=True); time.sleep(30)
    if x is None:
        print("giving up on",i,flush=True); sys.exit(1)
    u=x.usage_metadata
    out.write(json.dumps({"pred":x.text or "","ref":r["summary"],"latency":lat,
        "in_tok":u.prompt_token_count or 0,"out_tok":u.candidates_token_count or 0})+"\n"); out.flush()
    print(i,round(lat,1),flush=True); time.sleep(5)
