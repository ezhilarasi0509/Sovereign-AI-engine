import json, time, re, sys, numpy as np
from dotenv import load_dotenv; load_dotenv(".env")
from google import genai
from prompt import load
JM=sys.argv[1] if len(sys.argv)>1 else "gemini-3.1-flash-lite"
N=50
bills=load("data/test.jsonl",N)
c=genai.Client()
Q=("You grade summaries of US congressional bills.\nBILL:\n{b}\n\nREFERENCE SUMMARY:\n{ref}\n\nCANDIDATE SUMMARY:\n{c}\n\n"
   "Rate the CANDIDATE. faithfulness: 1-5 (5 = every claim is supported by the bill, no invented facts). "
   "coverage: 1-5 (5 = covers the key provisions in the reference). "
   'Reply with JSON only: {{"faithfulness":n,"coverage":n}}')
out={}
for name,f in [("finetuned_16bit","results/finetuned_t4.jsonl"),("q4_k_m","results/billsum-q4b.jsonl"),
               ("base_qwen","results/qwen2.5_3b.jsonl"),("gemini","results/gemini-3.1-flash-lite.jsonl")]:
    rows=[json.loads(l) for l in open(f)][:N]; sc=[]
    for b,r in zip(bills,rows):
        for a in range(4):
            try:
                t=c.models.generate_content(model=JM,contents=Q.format(b=b["text"],ref=r["ref"],c=r["pred"])).text
                j=json.loads(re.search(r"\{.*\}",t,re.S).group(0)); sc.append((j["faithfulness"],j["coverage"])); break
            except Exception as e:
                time.sleep(15)
        time.sleep(3)
    s=np.array(sc); out[name]={"n":len(sc),"faithfulness":round(float(s[:,0].mean()),2),"coverage":round(float(s[:,1].mean()),2)}
    print(name,out[name],flush=True)
json.dump(out,open("results/judge.json","w"))
