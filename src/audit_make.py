import json, random, os, csv
random.seed(1)
bills=[json.loads(l) for l in open("data/test.jsonl")][:200]
SYS={"ft16":"results/vllm_16bit_c8.jsonl","gem2":"results/gemini-3.1-flash-lite-2shot.jsonl","q4":"results/billsum-q4b.jsonl"}
D={k:[json.loads(l) for l in open(p)] for k,p in SYS.items()}
ids=sorted(random.sample(range(200),15))
os.makedirs("audit",exist_ok=True)
key={}; rows=[]
RULE=("Read the bill (use Cmd+F to search it). For EACH summary below, count the factual claims that the bill does NOT support: "
      "an amount, date, number, program, agency, name or requirement that is missing from the bill or contradicts it. "
      "Do not count omissions or style. Write the counts in audit/sheet.csv (one row per summary, a number such as 0, 1, 2).\n")
for n,i in enumerate(ids,1):
    order=list(SYS); random.shuffle(order)
    labels=dict(zip("ABC",order)); key[str(i)]=labels
    md=f"# Audit {n} of 15 (bill id {i})\n\n{RULE}\n"
    for L in "ABC":
        md+=f"## Summary {L}\n\n"+" ".join(D[labels[L]][i]["pred"].split()[:200])+"\n\n"
    md+="---\n\n# BILL TEXT\n\n"+bills[i]["text"]+"\n"
    open(f"audit/bill_{n:02d}.md","w").write(md)
    rows+=[[n,i,L,""] for L in "ABC"]
with open("audit/sheet.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["bill_no","bill_id","summary","unsupported_claims"]); w.writerows(rows)
json.dump(key,open("audit/.key.json","w"))
print("created",len(ids),"bills,",len(rows),"rows")
