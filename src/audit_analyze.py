import csv, json, os, numpy as np
N={"ft16":"Fine-tuned 16-bit","gem2":"Gemini 3.1 Flash-Lite (2-shot)","q4":"Fine-tuned Q4_K_M"}
def wilson(k,n):
    p=k/n; z=1.96; d=1+z*z/n; c=p+z*z/(2*n); a=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-a)/d,(c+a)/d
if os.path.exists("audit/sheet.csv") and os.path.exists("audit/.key.json"):
    key=json.load(open("audit/.key.json")); cnt={}
    for r in csv.DictReader(open("audit/sheet.csv")):
        v=r["unsupported_claims"].strip()
        if v=="": continue
        cnt.setdefault(r["bill_id"],{})[key[r["bill_id"]][r["summary"]]]=int(float(v))
    full={b:c for b,c in cnt.items() if len(c)==3}
    if len(full)<5:
        print("Only",len(full),"complete bills. Fill in at least 5 (all three summaries each).")
    else:
        rng=np.random.default_rng(0); ids=list(full); n=len(ids)
        A={s:np.array([full[b][s] for b in ids]) for s in N}
        def boot(f):
            bs=[f(rng.integers(0,n,n)) for _ in range(5000)]; return np.percentile(bs,[2.5,97.5])
        md=f"### 4.10 Manual faithfulness audit\n\nOne reviewer checked {n} randomly chosen test bills. For each bill the three summaries were shown anonymously (labels A, B, C in shuffled order, each cut to its first 200 words) and the reviewer counted claims the bill does not support (amounts, dates, names, programs or requirements that are missing from or contradict the bill).\n\n"
        md+="| System | Summaries | Avg unsupported claims | Summaries with at least one (95% Wilson CI) |\n|---|---|---|---|\n"
        for s in N:
            k=int((A[s]>0).sum()); lo,hi=wilson(k,n)
            md+=f"| {N[s]} | {n} | {A[s].mean():.2f} | {k}/{n} = {k/n*100:.0f}% [{lo*100:.0f}%, {hi*100:.0f}%] |\n"
        md+="\nPaired difference in average unsupported claims per summary (bootstrap over bills; a negative value means fewer unsupported claims):\n\n| Comparison | Mean difference | 95% CI |\n|---|---|---|\n"
        for a,b in [("ft16","gem2"),("ft16","q4")]:
            lo,hi=boot(lambda i:(A[a][i]-A[b][i]).mean())
            md+=f"| {N[a]} minus {N[b]} | {(A[a]-A[b]).mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] |\n"
        md+=f"\nLimits: a single reviewer who knew which project this was, only {n} bills, and style cues can still hint at the system. Treat this as a spot check, not a measurement of overall factual accuracy.\n\n"
        open("results/audit.md","w").write(md)
        with open("results/audit_results.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["bill_id","system","unsupported_claims"])
            for b in ids:
                for s in N: w.writerow([b,s,full[b][s]])
        print(md)
if os.path.exists("results/audit.md"):
    s=open("README.md").read(); md=open("results/audit.md").read()
    if "### 4.10" not in s and "## 5. Which option should you pick?" in s:
        open("README.md","w").write(s.replace("## 5. Which option should you pick?",md+"## 5. Which option should you pick?")); print("README updated")
