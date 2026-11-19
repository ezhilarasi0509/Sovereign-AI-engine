import csv, re, subprocess, textwrap, sys, os, tempfile
SHEET="audit/sheet.csv"
rows=list(csv.DictReader(open(SHEET)))
def save():
    with open(SHEET,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["bill_no","bill_id","summary","unsupported_claims"]); w.writeheader(); w.writerows(rows)
def show(summ):
    for L,t in summ.items():
        print(f"\n--- Summary {L} ---"); print(textwrap.fill(t.strip(),100))
bills=sorted({int(r["bill_no"]) for r in rows})
for n in bills:
    mine=[r for r in rows if int(r["bill_no"])==n]
    if all(r["unsupported_claims"].strip()!="" for r in mine): continue
    md=open(f"audit/bill_{n:02d}.md").read()
    p=re.split(r"\n## Summary ([ABC])\n\n",md)
    summ={p[1]:p[2],p[3]:p[4]}
    tail=p[6].split("\n\n---\n\n# BILL TEXT\n\n",1)
    summ[p[5]]=tail[0]; bill=tail[1] if len(tail)>1 else ""
    summ=dict(sorted(summ.items()))
    print("\n"+"="*80+f"\nBILL {n} of {len(bills)}"); show(summ)
    input("\nPress Enter to open the bill (space = next page, /word = search, q = close it) ... ")
    tmp=tempfile.NamedTemporaryFile("w",suffix=".txt",delete=False); tmp.write(bill); tmp.close()
    subprocess.run(["less","-R",tmp.name]); os.unlink(tmp.name)
    for L in "ABC":
        r=[x for x in mine if x["summary"]==L][0]
        if r["unsupported_claims"].strip()!="": continue
        print(f"\nSummary {L}: {summ[L].strip()[:150]}...")
        while True:
            v=input(f"Unsupported claims in Summary {L} (a number, r = reread summaries, q = save and quit): ").strip().lower()
            if v=="q": save(); print("saved. Run the same command to resume."); sys.exit()
            if v=="r": show(summ); continue
            if v.isdigit(): r["unsupported_claims"]=v; break
            print("enter a whole number")
    save(); print("saved bill",n)
print("all bills done")
