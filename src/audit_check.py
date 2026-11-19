import re, sys, textwrap
n=int(sys.argv[1]); md=open(f"audit/bill_{n:02d}.md").read()
p=re.split(r"\n## Summary ([ABC])\n\n",md)
summ={p[1]:p[2],p[3]:p[4]}
tail=p[6].split("\n\n---\n\n# BILL TEXT\n\n",1); summ[p[5]]=tail[0]
raw=tail[1].lower(); nc=raw.replace(",","")
for L in "ABC":
    t=summ[L].strip()
    nums=sorted({x.rstrip(".,").replace(",","") for x in re.findall(r"\d[\d,]*\.?\d*",t)})
    words=sorted({w.lower() for w in re.findall(r"[A-Za-z]{6,}",t)})
    print(f"\n=== Summary {L} ===\n"+textwrap.fill(t,100))
    print("  numbers not in bill:",[x for x in nums if x not in nc] or "none")
    print("  words not in bill:  ",[w for w in words if w[:6] not in raw] or "none")
