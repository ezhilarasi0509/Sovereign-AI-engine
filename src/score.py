import json, sys, numpy as np
from rouge_score import rouge_scorer
s=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True)
for f in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f)]
    sc=[s.score(r["ref"],r["pred"]) for r in rows]
    lat=[r["latency"] for r in rows]
    print(f, len(rows),
      "R1 %.3f R2 %.3f RL %.3f"%tuple(np.mean([x[k].fmeasure for x in sc]) for k in ["rouge1","rouge2","rougeL"]),
      "p50 %.1fs p95 %.1fs"%(np.percentile(lat,50),np.percentile(lat,95)),
      "avg_in %d avg_out %d"%(np.mean([r["in_tok"] for r in rows]),np.mean([r["out_tok"] for r in rows])))
