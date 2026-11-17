import json, sys, numpy as np
from bert_score import score
for f in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f)]
    P,R,F=score([r["pred"] for r in rows],[r["ref"] for r in rows],
                model_type="distilbert-base-uncased",lang="en",verbose=False)
    print(f, len(rows), "BERTScore F1 %.3f"%F.mean().item(), flush=True)
