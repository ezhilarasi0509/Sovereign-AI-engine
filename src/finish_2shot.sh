set -e
cd ~/sovereign-ai; source .venv/bin/activate
[ "$(wc -l < results/gemini-3.1-flash-lite-2shot.jsonl)" -ge 200 ] || { echo "2-shot run not finished"; exit 1; }
python - <<'PYEOF'
import json, numpy as np, sys
a=[json.loads(l) for l in open("results/gemini-3.1-flash-lite.jsonl")]
b=[json.loads(l) for l in open("results/gemini-3.1-flash-lite-2shot.jsonl")]
d=np.mean([y["in_tok"]-x["in_tok"] for x,y in zip(a,b)]); print("mean extra input tokens in 2-shot:",round(d))
sys.exit(0 if d>100 else 1)
PYEOF
python -W ignore - <<'PYEOF' 2>&1 | grep "bert_gem2"
import json
from bert_score import score
rows=[json.loads(l) for l in open("results/gemini-3.1-flash-lite-2shot.jsonl")]
P,R,F=score([r["pred"] for r in rows],[r["ref"] for r in rows],model_type="distilbert-base-uncased",lang="en",verbose=False)
json.dump({"f1":round(F.mean().item(),3)},open("results/bert_gem2.json","w")); print("bert_gem2",round(F.mean().item(),3))
PYEOF
rm -f docs/banner.md
bash src/build_readme.sh
python src/score.py results/gemini-3.1-flash-lite-2shot.jsonl results/gemini-3.1-flash-lite.jsonl results/vllm_16bit_c8.jsonl
git add -A
git ls-files | xargs grep -I -l -E "AIza[0-9A-Za-z_-]{20,}|hf_[A-Za-z0-9]{20,}" 2>/dev/null || true
git commit -q -m "Fix 2-shot baseline (examples were missing), add temperature note"
git push
git status -sb | head -2
