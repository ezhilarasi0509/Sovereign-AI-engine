set -e
cd ~/sovereign-ai; source .venv/bin/activate
[ "$(wc -l < results/gemini-3.1-flash-lite-2shot.jsonl)" -ge 200 ] || { echo "2-shot run not finished"; exit 1; }
python -W ignore - <<'PYEOF' 2>&1 | grep "bert_gem2"
import json
from bert_score import score
rows=[json.loads(l) for l in open("results/gemini-3.1-flash-lite-2shot.jsonl")]
P,R,F=score([r["pred"] for r in rows],[r["ref"] for r in rows],model_type="distilbert-base-uncased",lang="en",verbose=False)
json.dump({"f1":round(F.mean().item(),3)},open("results/bert_gem2.json","w")); print("bert_gem2",round(F.mean().item(),3))
PYEOF
python - <<'PYEOF'
def patch(path,pairs):
    s=open(path).read()
    for a,b in pairs:
        assert a in s, path+" missing: "+a[:70]
        s=s.replace(a,b)
    open(path,"w").write(s)
patch("src/make_readme.py",[
 ('BQ8=json.load(open("results/bert_q8.json"))["f1"]','BQ8=json.load(open("results/bert_q8.json"))["f1"]\nBG2=json.load(open("results/bert_gem2.json"))["f1"]'),
 ('("gem","Gemini 3.1 Flash-Lite","Google API (frontier baseline)","results/gemini-3.1-flash-lite.jsonl",0.820),',
  '("gem","Gemini 3.1 Flash-Lite, zero-shot","Google API","results/gemini-3.1-flash-lite.jsonl",0.820),\n     ("gem2","Gemini 3.1 Flash-Lite, 2-shot","Google API (2 example pairs in the prompt)","results/gemini-3.1-flash-lite-2shot.jsonl",BG2),'),
 ('["base","gem","ft","q8","q4"]','["base","gem","gem2","ft","q8","q4"]'),
 ('than a frontier API baseline on this task','than a low-cost API baseline (Gemini 3.1 Flash-Lite) on this task'),
 ('against a frontier API on 200 unseen bills','against a low-cost API model on 200 unseen bills'),
 ('| Gemini 3.1 Flash-Lite | Google API | Frontier API baseline |','| Gemini 3.1 Flash-Lite, zero-shot | Google API | Low-cost API baseline |\n| Gemini 3.1 Flash-Lite, 2-shot | Google API | Same model with two example bill and summary pairs in the prompt |'),
 ('data may leave your network | Frontier API |','data may leave your network | Hosted API |'),
 ('- The fine-tuned model beat the Gemini 3.1 Flash-Lite API baseline on every ROUGE metric and on BERTScore.',
  '- The fine-tuned model beat the zero-shot Gemini 3.1 Flash-Lite baseline on every ROUGE metric and on BERTScore.\n- Giving Gemini two example bill and summary pairs in the prompt (2-shot) moved its ROUGE-L from {gem[2]:.3f} to {R["gem2"][2]:.3f}, against {ft[2]:.3f} for the fine-tuned 16-bit model. The fine-tuned model is {"still ahead" if ft[2]>R["gem2"][2] else "not ahead"} on ROUGE-L; section 4.7 gives the confidence interval.'),
])
patch("src/analysis_md.py",[
 ('"gemini":"Gemini 3.1 Flash-Lite",','"gemini":"Gemini 3.1 Flash-Lite (zero-shot)","gem2":"Gemini 3.1 Flash-Lite (2-shot)",'),
 ('"gemini":"results/gemini-3.1-flash-lite.jsonl",','"gemini":"results/gemini-3.1-flash-lite.jsonl","gem2":"results/gemini-3.1-flash-lite-2shot.jsonl",'),
 ('[("ft16","gemini"),("ft16","base"),','[("ft16","gemini"),("ft16","gem2"),("ft16","base"),'),
 ("The fine-tuned model's lead over Gemini and the base model is statistically supported. The 16-bit versus Q4_K_M gap is not distinguishable from noise on 200 bills.",
  "A confidence interval that includes zero means the difference is not distinguishable from noise on 200 bills. Read each row accordingly."),
 ("fine-tuned minus Gemini is","fine-tuned minus zero-shot Gemini is"),
])
patch("src/cost_sens.py",[
 ('open("results/cost_sens.md","w").write(md); print(md)',
  'import os\nif os.path.exists("results/gemini-3.1-flash-lite-2shot.jsonl"):\n    g2=[json.loads(l) for l in open("results/gemini-3.1-flash-lite-2shot.jsonl")]\n    a2=1000*(np.mean([r["in_tok"] for r in g2])*IN_P+np.mean([r["out_tok"] for r in g2])*OUT_P)/1e6\n    md=md.rstrip("\\n")+"\\n"+f"- If the 2-shot prompt is used (more input tokens), the API cost is ${a2:.3f} per 1,000 calls and the crossover at $0.50/hr falls to about {0.5*HRS/a2*1000:,.0f} calls/month.\\n\\n"\nopen("results/cost_sens.md","w").write(md); print(md)'),
])
print("patched all")
PYEOF
bash src/build_readme.sh
python src/score.py results/gemini-3.1-flash-lite-2shot.jsonl results/gemini-3.1-flash-lite.jsonl results/vllm_16bit_c8.jsonl
git add -A
git ls-files | xargs grep -I -l -E "AIza[0-9A-Za-z_-]{20,}|hf_[A-Za-z0-9]{20,}" 2>/dev/null || true
git commit -q -m "Add 2-shot Gemini baseline, correct baseline wording"
git push
git status -sb | head -2
