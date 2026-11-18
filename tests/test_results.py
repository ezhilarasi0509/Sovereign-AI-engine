import glob, json, sys
sys.path.insert(0, "src")
from prompt import prompt
KEYS={"pred","ref","latency","in_tok","out_tok"}
CORE=["results/qwen2.5_3b.jsonl","results/gemini-3.1-flash-lite.jsonl","results/gemini-3.1-flash-lite-2shot.jsonl",
      "results/finetuned_t4.jsonl","results/vllm_16bit_c8.jsonl","results/vllm_16bit_c32.jsonl",
      "results/billsum-q4b.jsonl","results/billsum-q8.jsonl"]
def rows(f): return [json.loads(l) for l in open(f)]
def test_prompt():
    p=prompt("example"); assert "BILL:" in p and "200 words" in p and p.endswith("example")
def test_result_files_have_expected_keys():
    files=glob.glob("results/*.jsonl"); assert files
    for f in files:
        r=rows(f); assert r and all(KEYS <= x.keys() for x in r), f
def test_core_runs_cover_200_bills():
    for f in CORE: assert len(rows(f))==200, f
def test_same_reference_summaries_across_runs():
    base=[x["ref"] for x in rows(CORE[0])]
    for f in CORE[1:]: assert [x["ref"] for x in rows(f)]==base, f
