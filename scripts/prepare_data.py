from datasets import load_dataset
import json, os
os.makedirs("data", exist_ok=True)
ds = load_dataset("FiscalNote/billsum")
def dump(split, n, name):
    rows = [r for r in ds[split] if len(r["text"]) < 12000][:n]
    with open(f"data/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "summary": r["summary"]}) + "\n")
dump("train", 2000, "train")
dump("test", 200, "test")
