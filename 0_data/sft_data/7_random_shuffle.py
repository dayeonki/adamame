import json
import os
import glob
import random

SEED = 42
BASE_DIR = "sft_data"
OUT_DIR = os.path.join(BASE_DIR, "final")
os.makedirs(OUT_DIR, exist_ok=True)


def load_all(dir_path):
    records = []
    for filepath in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
        data = json.load(open(filepath, encoding="utf-8"))
        print(f"  Loaded {os.path.basename(filepath):35s} ({len(data)} records)")
        records.extend(data)
    return records


def save(records, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {os.path.basename(out_path)} ({len(records)} records)")


rng = random.Random(SEED)

# trit_only.json
print("\n[trit_only]")
trit = load_all(os.path.join(BASE_DIR, "trit_llama"))
rng.shuffle(trit)
save(trit, os.path.join(OUT_DIR, "trit_only.json"))
