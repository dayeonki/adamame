import json
import os
import glob

BASE_DIR = "sft_data"

CONFIGS = {
    "trit_split": {
        "prompt_field": "prompt",
        "response_field": "output",
        "out_dir": os.path.join(BASE_DIR, "trit_llama"),
    },
}

for dir_name, cfg in CONFIGS.items():
    in_dir = os.path.join(BASE_DIR, dir_name)
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "*.jsonl")))
    print(f"\n{'='*55}")
    print(f"Processing: {dir_name} -> {os.path.basename(out_dir)}")
    print(f"{'='*55}")

    for filepath in files:
        records = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append({
                    "instruction": obj[cfg["prompt_field"]],
                    "input": "",
                    "output": obj[cfg["response_field"]],
                })

        fname = os.path.splitext(os.path.basename(filepath))[0] + ".json"
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"  {os.path.basename(filepath):35s} -> {fname}  ({len(records)} records)")
