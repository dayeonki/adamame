import json
import os
import glob

input_dir = "sft_data/trit_split"

for filepath in glob.glob(os.path.join(input_dir, "*.jsonl")):
    updated_lines = []
    changed = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("instruction", "").startswith("## "):
                obj["instruction"] = obj["instruction"][3:]
                changed += 1
            updated_lines.append(json.dumps(obj, ensure_ascii=False))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")

    print(f"{os.path.basename(filepath)}: {changed} records updated")
