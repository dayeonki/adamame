import json
import os
from collections import defaultdict

data_dir = os.path.dirname(os.path.abspath(__file__))
THINK_END = "\n</think>"

results = defaultdict(list)

for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".jsonl"):
        continue

    lang = filename.split("_")[-1].replace(".jsonl", "")
    filepath = os.path.join(data_dir, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            response = row.get("output", "")
            idx = response.find(THINK_END)
            if idx != -1:
                trace = response[:idx]
            else:
                trace = response  # no </think> found; treat full response as trace
            results[lang].append(len(trace))

print(f"{'Lang':<6} {'Count':>7} {'Min':>7} {'Max':>8} {'Mean':>10} {'Total':>12}")
print("-" * 55)
for lang in sorted(results):
    lengths = results[lang]
    n = len(lengths)
    total = sum(lengths)
    mean = total / n if n > 0 else 0
    print(f"{lang:<6} {n:>7} {min(lengths):>7} {max(lengths):>8} {mean:>10.1f} {total:>12}")
