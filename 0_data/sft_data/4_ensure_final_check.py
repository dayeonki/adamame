import json
import os
import glob

BASE_DIR = "sft_data"

DIRS = {
    "trit_split": "output",
}

MARKER = "\n</think>\n"

total_files = 0
total_missing = 0

for dir_name, field in DIRS.items():
    dir_path = os.path.join(BASE_DIR, dir_name)
    files = sorted(glob.glob(os.path.join(dir_path, "*.jsonl")))

    print(f"\n{'='*60}")
    print(f"Directory: {dir_name}  (checking field: '{field}')")
    print(f"{'='*60}")

    for filepath in files:
        missing = []

        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(field, "")
                if MARKER not in text:
                    missing.append(i)

        fname = os.path.basename(filepath)
        if missing:
            print(f"  [FAIL] {fname}: {len(missing)} record(s) missing '\\n</think>\\n'")
            print(f"         Lines: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        else:
            print(f"  [OK]   {fname}: all records contain '\\n</think>\\n'")

        total_files += 1
        total_missing += len(missing)

print(f"\n{'='*60}")
print(f"Summary: {total_files} files checked, {total_missing} total missing records")
print(f"{'='*60}")
