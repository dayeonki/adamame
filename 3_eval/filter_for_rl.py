import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
from metrics import compute_acc, extract_boxed, extract_answer_parts

# ---------------------------------------------------------
DATASET_TYPE = "dapomath17k"
MODEL_NAME = ""
# ---------------------------------------------------------

RES_DIR = os.path.join(EVAL_DIR, "res")

LANGUAGES = {
    "dapomath17k": ["fr", "ja", "ko", "pt", "th"],
}

def candidate_correct(gold: str, candidate: str) -> int:
    parts = extract_answer_parts(candidate)
    trace, response = parts["reasoning_trace"], parts["response"]
    return compute_acc(gold, extract_boxed(trace), extract_boxed(response))


def filter_file(input_path: str, output_path: str) -> tuple[int, int, float]:
    kept = total = 0
    correct_nums = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            candidates = item.get("candidates", [])
            gold = str(item.get("answer", ""))
            num_correct = sum(candidate_correct(gold, c) for c in candidates)
            if 1 <= num_correct < 8:
                item["correct_num"] = num_correct
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                correct_nums.append(num_correct)
                kept += 1
    avg_correct = sum(correct_nums) / len(correct_nums) if correct_nums else 0.0
    return kept, total, avg_correct


if __name__ == "__main__":
    langs = LANGUAGES.get(DATASET_TYPE)
    if langs is None:
        print(f"Unknown DATASET_TYPE '{DATASET_TYPE}'. Add it to LANGUAGES dict.")
        sys.exit(1)

    out_dir = os.path.join(RES_DIR, DATASET_TYPE, MODEL_NAME + "_filtered")
    total_kept = total_all = 0

    for lang in langs:
        in_path = os.path.join(RES_DIR, DATASET_TYPE, MODEL_NAME, f"{lang}.jsonl")
        out_path = os.path.join(out_dir, f"{lang}.jsonl")
        if not os.path.exists(in_path):
            print(f"[skip] {in_path} not found")
            continue
        kept, total, avg_correct = filter_file(in_path, out_path)
        total_kept += kept
        total_all += total
        print(f"[{lang}] {kept}/{total} kept ({kept/total*100:.1f}%) | avg correct_num={avg_correct:.3f}" if total else f"[{lang}] empty")

    print(f"\nTotal: {total_kept}/{total_all} kept -> {out_dir}")
