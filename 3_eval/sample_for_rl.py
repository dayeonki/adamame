import argparse
import json
import os
import random

# ---------------------------------------------------------
DATASET_NAME = "dapomath17k"
MODEL_NAME   = ""
LANGUAGES    = ["fr", "ja", "ko", "pt", "th"]
# ---------------------------------------------------------

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(EVAL_DIR, "res")


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="train samples per language")
    parser.add_argument("--m", type=int, default=0, help="val samples per language (non-overlapping with train)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    seen = set()
    train_rows, val_rows = [], []

    for lang in LANGUAGES:
        path = os.path.join(RES_DIR, DATASET_NAME, MODEL_NAME, f"{lang}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue

        rows = load_jsonl(path)
        rng.shuffle(rows)

        train_lang, val_lang = [], []
        for row in rows:
            q = row["question"]
            if q in seen:
                continue
            seen.add(q)
            if len(train_lang) < args.n:
                train_lang.append({"question": q, "answer": row["answer"], "lang": lang})
            elif len(val_lang) < args.m:
                val_lang.append({"question": q, "answer": row["answer"], "lang": lang})
            else:
                break

        if len(train_lang) < args.n:
            print(f"[warn] {lang}: only {len(train_lang)} train rows (wanted {args.n})")
        if len(val_lang) < args.m:
            print(f"[warn] {lang}: only {len(val_lang)} val rows (wanted {args.m})")

        train_rows.extend(train_lang)
        val_rows.extend(val_lang)
        print(f"[{lang}] {len(rows)} total -> train={len(train_lang)}, val={len(val_lang)}")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    out_dir = os.path.join(RES_DIR, DATASET_NAME)
    train_path = os.path.join(out_dir, f"train_{DATASET_NAME}_{args.n * len(LANGUAGES)}.jsonl")
    val_path   = os.path.join(out_dir, f"val_{DATASET_NAME}_{args.m * len(LANGUAGES)}.jsonl")

    save_jsonl(train_rows, train_path)
    save_jsonl(val_rows, val_path)

    print(f"\nTrain: {len(train_rows)} rows -> {train_path}")
    print(f"Val:   {len(val_rows)} rows -> {val_path}")
