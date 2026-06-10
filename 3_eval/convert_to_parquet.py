import argparse
import json
from pathlib import Path
import pandas as pd


PROMPT_LANG_MAP = {
    "fr": "Veuillez raisonner étape par étape et placer votre réponse finale dans \\boxed{{}}.\n\n{question}",
    "ja": "段階的に考えて、最終的な答えを \\boxed{{}} の中に入れてください。\n\n{question}",
    "ko": "단계별로 추론하고, 최종 답을 \\boxed{{}} 안에 넣어주세요.\n\n{question}",
    "pt": "Por favor, justifique passo a passo e coloque a sua resposta final dentro de \\boxed{{}}.\n\n{question}",
    "th": "กรุณาให้เหตุผลทีละขั้นตอน และใส่คำตอบสุดท้ายของคุณไว้ภายใน \\boxed{{}}\n\n{question}",
}


def transform_row(row: dict, split: str, idx: int, data_source: str) -> dict:
    if "prompt" in row and "reward_model" in row and "data_source" in row:
        out = dict(row)
        out.setdefault("extra_info", {"split": split, "index": idx})
        return out

    question = row.get("question")
    answer = row.get("answer")
    type = row.get("type")
    lang = row.get("lang", "unknown")

    if question is None or answer is None:
        raise ValueError(
            "Input row must contain either "
            "(prompt, reward_model, data_source) or (question, answer)."
        )

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": PROMPT_LANG_MAP[lang].format(question=question) if lang in PROMPT_LANG_MAP else str(question)}],
        "reward_model": {
            "style": "rule",
            "ground_truth": str(answer),
            "lang": str(lang),
        },
        "extra_info": {"split": split, "index": idx, "type": str(type)},
        "lang": str(lang),
    }


def convert_jsonl_to_parquet(
    src: Path,
    dst: Path,
    split: str,
    data_source: str,
) -> None:
    if not src.exists():
        print(f"Skip missing file: {src}")
        return

    rows = []
    with src.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(transform_row(row=row, split=split, idx=idx, data_source=data_source))

    df = pd.DataFrame(rows)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    print(f"{src} -> {dst} ({len(df)} rows)")
    print(json.dumps(df.iloc[0].to_dict(), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-source",
        type=str,
        default="zoey/gsm8k",
        help="Value used for each row's data_source field.",
    )
    args = parser.parse_args()

    script_dir = Path("")
    pairs = [
        (script_dir / "train_dapomath17k_1k.jsonl", script_dir / "train_dapomath17k_1k.parquet", "train"),
        (script_dir / "train_dapomath17k_5k.jsonl", script_dir / "train_dapomath17k_5k.parquet", "train"),
        (script_dir / "val_dapomath17k.jsonl", script_dir / "val_dapomath17k.parquet", "val"),
    ]

    for src, dst, split in pairs:
        convert_jsonl_to_parquet(src=src, dst=dst, split=split, data_source=args.data_source)


if __name__ == "__main__":
    main()
