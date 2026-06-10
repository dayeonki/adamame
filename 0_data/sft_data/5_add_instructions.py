import json
import os

SPLIT_DIR = os.path.join(os.path.dirname(__file__), "trit_split")

INSTRUCTIONS = {
    "fr": "Veuillez raisonner étape par étape et placer votre réponse finale dans \\boxed{}.\n\n",
    "ja": "段階的に考えて、最終的な答えを \\boxed{} の中に入れてください。\n\n",
    "ko": "단계별로 추론하고, 최종 답을 \\boxed{} 안에 넣어주세요.\n\n",
    "pt": "Por favor, justifique passo a passo e coloque a sua resposta final dentro de \\boxed{}.\n\n",
    "th": "กรุณาให้เหตุผลทีละขั้นตอน และใส่คำตอบสุดท้ายของคุณไว้ภายใน \\boxed{}\n\n",
}

for filename in os.listdir(SPLIT_DIR):
    if not filename.endswith(".jsonl"):
        continue

    # Extract language code from filename, e.g. "mthinker_fr.jsonl" -> "fr"
    lang = filename.replace("trit_", "").replace(".jsonl", "")
    if lang not in INSTRUCTIONS:
        print(f"Skipping {filename}: no instruction defined for language '{lang}'")
        continue

    prefix = INSTRUCTIONS[lang]
    filepath = os.path.join(SPLIT_DIR, filename)

    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["prompt"] = prefix + row["instruction"]
            rows.append(row)

    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Processed {filename}: {len(rows)} rows")
