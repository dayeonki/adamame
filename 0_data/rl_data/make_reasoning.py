import argparse
import json
import os
import time
from pathlib import Path
import tiktoken
from openai import OpenAI


# USD per 1M tokens
PRICE_INPUT_PER_1M = 0.05
PRICE_OUTPUT_PER_1M = 0.40
os.environ["OPENAI_API_KEY"] = "" 

PROMPT_LANG_MAP = {
    "en": "Please reason step by step, and put your final answer within \\boxed{{}}.\{question}",
    "bn": "অনুগ্রহ করে ধাপে ধাপে যুক্তি দেখান, এবং আপনার চূড়ান্ত উত্তরের চারপাশে \\boxed{{}} ব্যবহার করুন।\n{question}",
    "de": "Bitte denken Sie Schritt für Schritt nach und schreiben Sie Ihre endgültige Antwort in \\boxed{{}}.\n{question}",
    "es": "Por favor, razona paso a paso y coloca tu respuesta final dentro de \\boxed{{}}.\n{question}",
    "fr": "Veuillez raisonner étape par étape et placer votre réponse finale dans \\boxed{{}}.\n{question}",
    "ja": "段階的に考えて、最終的な答えを \\boxed{{}} の中に入れてください。\n{question}",
    "ko": "단계별로 추론하고, 최종 답을 \\boxed{{}} 안에 넣어주세요.\n{question}",
    "ru": "Пожалуйста, рассуждайте шаг за шагом и поместите свой окончательный ответ в \\boxed{{}}.\n{question}",
    "sw": "Tafadhali toa hoja hatua kwa hatua, na weka jibu lako la mwisho ndani ya \\boxed{{}}.\n{question}",
    "pt": "Por favor, justifique passo a passo e coloque a sua resposta final dentro de \\boxed{{}}.\n{question}",
    "te": "దయచేసి దశలవారీగా ఆలోచించండి, మరియు మీ తుది సమాధానాన్ని \\boxed{{}} లో వ్రాయండి.\n{question}",
    "th": "กรุณาให้เหตุผลทีละขั้นตอน และใส่คำตอบสุดท้ายของคุณไว้ภายใน \\boxed{{}}\n{question}",
    "zh": "请逐步推理，并将最终答案写在 \\boxed{{}} 中。\n{question}",
}


def get_encoder(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def compute_cost(input_tokens: int, output_tokens: int):
    input_cost = input_tokens * PRICE_INPUT_PER_1M / 1_000_000
    output_cost = output_tokens * PRICE_OUTPUT_PER_1M / 1_000_000
    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL path (e.g., SFT/trit/fr.jsonl)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--prompt-lang", default="fr")
    parser.add_argument("--max-output-tokens", type=int, default=50000)
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_template = PROMPT_LANG_MAP[args.prompt_lang]
    enc = get_encoder(args.model)
    client = OpenAI()

    totals = {
        "rows": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost_usd": 0.0,
    }
    processed_ids = set()

    # Resume mode: if output already exists, read processed ids and append new rows.
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fprev:
            for prev_line_idx, prev_line in enumerate(fprev, start=1):
                prev_line = prev_line.strip()
                if not prev_line:
                    continue
                try:
                    prev_row = json.loads(prev_line)
                    prev_id = prev_row.get("id")
                    if prev_id is not None:
                        processed_ids.add(str(prev_id))
                except json.JSONDecodeError:
                    print(f"[WARN] Skipping malformed output line {prev_line_idx}: {out_path}")
        print(f"[INFO] Found {len(processed_ids)} already processed ids in {out_path}")

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("a", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin, start=1):
            row = json.loads(line)
            row_id = str(row.get("id", str(line_idx)))
            if row_id in processed_ids:
                continue

            question = row["query"]
            prompt = prompt_template.format(question=question)
            print("=" * 20, " ✏️ Prompt ✏️ ", "=" * 20)
            print(prompt)

            start_time = time.time()
            # Estimate prompt tokens pre-request with tiktoken
            est_input_tokens = len(enc.encode(prompt))

            try:
                resp = client.responses.create(
                    model=args.model,
                    reasoning={"effort": args.reasoning_effort},
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=args.max_output_tokens,
                )
                elapsed = time.time() - start_time
                minutes, seconds = divmod(int(elapsed), 60)

                print("=" * 20, " 💭 Thinking & Answer 💭 ", "=" * 20)
                print(resp.output_text)

                usage = getattr(resp, "usage", None)
                if usage is not None:
                    input_tokens = int(getattr(usage, "input_tokens", est_input_tokens) or est_input_tokens)
                    output_tokens = int(getattr(usage, "output_tokens", len(enc.encode(resp.output_text))) or 0)
                else:
                    input_tokens = est_input_tokens
                    output_tokens = len(enc.encode(resp.output_text))

                costs = compute_cost(input_tokens, output_tokens)

                out_obj = {
                    "id": row_id,
                    "language": row.get("language", args.prompt_lang),
                    "query": question,
                    "en_query": row.get("en_query"),
                    "reward_model": row.get("reward_model"),
                    "prompt": prompt,
                    "response": resp.output_text,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    "cost_estimate_usd": costs,
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                processed_ids.add(row_id)

                totals["rows"] += 1
                totals["input_tokens"] += input_tokens
                totals["output_tokens"] += output_tokens
                totals["total_cost_usd"] += costs["total_cost_usd"]

                print("=" * 20, " ⏱️ Generation time (single) ⏱️ ", "=" * 20)
                print(f"{minutes:02d}:{seconds:02d}")

                print(
                    f"Input tokens: {input_tokens} Output tokens: {output_tokens} \n"
                    f"💸 Cost: ${costs['total_cost_usd']:.6f} 💸"
                )

                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)
            
            except:
                continue


    print("\n" + "=" * 20, " 💰 Total cost 💰 ", "=" * 20)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
