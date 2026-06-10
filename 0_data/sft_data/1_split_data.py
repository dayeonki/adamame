import json
import os
import re
from lingua import Language, LanguageDetectorBuilder
from math_verify import parse, verify


def strip_math(text: str) -> str:
    # remove display math \[...\] and \(...\)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.DOTALL)
    # remove $...$ and $$...$$
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^\$]+?\$", " ", text)
    # remove LaTeX commands and their braced args e.g. \frac{}{}, \sqrt{}
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    # remove inline math-like tokens: numbers, variables, operators
    text = re.sub(r"\b[a-zA-Z0-9_^+\-=<>*/|]+\b", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

LANGUAGES = [Language.FRENCH, Language.JAPANESE, Language.KOREAN, Language.PORTUGUESE, Language.THAI]
LANG_CODE_MAP = {lang.iso_code_639_1.name.lower(): lang for lang in LANGUAGES}
detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()

input_dir = "sft_data/trit"
output_dir = "sft_data/trit_split"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".jsonl"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    replaced = 0
    skipped = 0
    no_boxed = 0
    wrong_answer = 0
    wrong_lang = 0

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            d = json.loads(line)
            response = d["response"]
            idx = response.rfind("\n\n")
            if idx == -1:
                skipped += 1
                continue

            new_response = response[:idx] + "\n</think>\n" + response[idx + 2:]
            after_think = new_response.split("\n</think>\n", 1)[-1]

            if "\\boxed{" not in after_think:
                no_boxed += 1
                continue

            ground_truth = d["reward_model"]["ground_truth"]
            try:
                gold = parse(ground_truth)
                pred = parse(after_think)
                if not verify(gold, pred):
                    wrong_answer += 1
                    continue
            except Exception:
                wrong_answer += 1
                continue

            trace = new_response.split("\n</think>\n", 1)[0]
            detected = detector.detect_language_of(strip_math(trace))
            detected_code = detected.iso_code_639_1.name.lower() if detected else None
            d["reward_model"]["language"] = detected_code
            if detected_code != d["language"]:
                wrong_lang += 1
                continue

            d["response"] = new_response
            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
            replaced += 1

    print(
        f"{filename}: {replaced} kept, "
        f"{skipped} dropped (no \\n\\n), "
        f"{no_boxed} dropped (no \\boxed{{}}), "
        f"{wrong_answer} dropped (wrong answer), "
        f"{wrong_lang} dropped (language mismatch)"
    )
