import csv
import re
import numpy as np
import json
import os
from lingua import Language, LanguageDetectorBuilder
from math_verify import parse, verify
from collections import Counter

# ---------------------------------------------------------
DATASET_TYPE = "mgsmv2" # Change to mgsmv2 or msvamp
MODEL_NAMES = [
    "", # Add any model variants to evaluate
]
# ---------------------------------------------------------

def parse_time_str(t):
    if not t or not isinstance(t, str):
        return None
    try:
        parts = t.split(":")
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
    except Exception:
        pass
    return None


if "mgsm" in DATASET_TYPE:
    detector = LanguageDetectorBuilder.from_languages(
        Language.BENGALI, Language.GERMAN, Language.ENGLISH, Language.SPANISH,
        Language.FRENCH, Language.JAPANESE, Language.KOREAN, Language.RUSSIAN,
        Language.SWAHILI, Language.TELUGU, Language.THAI, Language.CHINESE,
    ).build()
else:
    detector = LanguageDetectorBuilder.from_languages(
        Language.ARABIC, Language.ENGLISH, Language.SPANISH, Language.FRENCH,
        Language.JAPANESE, Language.KOREAN, Language.PORTUGUESE, Language.THAI, 
        Language.VIETNAMESE, Language.CHINESE
    ).build()


LANGUAGE_ISO_MAP = {
    "Language.BENGALI": "bn",
    "Language.GERMAN": "de",
    "Language.ENGLISH": "en",
    "Language.SPANISH": "es",
    "Language.FRENCH": "fr",
    "Language.JAPANESE": "ja",
    "Language.KOREAN": "ko",
    "Language.RUSSIAN": "ru",
    "Language.SWAHILI": "sw",
    "Language.TELUGU": "te",
    "Language.THAI": "th",
    "Language.CHINESE": "zh",
    "Language.ARABIC": "ar",
    "Language.PORTUGUESE": "pt",
    "Language.VIETNAMESE": "vi",
}


def detect_lang(text):
    lang = str(detector.detect_language_of(text))
    return LANGUAGE_ISO_MAP.get(lang, None)


def _lingua_lang_label(raw, iso):
    if iso is not None:
        return iso
    return raw.replace("Language.", "").lower()


# Languages whose scripts are Latin (check for non-Latin char intrusion at word level)
LATIN_SCRIPT_LANGS = {"de", "en", "es", "fr", "id", "it", "pt", "sw", "tr", "vi"}
# Languages whose scripts are non-Latin (check for English/Latin word intrusion)
NON_LATIN_SCRIPT_LANGS = {"ar", "bn", "hi", "ja", "ko", "ru", "te", "th", "zh"}


def strip_math(text: str) -> str:
    # Display math: $$...$$ and \[...\]
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    # Inline math: $...$ and \(...\)
    text = re.sub(r"\$[^\$]+?\$", " ", text)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.DOTALL)
    # LaTeX environments: \begin{...}...\end{...}
    text = re.sub(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", " ", text, flags=re.DOTALL)
    # \boxed{...} and other common LaTeX commands with braces
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    # Bare numbers and arithmetic (sequences of digits/operators/punctuation only)
    text = re.sub(r"(?<!\w)[\d\+\-\*\/\=\^\.\,\(\)\[\]\{\}\%\_]+(?!\w)", " ", text)
    return text.strip()


def _is_latin_alpha(c: str) -> bool:
    if not c.isalpha():
        return True
    cp = ord(c)
    # Basic Latin + Latin-1 Supplement + Latin Extended A/B + Latin Extended Additional
    return cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF


def check_line_level(text: str, query_lang: str) -> bool:
    lines = [strip_math(l).strip() for l in text.split("\n")]
    for line in lines:
        if not line:
            continue
        detected = detect_lang(line)
        if detected is not None and detected != query_lang:
            return False
    return True


def check_word_level(text: str, query_lang: str) -> bool:
    text = strip_math(text)
    if query_lang in NON_LATIN_SCRIPT_LANGS:
        # Any run of 2+ Latin letters in a non-Latin-script trace signals confusion
        return not bool(re.search(r"[a-zA-Z]{2,}", text))
    if query_lang in LATIN_SCRIPT_LANGS:
        return all(_is_latin_alpha(c) for c in text)
    return True


def analyze_code_switching(text):
    empty = {
        "cs": False,
        "code_switching": False,
        "num_cs_segments": 0,
        "cs_languages": [],
        "cs_segments": [],
    }
    if not text or not str(text).strip():
        return empty
    try:
        text = strip_math(str(text))
        if not text:
            return empty
        results = list(detector.detect_multiple_languages_of(text))
    except Exception:
        return empty
    if not results:
        return empty

    cs_segments = []
    labels = []
    for r in results:
        raw = str(r.language)
        iso = LANGUAGE_ISO_MAP.get(raw, None)
        label = _lingua_lang_label(raw, iso)
        labels.append(label)
        cs_segments.append({
            "lang": iso,
            "lang_label": label,
            "start_index": r.start_index,
            "end_index": r.end_index,
        })

    distinct = sorted(set(labels))
    code_switching = len(set(labels)) >= 2
    return {
        "cs": code_switching,
        "code_switching": code_switching,
        "num_cs_segments": len(cs_segments),
        "cs_languages": distinct,
        "cs_segments": cs_segments,
    }


def compute_acc(gold_answer, trace_answer, response_answer):
    try:
        gold = parse(gold_answer)
    except Exception:
        return 0
    for candidate in (response_answer, trace_answer):
        if not candidate:
            continue
        try:
            parsed = parse(candidate)
            if verify(gold, parsed):
                return 1
        except Exception:
            continue
    return 0


def extract_boxed(text):
    # Use a brace-counting approach to handle nested braces (e.g. \boxed{\frac{3}{4}})
    idx = text.find(r"\boxed{")
    if idx == -1:
        return ""
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip()


def extract_answer_parts(output_text: str) -> dict:
    s = str(output_text)
    if "</think>" in s:
        thinking_content, answer_content = s.split("</think>", 1)
        thinking_content = (
            thinking_content
            .replace("<think>\n", "")
            .replace("<think>", "")
            .strip()
        )
        answer_content = answer_content.strip()
    else:
        thinking_content = ""
        answer_content = s.strip()
    return {"reasoning_trace": thinking_content, "response": answer_content}


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)

if __name__ == "__main__":
    languages = ["fr", "ja", "ko", "th", "bn", "en", "es", "ru", "sw", "te", "zh", "de"]

    for MODEL_NAME in MODEL_NAMES:
        print(f"\n{'='*60}\n[Model] {MODEL_NAME}\n{'='*60}")
        csv_rows = []

        for language in languages:
            print(f"[Language] {language}")
            try:
                input_file = os.path.join(SCRIPT_DIR, "res", DATASET_TYPE, MODEL_NAME, f"{language}.jsonl")

                pass_at_k_list = []
                acc_list = []
                time_list = []
                ttc_list = []
                lang_counter = Counter()
                lang_consistency_list = []
                cs_flags = []
                cs_mixtures = Counter()
                lpr_list = []
                wpr_list = []

                with open(input_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        if not line.strip():
                            continue

                        item = json.loads(line)
                        candidates = item.get("candidates", [])
                        fallback_trace = str(item.get("reasoning_trace", ""))
                        fallback_response = str(item.get("response", ""))

                        if candidates:
                            parsed = [extract_answer_parts(c) for c in candidates]
                            traces_and_responses = [
                                (p["reasoning_trace"], p["response"]) for p in parsed
                            ]
                            avg_tokens = np.mean([len(c) / 4 for c in candidates])
                        else:
                            print("No candidates found!")
                            trace_only = (
                                fallback_trace
                                .replace("<think>\n", "")
                                .replace("\n</think>", "")
                                .replace("<think>", "")
                            )
                            traces_and_responses = [(trace_only, fallback_response)]
                            avg_tokens = len(fallback_trace + fallback_response) / 4
                        ttc_list.append(avg_tokens / 8192)

                        candidates_info = []
                        for trace, _ in traces_and_responses:
                            lang_code = detect_lang(trace)
                            cs_info = analyze_code_switching(trace)
                            line_pass = check_line_level(trace, language)
                            word_pass = check_word_level(trace, language) if line_pass else None
                            candidates_info.append({
                                "lang_code": lang_code,
                                "cs": cs_info["cs"],
                                "code_switching": cs_info["code_switching"],
                                "num_cs_segments": cs_info["num_cs_segments"],
                                "cs_languages": cs_info["cs_languages"],
                                "line_pass": line_pass,
                                "word_pass": word_pass,
                            })

                        gold_answer = item.get("answer", "")
                        candidate_accs = [
                            compute_acc(gold_answer, extract_boxed(trace), extract_boxed(response))
                            for trace, response in traces_and_responses
                        ]
                        pass_at_k_list.append(1 if any(a == 1 for a in candidate_accs) else 0)
                        acc_list.append(candidate_accs[0])

                        cs_flags.append(np.mean([1 if c["cs"] else 0 for c in candidates_info]))
                        lang_consistency_list.append(
                            np.mean([1 if c["lang_code"] == language else 0 for c in candidates_info])
                        )
                        lpr_list.append(np.mean([1 if c["line_pass"] else 0 for c in candidates_info]))
                        word_scores = [c["word_pass"] for c in candidates_info if c["word_pass"] is not None]
                        if word_scores:
                            wpr_list.append(np.mean([1 if w else 0 for w in word_scores]))

                        for c in candidates_info:
                            lang_counter[c["lang_code"]] += 1
                        majority_cs_langs = Counter(
                            tuple(c["cs_languages"]) for c in candidates_info if c["cs"]
                        ).most_common(1)
                        if majority_cs_langs:
                            cs_mixtures[majority_cs_langs[0][0]] += 1

                        t_str = item.get("time", "")
                        t_sec = parse_time_str(t_str)
                        if t_sec is not None:
                            time_list.append(t_sec)

                pass_at_4 = np.mean(pass_at_k_list) if pass_at_k_list else 0.0
                pass_at_1 = np.mean(acc_list) if acc_list else 0.0
                avg_time = np.mean(time_list) if time_list else 0.0
                code_switch_rate = np.mean(cs_flags) if cs_flags else 0.0
                language_consistency = np.mean(lang_consistency_list) if lang_consistency_list else 0.0
                ttc_rate = np.mean(ttc_list) if ttc_list else 0.0
                lpr = np.mean(lpr_list) if lpr_list else 0.0
                wpr = np.mean(wpr_list) if wpr_list else 0.0
                lcpr = (2 * lpr * wpr / (lpr + wpr)) if (lpr + wpr) > 0 else 0.0

                if cs_mixtures:
                    _parts = [f"{'+'.join(k)}: {v}" for k, v in cs_mixtures.most_common()]
                    cs_mix_str = "{" + ", ".join(_parts) + "}"
                else:
                    cs_mix_str = "{}"
                csv_rows.append({
                    "language": language,
                    "pass@1": round(pass_at_1, 3),
                    "pass@4": round(pass_at_4, 3),
                    "language_consistency": round(language_consistency, 3),
                    "code_switch_rate": round(code_switch_rate, 3),
                    "LPR": round(lpr, 3),
                    "WPR": round(wpr, 3),
                    "LCPR": round(lcpr, 3),
                    "ttc_rate": round(ttc_rate, 3),
                    "avg_time": round(avg_time, 2),
                    "cs_mixtures": cs_mix_str,
                })
            except Exception as e:
                print(f"[Error] {language}: {e}")
                continue

        csv_path = os.path.join(SCRIPT_DIR, "summary", f"{DATASET_TYPE}_{MODEL_NAME}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = ["language", "pass@1", "pass@4", "language_consistency", "code_switch_rate", "LPR", "WPR", "LCPR", "ttc_rate", "avg_time", "cs_mixtures"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved summary CSV -> {csv_path}")

        # --- summary table: metrics as rows, languages as columns ---
        TABLE_METRICS = [
            "pass@1", "pass@4",
            "language_consistency", "code_switch_rate",
            "LPR", "WPR", "LCPR",
            "ttc_rate", "avg_time",
        ]
        langs_done = [row["language"] for row in csv_rows]
        label_w = max(len(m) for m in TABLE_METRICS) + 2
        col_w = 8
        header = f"{'Metric':<{label_w}}" + "".join(f"{lg:>{col_w}}" for lg in langs_done)
        sep = "-" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep)
        for metric in TABLE_METRICS:
            row_str = f"{metric:<{label_w}}"
            for row in csv_rows:
                val = row.get(metric, "")
                row_str += f"{val:>{col_w}}"
            print(row_str)
        print(sep)
