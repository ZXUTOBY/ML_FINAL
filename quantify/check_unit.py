"""
ner_unit_counter_split.py

Build a frequency dictionary of NER-to-unit-token counts (see comments in the
previous version) and then split the result into two JSON files:

    • ner_unit_freq_ge100.json  – counts ≥ 100
    • ner_unit_freq_lt100.json  – counts  < 100
"""

import json
import re
from collections import defaultdict
from pathlib import Path

THRESHOLD = 100                      # the split point


# ───────────────────────── helper loaders ──────────────────────────
def load_ner_terms(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


# ───────────────────────── unit-token extraction ───────────────────
def unit_token(tokens: list[str]) -> str | None:
    if len(tokens) < 2:
        return None

    second = tokens[1]

    if second.startswith("("):                       # special case
        for tok in tokens[1:]:
            if re.search(r"\boz\.?", tok, flags=re.I):
                return "oz."
        return re.sub(r"[()\.,]", "", second) or None

    return second                                    # normal case


# ───────────────────────── core aggregation ────────────────────────
def build_freq(ner_set: set[str], recipes_path: str) -> dict[str, dict[str, int]]:
    freq = defaultdict(lambda: defaultdict(int))

    with open(recipes_path, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    for recipe in recipes:
        ing_lines = recipe.get("ingredients", [])
        for ner in recipe.get("NER", []):
            ner_lc = ner.lower()
            if ner_lc not in ner_set:
                continue

            match_line = next((l for l in ing_lines if ner_lc in l.lower()), None)
            if not match_line:
                continue

            tok = unit_token(match_line.split())
            if tok:
                freq[ner_lc][tok] += 1

    return freq


# ───────────────────────── writer ───────────────────────────────────
def split_and_save(
    freq: dict[str, dict[str, int]],
    over_path: str = "ner_unit_freq_ge100.json",
    under_path: str = "ner_unit_freq_lt100.json",
) -> None:
    over = {}
    under = {}

    for ner, sub in freq.items():
        over[ner] = {u: c for u, c in sub.items() if c >= THRESHOLD}
        under[ner] = {u: c for u, c in sub.items() if c < THRESHOLD}

    with open(over_path, "w", encoding="utf-8") as f_over:
        json.dump(over, f_over, indent=2)

    with open(under_path, "w", encoding="utf-8") as f_under:
        json.dump(under, f_under, indent=2)

    print(f"≥{THRESHOLD}: {over_path}")
    print(f"< {THRESHOLD}: {under_path}")


# ───────────────────────── main entry ───────────────────────────────
def main(
    ing_path: str = "../data/ingredients.txt",
    rec_path: str = "../data/train_data.json",
) -> None:
    ner_terms = load_ner_terms(ing_path)
    freq_dict = build_freq(ner_terms, rec_path)
    split_and_save(freq_dict)


if __name__ == "__main__":
    main()

