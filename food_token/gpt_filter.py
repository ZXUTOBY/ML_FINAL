#!/usr/bin/env python
"""
gpt_filter.py
──────────────────────────────────────────────────────────
Example
-------
python gpt_filter.py --in 1_all.txt --out 1_filtered.txt \
                     --start 0 --end 10000 \
                     --batch 50 --model gpt-3.5-turbo
"""

import argparse
import os
import time
from pathlib import Path
from openai import OpenAI, OpenAIError

# ─────────────────────────────────────────────────────────
# Configuration (Single word)
# ─────────────────────────────────────────────────────────

SYSTEM_MSG_Single = (
"""
You are an INGREDIENT-LIST VALIDATOR.
Your task is to output exactly N digits (0 or 1) based on whether each item is a valid English-language standalone ingredient.

╔═ DECISION RULE ═══════════════════════════════════════════════════╗
Mark:

1 = Common, recognized foodstuff, seasoning, spice, condiment, or well-established compound ingredient.

0 = Anything else: brands, places, partial names, actions, vague words, non-edibles, uncommon varietals, or if any doubt exists.

❗ Always default to 0 if not 100% certain. ❗ Reject standalone brands, places, incomplete morphemes, non-edible materials.

╔═ OUTPUT FORMAT ═══════════════════════════════════════════════════╗

Return exactly N digits (0 or 1), separated by commas.

No extra spaces or text.

Example: 1,0,1

╔═ EXEMPLARS (N = 20) ══════════════════════════════════════════════╗


Item	Result	Reason
velveeta	0	brand
gochu	0	incomplete morpheme
bouillon	1	recognized food ingredient
aioli	1	condiment
tang	0	brand
gum	0	container/additive word
tamari	1	compound soy sauce
xanthan	0	incomplete (needs “gum”)
carnaroli	0	obscure varietal
matcha	1	powder
bonito	1	edible flakes
mirin	1	cooking wine
agar	1	setting agent
chai	1	spice tea
lox	1	cured salmon
tajin	0	brand
garam	0	incomplete (needs “masala”)
panko	1	bread crumbs
malt	1	edible extract
miso	1	fermented paste
Answer:
0,0,1,1,0,0,1,0,0,1,1,1,1,1,1,0,0,1,1,1

╔═ REMINDERS ═══════════════════════════════════════════════════════╗

Places, brands, partial names = 0

Incomplete or rare varietals = 0

If any doubt = 0

BEGIN!
"""
)

DEFAULT_BATCH = 20          # lines per API call
SLEEP_BETWEEN_REQ = 0.5     # seconds – keeps us well below rate limits



# ─────────────────────────────────────────────────────────
# Configuration (Double word)
# ─────────────────────────────────────────────────────────

SYSTEM_MSG_Double = (
"""
You are an INGREDIENT-LIST VALIDATOR.

TASK
----
You will receive exactly **N items** (each item contains exactly **two words**, one per line).

Your job:  
→ For each item, determine whether it forms a valid, standalone ingredient name according to strict culinary identity rules.

→ Output **exactly N digits** (0 or 1), separated by commas, all on a single line — no extra text, no spaces, no newlines.

CORE DECISION LOGIC
--------------------
✅ Mark **1** if and only if:
- The two words **together** form a recognized, standalone ingredient name commonly used directly in English-language recipes.
- Both words are **essential**: removing either word would **change the meaning** and cause the ingredient name to be **wrong, incomplete, or misleading**.
- The phrase names a **raw, natural, or minimally processed ingredient**, not a dish, not a product, not a form, and not a style.

❌ Mark **0** in all other cases, including when:
- **One word alone** fully captures the correct ingredient without any essential flavor, function, or identity loss.
- The phrase describes a **cooking preparation**, **form**, **texture**, or **cut** (e.g., "ground turkey", "shredded coconut").
- The phrase describes a **style**, **flavor intensity**, **regional variety**, or **brand-specific version** (e.g., "smoked paprika", "garam masala", "manzanilla olives").
- The phrase names a **dish**, **recipe**, or **combined seasoning product** (e.g., "buffalo wings", "taco seasoning", "dry rub").
- The phrase simply combines two distinct foods (e.g., "tomato garlic", "bean filling") without forming one single true identity.

FLAVOR & IDENTITY TEST
-----------------------
For each item, apply the following reasoning steps:

1. **Word Removal Check:**  
   - If either word can be removed without losing the true identity of the ingredient, mark **0**.
   - Example: "chocolate wafers" → "chocolate" alone still captures the main ingredient → **0**.

2. **Form or Preparation Check:**  
   - If one word only describes the preparation, cut, or form of the ingredient (e.g., "ground", "shredded", "sliced"), mark **0**.

3. **Style or Variation Check:**  
   - If the two-word phrase describes a style, regional type, or varietal that doesn't fundamentally alter the core food, mark **0**.

4. **Dish/Product Check:**  
   - If the phrase describes a complete dish, recipe, or seasoning blend and not a raw ingredient, mark **0**.

5. **Standalone Identity Check:**  
   - Only if both words **must** stay together to uniquely identify a real ingredient (one that would be wrong or incomplete if simplified), mark **1**.

If you are uncertain at any point, always default to **0**.

CRITICAL EXAMPLES (N = 18)
---------------------------
Items:
 1. brown sugar           # ✓ distinct compound (brown sugar ≠ sugar)
 2. paper towel           # ✗ container
 3. olive oil             # ✓ distinct edible oil (olive oil ≠ olive)
 4. overripe banana       # ✗ adjective + fruit (banana alone suffices)
 5. buffalo wings         # ✗ dish, not ingredient
 6. dry rub               # ✗ seasoning method
 7. cumin ground          # ✗ wrong order, preparation style
 8. chicken breast        # ✓ distinct cut of chicken
 9. tomato garlic         # ✗ two separate foods
10. xanthan gum           # ✓ accepted food additive
11. smoky flavor          # ✗ flavor descriptor
12. bread flour           # ✓ distinct flour type
13. szechuan pepper       # ✓ unique pepper species (not generic pepper)
14. manzanilla olives     # ✗ olives alone sufficient
15. pickled jalapeno      # ✓ pickling changes jalapeño flavor drastically
16. cherry tomatoes      # ✓ distinct tomato variety (different flavor/texture)
17. garam masala          # ✗ masala alone identifies the ingredient
18. chocolate wafers      # ✗ chocolate alone identifies the food

Answer:
1,0,1,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0

OUTPUT FORMAT
--------------
• A single line containing exactly N digits (0 or 1), separated by commas.
• No extra spaces, no extra text, no blank lines.
• Example for N = 3: `1,0,1`

MANDATORY FINAL SELF-CHECK
---------------------------
Before submitting your final output:
1. Count the number of commas.
2. Add 1 to the number of commas.
3. Confirm that this number equals exactly **N** (the number of input items).
4. If it does not match, correct your output immediately.

REMINDERS
---------
✅ Both words must be essential to define a true standalone ingredient.  
✅ If either word alone suffices to represent the ingredient, reject.  
✅ Reject anything describing style, texture, preparation, form, brand, dish, or flavor strength.  
✅ Always default to rejection (0) if unsure.  
✅ No missing or extra digits allowed.

BEGIN!
"""
)


DEFAULT_BATCH = 20          # lines per API call
SLEEP_BETWEEN_REQ = 0.5     # seconds – keeps us well below rate limits




# ─────────────────────────────────────────────────────────
# GPT helper
# ─────────────────────────────────────────────────────────
def gpt_classify(batch, model="gpt-3.5-turbo", temperature: float = 0.0):
    """
    Ask GPT to flag each line in *batch* as ingredient (1) or not (0).

    Returns
    -------
    list[int]
        Same length as *batch*, containing 0 or 1 for each item.
    """
    client = OpenAI()  # reads OPENAI_API_KEY (and optionally OPENAI_ORG_ID)

    prompt = "Items:\n" + "\n".join(
        f"{i + 1}. {line.strip()}" for i, line in enumerate(batch)
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG_Double},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    raw = response.choices[0].message.content.strip()
    flags = [int(tok) for tok in raw.replace(" ", "").split(",")]

    if len(flags) != len(batch):
        raise ValueError(
            f"Expected {len(batch)} flags, got {len(flags)} – raw output: {raw}"
        )
    return flags


# ─────────────────────────────────────────────────────────
# Main filter routine
# ─────────────────────────────────────────────────────────
def filter_file(
    in_path: Path,
    out_path: Path,
    start: int,
    end: int | None,
    batch_size: int,
    model: str,
):
    """Stream lines[start:end] through GPT and write ingredients to out_path."""
    lines = in_path.read_text(encoding="utf-8").splitlines(keepends=True)
    end = end if end is not None else len(lines)  # default → EOF
    subset = lines[start:end]

    kept_lines: list[str] = []

    for i in range(0, len(subset), batch_size):
        batch = subset[i : i + batch_size]
        try:
            flags = gpt_classify(batch, model=model)
        except OpenAIError as e:
            print(f"[{start+i}-{start+i+len(batch)-1}] OpenAI error: {e}")
            continue
        except Exception as e:
            print(f"[{start+i}-{start+i+len(batch)-1}] {e}")
            continue

        kept_lines.extend(
            line for line, keep_flag in zip(batch, flags) if keep_flag == 1
        )
        time.sleep(SLEEP_BETWEEN_REQ)

    out_path.write_text("".join(kept_lines), encoding="utf-8")
    print(
        f"✓ Wrote {len(kept_lines)} of {len(subset)} lines "
        f"({in_path.name}[{start}:{end}]) → {out_path}"
    )


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Filter ingredients with GPT")

    # renamed flags (avoid the keyword 'in')
    parser.add_argument("--input",  dest="input_file",
                        default="1_all.txt",      help="input file name")
    parser.add_argument("--output", dest="output_file",
                        default="1_filtered.txt", help="output file name")

    parser.add_argument("--start", type=int, default=0,
                        help="first line (0-based, inclusive)")
    parser.add_argument("--end",   type=int,
                        help="last line (0-based, exclusive). Omit or -1 for EOF")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help="lines per OpenAI request (≤100 recommended)")
    parser.add_argument("--model", default="gpt-3.5-turbo",
                        help="OpenAI model id")

    args = parser.parse_args()

    # use the new attribute names here
    filter_file(
        Path(args.input_file),
        Path(args.output_file),
        start=args.start,
        end=None if args.end in (None, -1) else args.end,
        batch_size=args.batch,
        model=args.model,
    )



if __name__ == "__main__":
    # Ensure the key is present before running
    if os.getenv("OPENAI_API_KEY") is None:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. "
            "Set it in your environment before running this script."
        )
    main()
