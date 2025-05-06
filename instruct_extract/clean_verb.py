from __future__ import annotations
import argparse
import os
import sys
from typing import Iterable, List, Set

import openai

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL       = "gpt-4.1"   # or "gpt-4o-mini" if you have access
BATCH_SIZE  = 10                     # number of verbs per API call
TEMPERATURE = 0.0                    # deterministic filtering

# Create a single client (reads OPENAI_API_KEY from env)
client = openai.OpenAI()

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def chunk_iterable(iterable: List[str], n: int = BATCH_SIZE) -> Iterable[List[str]]:
    """Yield successive n-item chunks from *iterable*."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def ask_gpt_for_food_verbs(verbs_block: str) -> str:
    """
    Given ≤10 verbs (newline-separated), return only those that are *possible*
    to use in cooking or food processing contexts.  One verb per line; same
    spelling/case; alphabetical order.
    """
    system_msg = (
        "You are an expert culinary lexicographer.\n"
        "Task: From a newline-separated list of English verbs, DELETE only those "
        "that are **impossible** to apply when cooking, preparing, processing, "
        "serving, or preserving food. KEEP any verb that *could even conceivably* "
        "describe an action on food, no matter how rare or specialised.\n\n"
        "However, you must DELETE any verbs that are clearly just ingredient names used as verbs "
        "(e.g. 'salt', 'sugar', 'pepper'), even though they *can* be used as culinary verbs, "
        "because they are more appropriately considered ingredients, not actions.\n\n"
        "Examples of verbs to DELETE (impossible with food): legislate, orbit, teleport, litigate.\n"
        "Examples to DELETE (ingredients as verbs): salt, sugar, pepper, cocoa, oil.\n"
        "Examples to KEEP (possible culinary actions): bake, boil, broil, acidify, "
        "laminate, scorch, tattoo (e.g. 'tattoo grill marks onto a steak').\n\n"
        "Output rules:\n"
        "• Preserve spelling & capitalisation exactly.\n"
        "• One verb per line, sorted alphabetically.\n"
        "• No explanations, headers, or bullets."
    )

    # One quick demonstration so the model “locks on” to the rule
    demo_user      = "orbit\nboil\nlegislate\nscorch\njump"
    demo_assistant = "boil\nscorch"

    user_msg = f"Here are the verbs:\n{verbs_block.strip()}"

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=128,
        messages=[
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": demo_user},
            {"role": "assistant", "content": demo_assistant},
            {"role": "user",      "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


def save_text(block: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(block + "\n")


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Filter cooking-related verbs via GPT.")
    parser.add_argument("verb_file", help="Path to verb_only.txt (one verb per line)")
    parser.add_argument("-o", "--output", default="food_verbs.txt",
                        help="Output file (default: food_verbs.txt)")
    args = parser.parse_args()

    # Make sure the API key exists
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # Read the verb list (strip blank lines)
    with open(args.verb_file, encoding="utf-8") as f:
        verb_lines: List[str] = [ln.rstrip("\n") for ln in f if ln.strip()]

    # Batch through GPT and collect unique keeps
    kept: Set[str] = set()
    for chunk in chunk_iterable(verb_lines):
        chunk_block = "\n".join(chunk)
        gpt_keep    = ask_gpt_for_food_verbs(chunk_block)
        kept.update(v for v in gpt_keep.splitlines() if v)

    # Write results
    output_block = "\n".join(sorted(kept))
    save_text(output_block, args.output)
    print(f"✓ Finished.  {len(kept)} cooking-related verbs written to {args.output}")


if __name__ == "__main__":
    main()