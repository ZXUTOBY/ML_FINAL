import json
import re

def split_sentences(text: str) -> list[str]:
    """
    Splits a block of text into individual sentences.
    Keeps the punctuation at the end of each sentence.
    """
    # This regex splits on period, exclamation, or question mark followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

# File paths
input_path = "../data/train_data.json"
output_path = "../data/train_data_cleaned.json"

# Load the original recipe file
with open(input_path, "r", encoding="utf-8") as f:
    recipes = json.load(f)

# Process each recipe
for recipe in recipes:
    new_instructions = []
    for step in recipe.get("instructions", []):
        if isinstance(step, str):
            sentences = split_sentences(step)
            new_instructions.extend(sentences)
    recipe["instructions"] = new_instructions

# Save the cleaned recipe file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(recipes, f, indent=2, ensure_ascii=False)

print(f"Processed {len(recipes)} recipes. Cleaned file saved as '{output_path}'.")


