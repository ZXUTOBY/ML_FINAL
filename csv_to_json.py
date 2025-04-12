import pandas as pd
import ast
import json

# Load the CSV file
df = pd.read_csv('data/first_20_rows.csv')

# Convert each row to the desired JSON structure
records = []
for _, row in df.iterrows():
    try:
        ingredients = ast.literal_eval(row['ingredients'])
        instructions = ast.literal_eval(row['directions'])
        ner = ast.literal_eval(row['NER'])
    except (ValueError, SyntaxError):
        # If parsing fails, skip or handle appropriately
        continue

    record = {
        "title": row['title'],
        "ingredients": ingredients,
        "instructions": instructions,
        "NER": ner
    }
    records.append(record)

# Save to JSON file
output_path = 'data/20rows.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"Converted {len(records)} recipes to JSON and saved to {output_path}")
