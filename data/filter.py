import json

def filter_recipes_by_name(input_file='train_data.json', output_file='filtered_recipes.json'):
    target_titles = {"Spaghetti Bolognese", "Chicken Tikka Masala"}

    # Load all recipes from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter recipes by title
    filtered = [recipe for recipe in data if recipe.get('title') in target_titles]

    # Save filtered recipes to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"{len(filtered)} recipe(s) written to {output_file}.")

# Example usage
if __name__ == "__main__":
    filter_recipes_by_name()






