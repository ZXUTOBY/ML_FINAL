import json
import numpy as np
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re

# load data
embeddings = np.load("/Users/chrisdeng/Desktop/COSC410/final/ML_FINAL/Autoencoder/recipe_embeddings.npy")
with open("/Users/chrisdeng/Desktop/COSC410/final/ML_FINAL/500rows.json") as f:
    recipes = json.load(f)
with open("/Users/chrisdeng/Desktop/COSC410/final/ML_FINAL/Autoencoder/ingredient_to_index.json") as f:
    ingredient_to_index = json.load(f)
with open("/Users/chrisdeng/Desktop/COSC410/final/ML_FINAL/Autoencoder/ingredient_base_quantity.json") as f:
    ingredient_base_quantity = json.load(f)

input_dim = len(ingredient_to_index)

# redefine model class
class RecipeAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, bottleneck_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim), nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_bottleneck(self, x):
        return self.encoder(x)

# load trained model
model = RecipeAutoencoder(input_dim=input_dim)
model.load_state_dict(torch.load("/Users/chrisdeng/Desktop/COSC410/final/ML_FINAL/Autoencoder/trained_autoencoder.pth"))
model.eval()

# encode ingredient
def encode_ingredients(ingredients):
    vec = [0.0] * input_dim
    for ing in ingredients:
        ing = ing.lower().strip()
        if ing in ingredient_to_index:
            vec[ingredient_to_index[ing]] = ingredient_base_quantity.get(ing, 0.5)
    return torch.tensor([vec], dtype=torch.float32)

# recommend recipes
def recommend_by_ingredients(user_ingredients, top_k=5):
    input_tensor = encode_ingredients(user_ingredients)
    with torch.no_grad():
        query_vec = model.get_bottleneck(input_tensor).numpy()
    sims = cosine_similarity(normalize(query_vec), normalize(embeddings))[0]

    results = []
    for i, rec in enumerate(recipes):
        overlap = sum(ing.lower() in rec["NER"] for ing in user_ingredients)
        if overlap >= 1:
            score = sims[i] + 0.1 * overlap  # Adjust weight of overlap
            results.append((i, score, rec))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    print(" Enter your ingredients (comma or space separated):")
    user_input = input("> ")
    user_ingredients = [x.strip().lower() for x in re.split(r"[\s,]+", user_input) if x.strip()]

    results = recommend_by_ingredients(user_ingredients)

    print(f"\n You entered: {user_ingredients}")
    print("\n Top Recommended Recipes:")
    for i, score, rec in results:
        print(f"\n {rec['title']} (score: {score:.4f})")
        print("Ingredients:")
        for ing in rec['ingredients']:
            print(f"  - {ing}")
        print("Instructions:")
        for step in rec['instructions']:
            print(f"  • {step}")
