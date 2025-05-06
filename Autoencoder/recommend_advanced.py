import json
import numpy as np
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re
from collections import defaultdict
import random
import math


# Load data
with open("data/train_data.json") as f:
    recipes = json.load(f)
with open("Autoencoder/ingredient_to_index.json") as f:
    ingredient_to_index = json.load(f)
with open("Autoencoder/ingredient_base_quantity.json") as f:
    ingredient_base_quantity = json.load(f)
with open("Autoencoder/verb_to_index.json") as f:
    verb_to_index = json.load(f)

# Build token-to-ingredient variants map
token_to_variants = defaultdict(set)

for full_ing in ingredient_to_index:
    tokens = full_ing.lower().strip().split()
    for token in tokens:
        token_to_variants[token].add(full_ing)

# Load .npy saved via np.memmap
NUM_RECIPES = len(recipes)
BOTTLENECK_DIM = 64  # should match your autoencoder setting

ing_embeddings = np.memmap(
    "Autoencoder/ingredients_embeddings.npy",
    dtype='float32',
    mode='r',
    shape=(NUM_RECIPES, BOTTLENECK_DIM)
)

assert ing_embeddings.shape == (len(recipes), BOTTLENECK_DIM), \
    f"Expected shape {(len(recipes), BOTTLENECK_DIM)}, got {ing_embeddings.shape}"

instr_embeddings = np.load("Autoencoder/instruction_embeddings.npy")

input_dim = len(ingredient_to_index)
verb_dim = len(verb_to_index)

# Define model classes
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

class InstructionAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, bottleneck_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_bottleneck(self, x):
        return self.encoder(x)

# Load trained models
ing_model = RecipeAutoencoder(input_dim=input_dim)
ing_model.load_state_dict(torch.load("Autoencoder/trained_autoencoder.pth", map_location=torch.device('cpu')))
ing_model.eval()

instr_model = InstructionAutoencoder(input_dim=verb_dim)
instr_model.load_state_dict(torch.load("Autoencoder/trained_instruction_autoencoder.pth", map_location=torch.device('cpu')))
instr_model.eval()

def encode_ingredients(ingredients):
    vec = [0.0] * input_dim
    used_indices = set()

    for ing in ingredients:
        ing_norm = ing.lower().strip()

        if " " in ing_norm:
            if ing_norm in ingredient_to_index:
                idx = ingredient_to_index[ing_norm]
                vec[idx] = ingredient_base_quantity.get(ing_norm, 0.5)
                used_indices.add(idx)
            base = ing_norm.split()[-1]
            if base in ingredient_to_index and ingredient_to_index[base] not in used_indices:
                idx = ingredient_to_index[base]
                vec[idx] = ingredient_base_quantity.get(base, 0.5)
                used_indices.add(idx)

        else:
            # Single‑word input: only activate itself
            if ing_norm in ingredient_to_index:
                idx = ingredient_to_index[ing_norm]
                vec[idx] = ingredient_base_quantity.get(ing_norm, 0.5)
                used_indices.add(idx)

    return torch.tensor([vec], dtype=torch.float32)

# Instruction NER encoder
def encode_instruction_NER(verbs):
    vec = [0.0] * verb_dim
    for entry in verbs:
        if isinstance(entry, dict) and len(entry) == 1:
            verb, weight = list(entry.items())[0]
            if verb in verb_to_index:
                vec[verb_to_index[verb]] = weight
    return torch.tensor([vec], dtype=torch.float32)

# Recommendation function
def recommend(
    user_ingredients,
    ref_recipes,
    top_k=1,
    rerank_k=100,
    alpha=0.9,
    random_from_top=10  # pick one recipe at random from the top N
):
    # ingredient query embedding
    input_tensor = encode_ingredients(user_ingredients)
    with torch.no_grad():
        ing_query_vec = ing_model.get_bottleneck(input_tensor).numpy()
    ing_sims = cosine_similarity(
        normalize(ing_query_vec),
        normalize(ing_embeddings)
    )[0]

    # build per‑input acceptable variants
    required_variants_list = []
    for ing in user_ingredients:
        ing_norm = ing.lower().strip()
        if " " in ing_norm:
            required_variants_list.append({ing_norm})
        else:
            required_variants_list.append(token_to_variants.get(ing_norm, {ing_norm}))

    # filter by partial match: 2/3 if > 4 ingredients, else all
    required_matches = (
        math.ceil(2 / 3 * len(required_variants_list))
        if len(required_variants_list) > 4 else len(required_variants_list)
    )

    sorted_indices = ing_sims.argsort()[::-1]
    filtered_top_indices = []
    for i in sorted_indices:
        recipe_ing_set = {
            r.lower().strip()
            for r in recipes[i].get("ingredients", [])
        }

        match_count = sum(
            any(
                any(var in recipe_ing for recipe_ing in recipe_ing_set)
                for var in variants
            )
            for variants in required_variants_list
        )

        if match_count >= required_matches:
            filtered_top_indices.append(i)

        if len(filtered_top_indices) == rerank_k:
            break


    # encode preferred recipes by instruction NER
    ref_vecs = []
    for ref in ref_recipes:
        encoded = encode_instruction_NER(ref.get("instruction_NER", []))
        with torch.no_grad():
            ref_vecs.append(instr_model.get_bottleneck(encoded).numpy()[0])
    if not ref_vecs:
        print("⚠️ No valid instruction_NER vectors found.")
        return []

    avg_ref_vec = np.mean(ref_vecs, axis=0, keepdims=True)

    # rerank by combined similarity
    results = []
    for i in filtered_top_indices:
        instr_sim = cosine_similarity(
            avg_ref_vec,
            instr_embeddings[i].reshape(1, -1)
        )[0][0]
        score = alpha * ing_sims[i] + (1 - alpha) * instr_sim
        results.append((i, score, recipes[i]))
    results.sort(key=lambda x: x[1], reverse=True)

    if random_from_top is not None:
        if not results:
            print("⚠️ No recipes found matching your ingredients.")
            return []
        # only sample from however many we actually have
        n = min(random_from_top, len(results))
        chosen = random.choice(results[:n])
        return [chosen]

    return results[:top_k]

# CLI main
if __name__ == "__main__":
    print("Enter your ingredients, separated by commas (e.g. ground beef, tomato):")
    user_input = input("Ingredients > ")
    # only split on commas so multi‑word ingredients stay intact
    user_ingredients = [
        x.strip().lower()
        for x in user_input.split(',')
        if x.strip()
    ]

    print("\nLoading preferred recipes from preferred_recipes.json...")
    with open("Autoencoder/preferred_recipes.json") as f:
        preferred_recipes = json.load(f)

    # e.g. to pick randomly from top 10, pass random_from_top=10
    results = recommend(
        user_ingredients,
        preferred_recipes,
        alpha=0.8,
        top_k=1
    )

    print(f"\nYou entered: {user_ingredients}")
    print("\nTop Recommendation:")
    for i, score, rec in results:
        print(f"\n{rec['title']} (score: {score:.4f})")
        print("Ingredients:")
        for ing in rec['ingredients']:
            print(f"  - {ing}")
        print("Instructions:")
        for step in rec['instructions']:
            print(f"  • {step}")

