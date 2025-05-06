import json
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import time
from torch.utils.data import Subset
import random
from torch.cuda.amp import GradScaler, autocast





# === Set up GPU device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Upload or load data ===
from google.colab import drive
drive.mount('/content/drive')

# === Load cleaned ingredients list ===
with open("/content/drive/MyDrive/ML_FINAL/ingredients.txt") as f1:
    all_ingredients = sorted(line.strip().lower() for line in f1 if line.strip())

ingredient_to_index = {ingredient: i for i, ingredient in enumerate(all_ingredients)}
input_dim = len(all_ingredients)

# === Assign random base quantities ===
ingredient_base_quantity = {
    ing: round(random.uniform(0.1, 1.0), 2)
    for ing in all_ingredients
}

# === Save vocab and base quantities ===
with open("ingredient_to_index.json", "w") as f:
    json.dump(ingredient_to_index, f)
with open("ingredient_base_quantity.json", "w") as f:
    json.dump(ingredient_base_quantity, f)



# === Load dataset ===
with open("/content/drive/MyDrive/ML_FINAL/train_data.json") as f:
    data = json.load(f)


# === Lazy vectorization dataset ===
class LazyRecipeDataset(torch.utils.data.Dataset):
    def __init__(self, data, ingredient_to_index, ingredient_base_quantity, input_dim):
        self.data = data
        self.ingredient_to_index = ingredient_to_index
        self.ingredient_base_quantity = ingredient_base_quantity
        self.input_dim = input_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        recipe = self.data[idx]
        vector = [0.0] * self.input_dim
        for ing_dict in recipe["NER"]:
            for ing, score in ing_dict.items():
                ing = ing.lower().strip()
                if ing in self.ingredient_to_index:
                    vector[self.ingredient_to_index[ing]] = self.ingredient_base_quantity[ing] * score
        vector = torch.tensor(vector, dtype=torch.float32)
        return vector, vector

# create dataset and loader
dataset = LazyRecipeDataset(data, ingredient_to_index, ingredient_base_quantity, input_dim)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# === Autoencoder Definition ===
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
    


scaler = GradScaler()





print("🚀 Start training...")

for bottleneck_dim in [32, 64, 128]:
    model = RecipeAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(49):
        start_time = time.time()
        total_loss = 0
        recipe_count = 0
        next_checkpoint = 100000

        model.train()
        for inputs, _ in loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, inputs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            recipe_count += inputs.size(0)
            if recipe_count >= next_checkpoint:
                print(f"   ✅ Trained {recipe_count} recipes in epoch {epoch}")
                next_checkpoint += 100000

        elapsed = time.time() - start_time
        

        if epoch == 48:
            # === Final Evaluation (MAE and MSE only) ===
            model.eval()
            mse_total = 0
            mae_total = 0
            count = 0

            with torch.no_grad():
                for inputs, _ in DataLoader(dataset, batch_size=32):
                    inputs = inputs.to(device)
                    with autocast():
                        outputs = model(inputs)

                    X_pred = outputs.cpu().numpy()
                    X_true = inputs.cpu().numpy()

                    mse_total += np.sum((X_pred - X_true) ** 2)
                    mae_total += np.sum(np.abs(X_pred - X_true))
                    count += len(X_true)

            mse = mse_total / (count * input_dim)
            mae = mae_total / (count * input_dim)

            print(f"Epoch {epoch:3d} | Time: {elapsed:.2f}s | Final MSE: {mse:.6f} | Final MAE: {mae:.6f}")
        else:
            print(f"Epoch {epoch:3d} | Time: {elapsed:.2f}s")



    print("📦 Done. Saving model and embeddings...")

    if bottleneck_dim == 64:
        save_path = "/content/drive/MyDrive/ML_FINAL/ingredients_embeddings.npy"
        bottleneck_dim_actual = model.get_bottleneck(torch.zeros(1, input_dim).to(device)).shape[1]
        memmap_out = np.memmap(save_path, dtype='float32', mode='w+', shape=(len(dataset), bottleneck_dim_actual))

        with torch.no_grad():
            i = 0
            save_loader = DataLoader(dataset, batch_size=128)
            for inputs, _ in tqdm(save_loader, total=len(dataset) // 512):
                inputs = inputs.to(device)
                vecs = model.get_bottleneck(inputs).cpu().numpy()
                memmap_out[i:i+len(vecs)] = vecs
                i += len(vecs)

        del memmap_out
        print(f"✅ Bottleneck vectors saved to: {save_path}")

        model_path = "/content/drive/MyDrive/ML_FINAL/trained_autoencoder.pth"
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to: {model_path}")
   