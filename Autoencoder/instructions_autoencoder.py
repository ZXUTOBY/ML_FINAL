import json
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# load data
with open("data/train_data.json") as f:
    data = json.load(f)

# build verb vocabulary from dict-style instruction_NER
all_verbs = sorted({
    list(d.keys())[0]
    for recipe in data
    for d in recipe.get("instruction_NER", [])
    if isinstance(d, dict) and len(d) == 1
})
verb_to_index = {verb: i for i, verb in enumerate(all_verbs)}
input_dim = len(verb_to_index)

# save verb_to_index mapping
with open("verb_to_index.json", "w") as f:
    json.dump(verb_to_index, f)


# vectorize instruction_NER
vectors = []
for recipe in data:
    vec = [0.0] * input_dim
    for entry in recipe.get("instruction_NER", []):
        if isinstance(entry, dict):
            for verb, weight in entry.items():
                if verb in verb_to_index:
                    vec[verb_to_index[verb]] = weight  # typically 0.5
    vectors.append(vec)

X_tensor = torch.tensor(vectors, dtype=torch.float32)
dataset = TensorDataset(X_tensor, X_tensor)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# define instruction autoencoder 
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

# train the model
model = InstructionAutoencoder(input_dim=input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training instruction autoencoder...")
for epoch in range(50):
    total_loss = 0
    for inputs, _ in loader:
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss:.6f}")

# evaluate 
model.eval()
with torch.no_grad():
    X_pred = model(X_tensor).numpy()
    X_true = X_tensor.numpy()
    bottleneck_vecs = model.get_bottleneck(X_tensor).numpy()

mse = mean_squared_error(X_true, X_pred)
mae = mean_absolute_error(X_true, X_pred)
cos_sim = np.mean([
    np.dot(X_true[i], X_pred[i]) / (np.linalg.norm(X_true[i]) * np.linalg.norm(X_pred[i]) + 1e-8)
    for i in range(len(X_true))
])

print("\n Evaluation Metrics:")
print(f"MSE : {mse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"Avg. Cosine Similarity: {cos_sim:.6f}")

np.save("instruction_embeddings.npy", bottleneck_vecs)
torch.save(model.state_dict(), "trained_instruction_autoencoder.pth")
print("Saved instruction_embeddings.npy and trained_instruction_autoencoder.pth")
