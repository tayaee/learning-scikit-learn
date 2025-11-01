import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

MODEL_PATH = "my_binary_classification_model.pth"
STATE_DICT_PATH = "my_model_state_dict.pth"
print(f"Model weights save path: {STATE_DICT_PATH}")
print("-" * 40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. Data Preparation and Model Definition ---

X_train_np = np.random.rand(1000, 10).astype("float32")
y_train_np = np.random.randint(0, 2, size=(1000, 1)).astype("float32")

X_train = torch.from_numpy(X_train_np).to(device)
y_train = torch.from_numpy(y_train_np).to(device)

X_new_np = np.random.rand(5, 10).astype("float32")
X_new = torch.from_numpy(X_new_np).to(device)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# --- 2. PyTorch Sequential Model Definition ---
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.sequential_model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.sequential_model(x)


model = SimpleNN().to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# --- 3. Model Training (Train) ---
print("Starting model training...")
model.train()
for epoch in range(5):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

print("Model training complete.")
print("-" * 40)


# --- 4. Model Saving (Saving State Dict) ---
torch.save(model.state_dict(), STATE_DICT_PATH)
print(f"Model weights (State Dict) successfully saved: {STATE_DICT_PATH}")
print("-" * 40)


# --- 5. Model Loading (Loading the Model) ---
loaded_model = SimpleNN().to(device)
loaded_model.load_state_dict(torch.load(STATE_DICT_PATH, map_location=device))
loaded_model.eval()

print("Model loading complete.")
print("-" * 40)


# --- 6. Prediction with Loaded Model ---

print("Performing prediction with the loaded model...")

with torch.no_grad():
    predictions_prob_tensor = loaded_model(X_new)

predictions_prob = predictions_prob_tensor.cpu().numpy().flatten()
predicted_classes = (predictions_prob > 0.5).astype(int)

print("\n[Predicted probabilities for new data (Loaded Model)]")
print(predictions_prob)

print("\n[Predicted classes]")
print(predicted_classes)
print("-" * 40)


# (Optional) Cleanup saved file
os.remove(STATE_DICT_PATH)
print(f"Saved file {STATE_DICT_PATH} deleted.")
