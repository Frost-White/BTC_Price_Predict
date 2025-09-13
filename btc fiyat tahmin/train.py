import pandas as pd
from torch.utils.data import DataLoader
from dataset import create_sequences_multifeature_pctchange, TimeseriesDataset
from model import LSTMForecast, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import StandardScaler  # Burası değişti!
import joblib

# --- PARAMETRE BLOKU ---
CSV_PATH = "data/train.csv"
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
TARGET_FEATURE = "Close"
TARGET_INDEX = FEATURES.index(TARGET_FEATURE)
WINDOW_SIZE = 30
PRED_SIZE = 10
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
MODEL_DIR = "outputs"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")
CHECKPOINT_FREQ = 10
TEST_RATIO = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.sort_values("Date")
feature_df = df[FEATURES].astype(float)

# Her feature için yüzde değişim (ilk satır NaN, atılacak)
pctchange_df = feature_df.pct_change().dropna()
pctchange_np = pctchange_df.values  # shape: (örnek_sayısı, feature_sayısı)

# Normalizasyon (tüm feature'lara birlikte, artık StandardScaler)
split_idx_raw = int((1 - TEST_RATIO) * len(pctchange_np))
scaler = StandardScaler()
scaler.fit(pctchange_np[:split_idx_raw])  # Sadece train split ile fit
joblib.dump(scaler, SCALER_PATH)
pctchange_scaled = scaler.transform(pctchange_np)

X, y = create_sequences_multifeature_pctchange(pctchange_scaled, WINDOW_SIZE, PRED_SIZE, TARGET_INDEX)
split_idx = int((1 - TEST_RATIO) * len(X))
train_ds = TimeseriesDataset(X[:split_idx], y[:split_idx])
test_ds = TimeseriesDataset(X[split_idx:], y[split_idx:])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMForecast().to(device)  # Tüm model parametreleri model.py'de

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

    if (epoch + 1) % CHECKPOINT_FREQ == 0 or (epoch + 1) == EPOCHS:
        checkpoint_path = os.path.join(
            MODEL_DIR,
            f"{DEFAULT_HIDDEN_SIZE}Hidden_{DEFAULT_NUM_LAYERS}Layer_lstm_model_epoch{epoch+1}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint kaydedildi: {checkpoint_path}")

print("Eğitim tamamlandı.")
