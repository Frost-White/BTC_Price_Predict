import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import joblib
from model import LSTMForecast

# Ayarlar
CSV_PATH = "data/train.csv"
MODEL_PATH = "guzel/256Hidden_5Layer_lstm_model_epoch90.pth"
SCALER_PATH = "outputs/scaler.save"

FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
TARGET_FEATURE = "Close"
TARGET_INDEX = FEATURES.index(TARGET_FEATURE)
WINDOW_SIZE = 30
PRED_SIZE = 10

# Veriyi oku
df = pd.read_csv(CSV_PATH)
df = df.sort_values("Date")
feature_df = df[FEATURES].astype(float)
close_prices = feature_df[TARGET_FEATURE].values

# Yüzde değişim ve ölçekleme
pctchange_df = feature_df.pct_change().dropna()
pctchange_np = pctchange_df.values
scaler = joblib.load(SCALER_PATH)
pctchange_scaled = scaler.transform(pctchange_np)

# Son 30 gün
input_seq = pctchange_scaled[-WINDOW_SIZE:]
X_input = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

last_real_price = close_prices[-1]

# Model yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMForecast(input_size=len(FEATURES), output_size=PRED_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Tahmin
with torch.no_grad():
    preds = model(X_input.to(device))
    tahmin_pct = preds[0].cpu().numpy().flatten()

tahmin_pct_inv = scaler.inverse_transform(
    np.stack([np.zeros(PRED_SIZE) if i != TARGET_INDEX else tahmin_pct for i in range(len(FEATURES))], axis=1)
)[:, TARGET_INDEX]

# Fiyat tahmini
predicted_prices = [last_real_price]
for pct in tahmin_pct_inv:
    predicted_prices.append(predicted_prices[-1] * (1 + pct))
predicted_prices = predicted_prices[1:]

# Grafik
plt.figure(figsize=(10,5))
plt.plot(range(PRED_SIZE), predicted_prices, marker='o', label="Tahmin (gelecek 10 gün)")
plt.title("Son 30 Günden Sonraki 10 Günlük Tahmin")
plt.xlabel("Tahmin Günü")
plt.ylabel("Fiyat")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
