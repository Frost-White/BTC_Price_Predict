import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import joblib
from dataset import create_sequences_multifeature_pctchange
from model import LSTMForecast

CSV_PATH = "data/train.csv"
MODEL_PATH = "guzel/256Hidden_5Layer_lstm_model_epoch90.pth"  # Buraya hangi checkpoint'i istiyorsan onu yaz
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
TARGET_FEATURE = "Close"
TARGET_INDEX = FEATURES.index(TARGET_FEATURE)
WINDOW_SIZE = 30
PRED_SIZE = 10
SCALER_PATH = "outputs/scaler.save"

df = pd.read_csv(CSV_PATH)
df = df.sort_values("Date")
feature_df = df[FEATURES].astype(float)
close = feature_df[TARGET_FEATURE].values

pctchange_df = feature_df.pct_change().dropna()
pctchange_np = pctchange_df.values

scaler = joblib.load(SCALER_PATH)
pctchange_scaled = scaler.transform(pctchange_np)

last_40 = pctchange_scaled[-40:]
input_seq = last_40[:30, :]
true_future = last_40[30:, TARGET_INDEX]
X_input = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

close = close[-(40+1):]
last_real_price = close[-11]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMForecast(input_size=len(FEATURES), output_size=PRED_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    preds = model(X_input.to(device))
    tahmin_pct = preds[0].cpu().numpy().flatten()
    tahmin_pct_inv = scaler.inverse_transform(
        np.stack([np.zeros(PRED_SIZE) if i != TARGET_INDEX else tahmin_pct for i in range(len(FEATURES))], axis=1)
    )[:, TARGET_INDEX]

    true_pct_inv = scaler.inverse_transform(
        np.stack([np.zeros(PRED_SIZE) if i != TARGET_INDEX else true_future for i in range(len(FEATURES))], axis=1)
    )[:, TARGET_INDEX]
    real_future_prices = [last_real_price]
    for pct in true_pct_inv:
        real_future_prices.append(real_future_prices[-1] * (1 + pct))
    real_future_prices = real_future_prices[1:]

    predicted_prices = [last_real_price]
    for pct in tahmin_pct_inv:
        predicted_prices.append(predicted_prices[-1] * (1 + pct))
    predicted_prices = predicted_prices[1:]

    # Sadece son 10 gün grafikle:
    plt.figure(figsize=(10,5))
    plt.plot(range(30, 40), real_future_prices, label="Gerçek (son 10 gün)")
    plt.plot(range(30, 40), predicted_prices, label="Tahmin (son 10 gün)")
    mse = np.mean((np.array(predicted_prices) - np.array(real_future_prices))**2)
    rmse = np.sqrt(mse)
    plt.title(f"Son 10 Gün Tahmini\nMSE: {mse:.2f} | RMSE: {rmse:.2f}\nModel: {os.path.basename(MODEL_PATH)}")
    plt.xlabel("Gün (Pencere)")
    plt.ylabel("Fiyat")
    plt.legend()
    plt.tight_layout()
    plt.show()
