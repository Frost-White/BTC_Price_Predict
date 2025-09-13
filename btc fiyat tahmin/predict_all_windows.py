import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import joblib
from dataset import create_sequences_multifeature_pctchange
from model import LSTMForecast

CSV_PATH = "data/train.csv"
MODEL_PATH = "guzel/256Hidden_5Layer_lstm_model_epoch100.pth"  # Seçtiğin modeli yaz
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

X, y = create_sequences_multifeature_pctchange(pctchange_scaled, WINDOW_SIZE, PRED_SIZE, TARGET_INDEX)
close = close[-len(pctchange_scaled):]  # align

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMForecast(input_size=len(FEATURES), output_size=PRED_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

mape_list = []
n_windows = X.shape[0]

indices = list(range(0, n_windows, WINDOW_SIZE))
last_10_indices = indices[-10:]  # Son 10 pencere

for i in last_10_indices:
    X_input = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
    last_real_price = close[i + WINDOW_SIZE - 1]
    with torch.no_grad():
        preds = model(X_input)
        tahmin_pct = preds[0].cpu().numpy().flatten()
        tahmin_pct_inv = scaler.inverse_transform(
            np.stack([np.zeros(PRED_SIZE) if k != TARGET_INDEX else tahmin_pct for k in range(len(FEATURES))], axis=1)
        )[:, TARGET_INDEX]

        true_pct_inv = scaler.inverse_transform(
            np.stack([np.zeros(PRED_SIZE) if k != TARGET_INDEX else y[i] for k in range(len(FEATURES))], axis=1)
        )[:, TARGET_INDEX]

        real_future_prices = [last_real_price]
        for pct in true_pct_inv:
            real_future_prices.append(real_future_prices[-1] * (1 + pct))
        real_future_prices = real_future_prices[1:]

        predicted_prices = [last_real_price]
        for pct in tahmin_pct_inv:
            predicted_prices.append(predicted_prices[-1] * (1 + pct))
        predicted_prices = predicted_prices[1:]

        mape = np.mean(np.abs((np.array(predicted_prices) - np.array(real_future_prices)) / np.array(real_future_prices))) * 100
        mape_list.append(mape)

        plt.figure(figsize=(10,5))
        plt.plot(range(i + WINDOW_SIZE, i + WINDOW_SIZE + PRED_SIZE), real_future_prices, label="Gerçek (10 gün)")
        plt.plot(range(i + WINDOW_SIZE, i + WINDOW_SIZE + PRED_SIZE), predicted_prices, label="Tahmin (10 gün)")
        plt.title(f"{i}-{i+WINDOW_SIZE} arası pencere | Ortalama Yüzdelik Sapma (MAPE): {mape:.2f}%")
        plt.xlabel("Gün (Pencere)")
        plt.ylabel("Fiyat")
        plt.legend()
        plt.tight_layout()
        plt.show()

print(f"Adımlı 10 pencere için ortalama yüzdelik sapma (MAPE): {np.mean(mape_list):.2f}%")
