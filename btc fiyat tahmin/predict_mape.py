# predict_mape_last_year.py

import pandas as pd
import numpy as np
import torch
import joblib
from dataset import create_sequences_multifeature_pctchange
from model import LSTMForecast

CSV_PATH = "data/train.csv"
MODEL_PATH = "guzel/256Hidden_5Layer_lstm_model_epoch90.pth"
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
TARGET_FEATURE = "Close"
TARGET_INDEX = FEATURES.index(TARGET_FEATURE)
WINDOW_SIZE = 30
PRED_SIZE = 10
SCALER_PATH = "outputs/scaler.save"

# --- Veri ve modeli yükle ---
df = pd.read_csv(CSV_PATH)
df = df.sort_values("Date")
df["Date"] = pd.to_datetime(df["Date"])
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

# --- Son yılın başlangıç indexini bul ---
son_tarih = df["Date"].iloc[-1]
yil_once = son_tarih - pd.Timedelta(days=365)
first_idx_of_last_year = df[df["Date"] >= yil_once].index[0]

# --- Sliding window (overlapping) şekilde tüm pencereler için ---
mape_list = []
n_windows = X.shape[0]

for i in range(n_windows):
    pencere_baslangic_indexi = i + WINDOW_SIZE - 1
    # Sadece son yılda başlayan pencereler
    if pencere_baslangic_indexi < first_idx_of_last_year:
        continue

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

print(f"SON YIL için toplam {len(mape_list)} pencere üzerinden ortalama yüzdelik sapma (MAPE): {np.mean(mape_list):.2f}%")
