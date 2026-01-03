# train_new.py
import os
import json
import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import matplotlib.pyplot as plt

from lion_pytorch import Lion

from model import LSTMRegressor
from loaddata import loaddata

# === Seed sabitleme ===
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# === Hiperparametreler ===
# NOT: lr, epochs, dropout, step scalar kaldı
windows = [31]          # pencere boyutları listesi
hiddens = [128]         # hidden size listesi
layers_list = [3]       # layer sayısı listesi
batch_sizes = [16]      # batch size listesi

step = 1
dropout = 0.1
lr = 0.0005
epochs = 270

# === Klasörler ===
os.makedirs("outputs", exist_ok=True)
os.makedirs("train_metrics", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Her window / hidden / layer / batch_size kombinasyonu için eğitim ===
for window in windows:
    # --- Veri yükle (window ve step'e göre) ---
    X, y, prev_close, meta = loaddata(
        data_dir="data",
        window=window,
        step=step,
        order="old_to_new",
        prev_zero="zero",
    )

    print(f"\n[INFO] Eğitim verisi yüklendi (window={window}, step={step}): "
          f"{meta['n_windows']} pencere")
    print(f"X shape: {X.shape}, y shape: {y.shape}, prev_close shape: {prev_close.shape}")

    # --- Tensör dönüşümü ---
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    prev_t = torch.tensor(prev_close, dtype=torch.float32).unsqueeze(1)
    dataset_len = len(X_t)

    for hidden in hiddens:
        for layers in layers_list:
            for batch_size in batch_sizes:
                print(f"\n🚀 Yeni eğitim başlıyor — "
                      f"window={window}, hidden={hidden}, layers={layers}, "
                      f"batch_size={batch_size}, lr={lr}, dropout={dropout}")

                train_loader = DataLoader(
                    TensorDataset(X_t, y_t, prev_t),
                    batch_size=batch_size,
                    shuffle=True
                )

                model = LSTMRegressor(
                    input_dim=meta["n_features"],
                    hidden=hidden,
                    layers=layers,
                    dropout=dropout,
                    out_dim=meta["n_outputs"]
                ).to(device)

                optimizer = Lion(model.parameters(), lr=lr)

                train_log = []
                start_time = datetime.now()

                for epoch in range(1, epochs + 1):
                    model.train()
                    total_loss = 0.0

                    for xb, yb, prevb in train_loader:
                        xb, yb, prevb = xb.to(device), yb.to(device), prevb.to(device)
                        optimizer.zero_grad()

                        # Model tahmini: [pred_close_ret, pred_high_delta, pred_low_delta]
                        pred = model(xb)

                        # === Loss hesaplama (doğrudan yüzde uzayında, MSE) ===
                        loss = F.mse_loss(pred, yb)

                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item() * len(xb)

                    avg_loss = total_loss / dataset_len
                    train_log.append({"epoch": epoch, "loss": avg_loss})
                    print(f"[win={window} hid={hidden} lay={layers} bs={batch_size}] "
                          f"[{epoch:03d}/{epochs}] loss={avg_loss:.6f}")

                # === Eğitim süresi ===
                duration_sec = (datetime.now() - start_time).total_seconds()
                print(f"⏱ Eğitim süresi (win={window}, hid={hidden}, layers={layers}, "
                      f"bs={batch_size}): {duration_sec:.2f} saniye")

                # === Kayıt işlemleri ===
                tag = (
                    f"win{window}_hid{hidden}_layers{layers}_"
                    f"bs{batch_size}_lr{lr}"
                )
                model_path = os.path.join("outputs", f"{tag}.pt")
                metrics_dir = os.path.join("train_metrics", tag)
                os.makedirs(metrics_dir, exist_ok=True)

                # Eğitim konfig bilgileri
                params = {
                    "window": window,
                    "step": step,
                    "hidden": hidden,
                    "layers": layers,
                    "dropout": dropout,
                    "lr": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "duration_sec": duration_sec,
                    "loss_type": "percent_space_MSE",
                }

                # feature_names mutlaka meta'dan gelsin (inference hizalaması için)
                feature_names = meta.get("feature_names")
                if feature_names is None:
                    raise RuntimeError("meta['feature_names'] yok; loaddata bunu doldurmalı.")

                # Checkpoint yapısı: run_lstm.py buna göre okuyor
                ckpt = {
                    "model_state": model.state_dict(),
                    "feature_names": feature_names,
                    "config": params,
                }

                # Model checkpoint
                torch.save(ckpt, model_path)

                # Train log
                with open(os.path.join(metrics_dir, "train_log.json"), "w") as f:
                    json.dump(train_log, f, indent=2)

                # Config’i ayrıca JSON olarak da kaydet
                with open(os.path.join(metrics_dir, "config.json"), "w") as f:
                    json.dump(params, f, indent=2)

                # === Loss grafiği ===
                loss_values = [entry["loss"] for entry in train_log]
                plt.figure(figsize=(8, 5))
                plt.plot(
                    range(1, len(loss_values) + 1),
                    loss_values,
                    label="Train Loss",
                    linewidth=2,
                )
                plt.xlabel("Epoch")
                plt.ylabel("Percent-space MSE")
                plt.title(f"Training Loss Curve\n{tag}")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend()
                plt.tight_layout()

                loss_fig_path = os.path.join(metrics_dir, "loss_curve.png")
                plt.savefig(loss_fig_path, dpi=150)
                plt.close()

                print(f"💾 Model kaydedildi → {model_path}")
                print(f"📈 Loss grafiği kaydedildi → {loss_fig_path}")
                print("✅ Eğitim tamamlandı")

print("\n🎯 Tüm hiperparametre kombinasyonları için eğitimler tamamlandı.")
