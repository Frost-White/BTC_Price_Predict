# train_by10.py
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
windows = [31]          # pencere boyutları listesi
hiddens = [128]         # hidden size listesi
layers_list = [3]       # layer sayısı listesi
batch_sizes = [16]      # batch size listesi

step = 1
dropout = 0.1
lr = 0.000005 # orj 0.0005
epochs = 400
save_every = 10         # her kaç epochta bir checkpoint alınacak

# === Klasörler ===
os.makedirs("outputs", exist_ok=True)
os.makedirs("train_metrics", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_checkpoint(
    base_tag,
    epoch,
    model,
    train_log,
    window,
    step,
    hidden,
    layers,
    dropout,
    lr,
    batch_size,
    start_time,
):
    """
    O ana kadarki modeli ve metrikleri kaydeder.
    base_tag: winXX_hidYY_layersZZ_bsBB_lrLLL
    epoch   : şu anki epoch
    """
    ckpt_tag = f"{base_tag}_ep{epoch:03d}"

    # Klasörler
    model_path = os.path.join("outputs", f"{ckpt_tag}.pt")
    metrics_dir = os.path.join("train_metrics", ckpt_tag)
    os.makedirs(metrics_dir, exist_ok=True)

    # Model ağırlıkları
    torch.save(model.state_dict(), model_path)

    # Şu ana kadarki train log (epoch 1..epoch)
    partial_log = [entry for entry in train_log if entry["epoch"] <= epoch]
    with open(os.path.join(metrics_dir, "train_log.json"), "w") as f:
        json.dump(partial_log, f, indent=2)

    # Config
    duration_sec = (datetime.now() - start_time).total_seconds()
    params = {
        "window": window,
        "step": step,
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
        "lr": lr,
        "epochs_total_planned": epochs,
        "epochs_trained": epoch,
        "batch_size": batch_size,
        "duration_sec": duration_sec,
        "loss_type": "percent_space_MSE",
    }
    with open(os.path.join(metrics_dir, "config.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Loss grafiği (o ana kadarki)
    loss_values = [entry["loss"] for entry in partial_log]
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(loss_values) + 1),
        loss_values,
        label="Train Loss",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Percent-space MSE")
    plt.title(f"Training Loss Curve\n{ckpt_tag}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    loss_fig_path = os.path.join(metrics_dir, "loss_curve.png")
    plt.savefig(loss_fig_path, dpi=150)
    plt.close()

    print(f"💾 [Ep {epoch:03d}] Checkpoint kaydedildi → {model_path}")
    print(f"📈 [Ep {epoch:03d}] Loss grafiği kaydedildi → {loss_fig_path}")


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

    print(
        f"\n[INFO] Eğitim verisi yüklendi (window={window}, step={step}): "
        f"{meta['n_windows']} pencere"
    )
    print(f"X shape: {X.shape}, y shape: {y.shape}, prev_close shape: {prev_close.shape}")

    # --- Tensör dönüşümü ---
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    prev_t = torch.tensor(prev_close, dtype=torch.float32).unsqueeze(1)
    dataset_len = len(X_t)

    for hidden in hiddens:
        for layers in layers_list:
            for batch_size in batch_sizes:
                print(
                    f"\n🚀 Yeni eğitim başlıyor — "
                    f"window={window}, hidden={hidden}, layers={layers}, "
                    f"batch_size={batch_size}, lr={lr}, dropout={dropout}"
                )

                train_loader = DataLoader(
                    TensorDataset(X_t, y_t, prev_t),
                    batch_size=batch_size,
                    shuffle=True,
                )

                model = LSTMRegressor(
                    input_dim=meta["n_features"],
                    hidden=hidden,
                    layers=layers,
                    dropout=dropout,
                    out_dim=meta["n_outputs"],
                ).to(device)

                optimizer = Lion(model.parameters(), lr=lr)

                train_log = []
                start_time = datetime.now()

                base_tag = (
                    f"win{window}_hid{hidden}_layers{layers}_"
                    f"bs{batch_size}_lr{lr}"
                )

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
                    print(
                        f"[win={window} hid={hidden} lay={layers} bs={batch_size}] "
                        f"[{epoch:03d}/{epochs}] loss={avg_loss:.6f}"
                    )

                    # --- Her 10 epochta (veya son epokta) checkpoint al ---
                    if (epoch % save_every == 0) or (epoch == epochs):
                        save_checkpoint(
                            base_tag=base_tag,
                            epoch=epoch,
                            model=model,
                            train_log=train_log,
                            window=window,
                            step=step,
                            hidden=hidden,
                            layers=layers,
                            dropout=dropout,
                            lr=lr,
                            batch_size=batch_size,
                            start_time=start_time,
                        )

print("\n🎯 Tüm hiperparametre kombinasyonları için eğitimler ve checkpointler tamamlandı.")
