import os
import json
import torch

from model import LSTMRegressor
from loaddata import loaddata
import eval as eval_mod  # evaluate_model fonksiyonu burada


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("eval_metrics", exist_ok=True)
    os.makedirs("eval_metrics/best_models", exist_ok=True)

    model_dir = "outputs"
    train_metrics_dir = "train_metrics"

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

    if not model_files:
        print("⚠️ Hiç .pt modeli bulunamadı.")
        return

    # --- En iyi modelleri takip etmek için değişkenler ---
    best_close = {"mae": float("inf"), "tag": None, "metrics": None}
    best_high = {"mae": float("inf"), "tag": None, "metrics": None}
    best_low = {"mae": float("inf"), "tag": None, "metrics": None}

    # --- Tüm modelleri dolaş ---
    for model_file in model_files:
        tag = os.path.splitext(model_file)[0]
        model_path = os.path.join(model_dir, model_file)

        config_path = os.path.join(train_metrics_dir, tag, "config.json")
        if not os.path.isfile(config_path):
            print(f"[WARN] {tag} için config.json bulunamadı, atlanıyor.")
            continue

        with open(config_path, "r") as f:
            cfg = json.load(f)

        window = cfg.get("window")
        step = cfg.get("step", 1)
        hidden = cfg.get("hidden")
        layers = cfg.get("layers")
        dropout = cfg.get("dropout", 0.0)

        print(f"\n🔍 Değerlendirilen model: {tag}")
        print(
            f"    window={window}, step={step}, hidden={hidden}, "
            f"layers={layers}, dropout={dropout}"
        )

        # --- Eval data yükle ---
        X, y, prev_close, meta = loaddata(
            data_dir="eval_data",
            window=window,
            step=step,
            order="old_to_new",
            prev_zero="zero",
        )
        print(
            f"[INFO] Eval verisi: X={X.shape}, y={y.shape}, prev_close={prev_close.shape}"
        )

        save_dir = os.path.join("eval_metrics", tag)
        os.makedirs(save_dir, exist_ok=True)

        # --- Model kur ---
        model = LSTMRegressor(
            input_dim=meta["n_features"],
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            out_dim=meta["n_outputs"],
        ).to(device)

        try:
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
        except Exception as e:
            print(f"[ERROR] {model_file} yüklenemedi: {e}")
            continue

        # --- Değerlendir ---
        metrics, outlier_info = eval_mod.evaluate_model(
            model, X, y, prev_close, device=device
        )

        # --- Kaydet ---
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(save_dir, "outlier_info.json"), "w") as f:
            json.dump(outlier_info, f, indent=2)

        # --- EKRANA YAZDIR ---
        print(f"📊 {tag} sonuçları (YÜZDE bazlı):")
        for k, v in metrics.items():
            line = (
                f"  {k}: MAE%={v['mae_pct']:.3f}%, "
                f"Min%={v['min_err_pct']:.3f}%, "
                f"Max%={v['max_err_pct']:.3f}%"
            )
            if k == "close" and "sign_acc_pct" in v:
                line += f", SignAcc%={v['sign_acc_pct']:.2f}%"

            # high / low için raw delta MAE%
            if k in ("high", "low") and "raw_delta_mae_pct" in v:
                line += f", RawDeltaMAE%={v['raw_delta_mae_pct']:.3f}%"

            print(line)

        for k, v in outlier_info.items():
            print(
                f"  ⚠️ {k}: {v['count']} pencere > 4xMAE% "
                f"(eşik={v['threshold_pct']:.3f}%)"
            )

        # --- En iyi close modelini güncelle ---
        if metrics["close"]["mae_pct"] < best_close["mae"]:
            best_close["mae"] = metrics["close"]["mae_pct"]
            best_close["tag"] = tag
            best_close["metrics"] = metrics

        # --- En iyi high (delta high gibi düşün) ---
        if metrics["high"]["mae_pct"] < best_high["mae"]:
            best_high["mae"] = metrics["high"]["mae_pct"]
            best_high["tag"] = tag
            best_high["metrics"] = metrics

        # --- En iyi low (delta low gibi düşün) ---
        if metrics["low"]["mae_pct"] < best_low["mae"]:
            best_low["mae"] = metrics["low"]["mae_pct"]
            best_low["tag"] = tag
            best_low["metrics"] = metrics

    # --- En iyi modelleri kaydet ---
    with open("eval_metrics/best_models/best_close.json", "w") as f:
        json.dump(best_close, f, indent=2)

    with open("eval_metrics/best_models/best_high.json", "w") as f:
        json.dump(best_high, f, indent=2)

    with open("eval_metrics/best_models/best_low.json", "w") as f:
        json.dump(best_low, f, indent=2)

    print("\n🏆 En iyi modeller kaydedildi → eval_metrics/best_models/")


if __name__ == "__main__":
    main()
