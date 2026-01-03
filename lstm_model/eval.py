import os
import json
import torch
import numpy as np

from model import LSTMRegressor
from loaddata import loaddata


def evaluate_model(model, X, y, prev_close, device="cpu"):
    model.eval()
    preds = []
    trues = []

    true_close_ret_list = []
    pred_close_ret_list = []

    # --- raw delta listeleri (fiyat dönüşümü yapmadan) ---
    true_high_delta_list = []
    pred_high_delta_list = []
    true_low_delta_list = []
    pred_low_delta_list = []

    with torch.no_grad():
        for i in range(len(X)):
            xb = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
            pred_vec = model(xb).cpu().numpy()[0]  # [close_ret, high_delta, low_delta] (tahmin)

            # === Referans fiyat ===
            ref = float(prev_close[i])  # dünkü kapanış

            # --- 1) GERÇEK değerler (y) ---
            cr, hd, ld = [float(v) for v in y[i]]  # close_ret, high_delta, low_delta (gerçek)

            # low_delta: close'a göre pozitif mesafe: (close - low)/close

            # --- 2) TAHMİN edilen raw değerler ---
            pred_cr, pred_hd, pred_ld = [float(v) for v in pred_vec]

            # --- 3) Bunlardan fiyatları hesapla ---
            true_close = ref * (1.0 + cr)
            true_high = true_close * (1.0 + hd)
            true_low = true_close * (1.0 - ld)

            pred_close = ref * (1.0 + pred_cr)
            pred_high = pred_close * (1.0 + pred_hd)
            pred_low = pred_close * (1.0 - pred_ld)

            preds.append([pred_close, pred_high, pred_low])
            trues.append([true_close, true_high, true_low])

            # --- CLOSE yön tahmini için close_ret'leri sakla ---
            true_close_ret_list.append(cr)
            pred_close_ret_list.append(pred_cr)

            # --- RAW delta'ları da sakla (fiyat dönüştürmeden) ---
            true_high_delta_list.append(hd)
            pred_high_delta_list.append(pred_hd)
            true_low_delta_list.append(ld)
            pred_low_delta_list.append(pred_ld)

    preds = np.array(preds)  # [N, 3]
    trues = np.array(trues)  # [N, 3]

    # --- 4) Fiyat uzayında hatalar: önce fiyat, sonra yüzde ---
    errors_price = np.abs(preds - trues)  # fiyat hatası
    denom = np.clip(np.abs(trues), 1e-8, None)  # 0'a bölme engeli
    errors_pct = errors_price / denom * 100.0  # yüzde hata

    metrics = {}
    outlier_info = {}

    names = ["close", "high", "low"]
    for i, name in enumerate(names):
        # Ortalama / min / max YÜZDE hata
        mae_pct = float(np.mean(errors_pct[:, i]))
        min_err_pct = float(np.min(errors_pct[:, i]))
        max_err_pct = float(np.max(errors_pct[:, i]))

        # Outlier eşiği: 4x MAE%
        thr_pct = 4.0 * mae_pct
        idxs = np.where(errors_pct[:, i] > thr_pct)[0].tolist()

        metrics[name] = {
            "mae_pct": mae_pct,
            "min_err_pct": min_err_pct,
            "max_err_pct": max_err_pct,
        }

        outlier_info[name] = {
            "threshold_pct": round(thr_pct, 3),
            "count": int(len(idxs)),
            "indexes": idxs,
        }

    # --- CLOSE için işaret (yön) doğruluğu ---
    true_cr_arr = np.array(true_close_ret_list)
    pred_cr_arr = np.array(pred_close_ret_list)

    true_sign = np.sign(true_cr_arr)
    pred_sign = np.sign(pred_cr_arr)

    sign_correct = (true_sign == pred_sign)
    sign_acc_pct = float(sign_correct.mean() * 100.0)

    metrics["close"]["sign_acc_pct"] = sign_acc_pct

    # --- RAW delta metrikleri (fiyat dönüşümü olmadan, sadece YÜZDE MAE) ---
    true_hd_arr = np.array(true_high_delta_list)
    pred_hd_arr = np.array(pred_high_delta_list)
    true_ld_arr = np.array(true_low_delta_list)
    pred_ld_arr = np.array(pred_low_delta_list)

    # delta'lar oransal (0.05 -> %5). Aradaki farkın ortalamasını alıp %'ye çeviriyoruz.
    high_delta_mae_pct = float(np.mean(np.abs(pred_hd_arr - true_hd_arr)) * 100.0)
    low_delta_mae_pct = float(np.mean(np.abs(pred_ld_arr - true_ld_arr)) * 100.0)

    # high / low metriclerine ekle
    metrics["high"]["raw_delta_mae_pct"] = high_delta_mae_pct
    metrics["low"]["raw_delta_mae_pct"] = low_delta_mae_pct

    return metrics, outlier_info


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Eval verisini yükle ===
    X, y, prev_close, meta = loaddata(
        data_dir="eval_data",
        window=31,  # eğitimde ne kullandıysan onunla sen oynarsın
        step=1,
        order="old_to_new",
        prev_zero="zero",
    )
    print(f"[INFO] Eval verisi: X={X.shape}, y={y.shape}, prev_close={prev_close.shape}")

    os.makedirs("eval_metrics_single", exist_ok=True)

    # Burada tek bir model için örnek; istersen dosya adını değiştir
    model_path = "outputs/win31_hid128_layers3_bs16_lr5e-05.pt"
    if not os.path.isfile(model_path):
        print(f"⚠️ {model_path} bulunamadı.")
        return

    model = LSTMRegressor(
        input_dim=meta["n_features"],
        hidden=128,
        layers=3,
        dropout=0.1,
        out_dim=meta["n_outputs"]
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    metrics, outlier_info = evaluate_model(model, X, y, prev_close, device=device)

    # Kaydetme opsiyonel, temel amaç burada print
    with open(os.path.join("eval_metrics_single", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join("eval_metrics_single", "outlier_info.json"), "w") as f:
        json.dump(outlier_info, f, indent=2)

    print("\n📊 Sonuçlar (YÜZDE bazlı):")
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

    print("\n✅ Eval tamamlandı.")


if __name__ == "__main__":
    main()
