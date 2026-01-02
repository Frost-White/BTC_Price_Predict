# main.py
import pandas as pd
from pathlib import Path

from getdata import fetch_btc_history
from proccessdata import preprocess_btc_dataframe, _parse_dollar_number
from window_percentage import window_percentage
from run_lstm import run_lstm_inference


def main():
    pd.set_option("display.float_format", "{:,.7f}".format)

    # --- 1️⃣ Model yolu: main.py'nin bulunduğu klasöre göre ---
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "model.pt"

    # --- 2️⃣ Veri çekme ---
    df_raw = fetch_btc_history(31)
    if df_raw.empty:
        print("❌ Veri alınamadı.")
        input("enter")
        return

    # --- 3️⃣ Ön işleme ---
    df_processed = preprocess_btc_dataframe(df_raw)

    # --- 4️⃣ Yüzdelik değişimler (pencere serisi) ---
    pct_df = window_percentage(df_processed, numeric_only=True, prev_zero="zero")

    # --- 5️⃣ LSTM tahmini (3 çıktı: close_ret, high_delta, low_delta) ---
    pred = run_lstm_inference(pct_df, str(model_path), hidden=128, layers=3)

    close_ret = float(pred.loc[0, "close_ret"])
    high_delta = float(pred.loc[0, "high_delta"])
    low_delta  = float(pred.loc[0, "low_delta"])

    # --- 6️⃣ Dünkü kapanıştan bugünkü fiyatları türet ---
    # df_raw'ın İLK satırı en güncel (dünkü) değerleri tutuyor
    raw_close_val = df_raw.iloc[0]["Close*"]
    # Virgülleri temizle, floata çevir
    close_prev = _parse_dollar_number(raw_close_val)

    # Bugünkü tahmini close
    close_today = close_prev * (1.0 + close_ret)

    # Bugünkü tahmini high / low
    high_today = close_today * (1.0 + high_delta)
    low_today  = close_today * (1.0 - low_delta)

    # --- 7️⃣ Sonuçları yazdır ---
    print("\n📅 Bugünün tahmin edilen fiyatları:")
    print(f"Dünkü Close (base) : {close_prev:,.2f}")
    print(f"Bugünkü Close tahmini: {close_today:,.2f}")
    print(f"Bugünkü High tahmini : {high_today:,.2f}")
    print(f"Bugünkü Low tahmini  : {low_today:,.2f}")

    input("\n✅ Program tamamlandı. Çıkmak için Enter’a basın...")


if __name__ == "__main__":
    main()
