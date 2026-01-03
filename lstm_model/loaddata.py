from typing import Tuple, Dict
import numpy as np
import pandas as pd

from src.read_data import read_all_csvs
from src.make_windows import make_reverse_windows
from src.window_percentage import window_percentage


def loaddata(
    data_dir: str = "data",
    window: int = 31,   # 31 timestep -> 32 ham veri
    step: int = 1,
    order: str = "old_to_new",
    prev_zero: str = "zero",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:

    data = read_all_csvs(data_dir, drop_date=True)
    if not data:
        raise FileNotFoundError(f"{data_dir} içinde CSV bulunamadı.")

    X_list, y_list, prev_close_list, sources = [], [], [], []
    feature_names = None  # 🔹 Burada başlatıyoruz

    for name, df in data.items():
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 5:
            print(f"[{name}] yeterli numerik kolon yok, atlandı.")
            continue

        df_use = df[num_cols[:5]].copy()  # Open, High, Low, Close, Volume
        wins = make_reverse_windows(df_use, window=window + 1, step=step, order=order)

        for w in wins:
            if len(w) < window + 1:
                continue

            hist = w.iloc[:-1]   # geçmiş 31 veri
            fut  = w.iloc[-1]    # 32. gün (gelecek)

            # === y: close_ret, high_delta, low_delta (hepsi yüzdelik) ===
            now_close = float(fut.iloc[3])  # bugünün close'u
            now_high  = float(fut.iloc[1])  # bugünün high'ı
            now_low   = float(fut.iloc[2])  # bugünün low'u

            prev_close = float(hist.iloc[-1, 3])  # dünün close'u

            # 1) dünkü close'a göre bugünkü close
            close_ret  = now_close / prev_close - 1.0

            # 2) bugünkü close'a göre ne kadar yukarıda (pozitif)
            high_delta = now_high / now_close - 1.0

            # 3) bugünkü close'a göre ne kadar aşağıda (pozitif mesafe)
            #    low_close_dist = (close - low) / close
            #    = 1 - now_low / now_close
            low_delta  = 1.0 - (now_low / now_close)

            arrY = np.array([close_ret, high_delta, low_delta], dtype=np.float32)

            # === prev_close: dünün kapanışı ===
            prev_close_list.append(prev_close)

            # === X: geçmiş yüzdelik pencere ===
            pct = window_percentage(hist, numeric_only=True, prev_zero=prev_zero)
            arrX = pct.to_numpy(dtype=np.float32)
            if arrX.shape[0] == 0:
                continue

            # 🔹 Feature isimlerini ilk geçerli pencerede kaydet
            if feature_names is None:
                feature_names = list(pct.columns)

            X_list.append(arrX)
            y_list.append(arrY)
            sources.append(name)

    # --- Stack & meta ---
    if not X_list:
        X = np.empty((0, window - 1, 5), dtype=np.float32)
        y = np.empty((0, 3), dtype=np.float32)
        prev_close_arr = np.empty((0,), dtype=np.float32)
    else:
        X = np.stack(X_list)
        y = np.stack(y_list)
        prev_close_arr = np.array(prev_close_list, dtype=np.float32)

    from collections import Counter
    meta = {
        "sources": sources,
        "per_source_counts": dict(Counter(sources)),
        "n_windows": len(X_list),
        "window_ham": window + 1,
        "window_return": window - 1,
        "step": step,
        "order": order,
        "timesteps": X.shape[1],
        "n_features": X.shape[2],
        "n_outputs": 3,
        "feature_names": feature_names,  # 🔹 BURADA ARTIK VAR
    }

    return X, y, prev_close_arr, meta
