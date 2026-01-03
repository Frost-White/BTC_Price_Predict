import pandas as pd


def _parse_dollar_number(val):
    """
    '$92,513.67' -> 92513.67
    '$80,275,884,583' -> 80275884583.0
    """
    if pd.isna(val):
        return None

    s = str(val).strip()
    if s in ("", "-", "--", "N/A"):
        return None

    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def preprocess_btc_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    CoinMarketCap BTC historical data -> modele hazır OHLCV
    """

    # 1️⃣ kronolojik sıraya çevir (eski → yeni)
    df = df.iloc[::-1].reset_index(drop=True)

    # 2️⃣ Date kolonunu sil
    for col in df.columns:
        if "date" in col.lower():
            df = df.drop(columns=[col])
            break

    # 3️⃣ Market Cap kolonunu sil
    for col in df.columns:
        if "market" in col.lower():
            df = df.drop(columns=[col])
            break

    # 4️⃣ kalan kolonları numeriğe çevir
    for col in df.columns:
        df[col] = df[col].apply(_parse_dollar_number)

    # 5️⃣ NaN içeren satırları temizle
    df = df.dropna().reset_index(drop=True)

    return df


# --- TEST ---
if __name__ == "__main__":
    from getdata import fetch_btc_history

    df_raw = fetch_btc_history(window_size=31)

    if df_raw.empty:
        print("❌ getdata boş döndü")
    else:
        df_processed = preprocess_btc_dataframe(df_raw)
        print("RAW:", df_raw.shape)
        print("PROCESSED:", df_processed.shape)
        print(df_processed)
