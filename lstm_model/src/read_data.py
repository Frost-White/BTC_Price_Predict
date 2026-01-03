# read_data.py
from pathlib import Path
import pandas as pd
import re

# Sık görülen tarih/saat kolon adları (istersen kapatabilirsin: drop_date=False)
COMMON_DATE_COLS = {
    "date", "datetime", "timestamp", "time", "open_time", "close_time"
}

def drop_date_like_columns(df: pd.DataFrame, extra_names=None) -> pd.DataFrame:
    """
    Tarih/saat benzeri kolonları düşer. (Sadece istersen)
    """
    extra = set(map(str.lower, extra_names or []))
    to_drop = []
    for col in df.columns:
        name = str(col).lower()
        if name in COMMON_DATE_COLS or name in extra:
            to_drop.append(col)
            continue
        # Not: raw string ile \b kullanılmalı (\\b değil)
        if re.search(r"(date|time|timestamp|datetime|_ts\b|\bts_)", name):
            to_drop.append(col)
    if to_drop:
        df = df.drop(columns=to_drop)
    return df

# Numerik gibi kabul edeceğimiz kolon isimleri (lower-case karşılaştırılır)
NUMERIC_LIKE_COLS = {
    "open", "high", "low", "close", "adj close", "adj_close",
    "volume", "vol", "turnover"
}

def _coerce_numeric_like_commas_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sadece virgül (,) temizleyip sayıya çevirir.
    Başka hiçbir kontrol yok. Sade ve hızlı.
    """
    out = df.copy()
    lower_map = {c: str(c).lower() for c in out.columns}

    for col in out.columns:
        lname = lower_map[col]
        # İsim eşleşmesi (tam ve basit regex) — sadece virgül temizlenir.
        if (lname in NUMERIC_LIKE_COLS) or re.search(r"(open|high|low|close|adj\s*close|vol|volume|turnover)", lname):
            s = out[col].astype(str)
            # Sadece virgül kaldır
            s = s.str.replace(",", "", regex=False)
            # pd.to_numeric: sayıya çevir; çeviremiyorsa NaN yapar
            out[col] = pd.to_numeric(s, errors="coerce")
    return out

def read_all_csvs(
    data_dir: str = "data",
    drop_date: bool = True,
    extra_date_cols=None
) -> dict[str, pd.DataFrame]:
    """
    data/ klasöründeki tüm .csv dosyalarını okur ve dict döndürür.
    Anahtar = dosya adı (uzantısız), Değer = DataFrame (virgüller temizlenmiş numerik kolonlar)
    """
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {p.resolve()}")

    files = sorted(p.glob("*.csv"))
    if not files:
        return {}

    out = {}
    for f in files:
        # UTF-8-SIG yaygın; gerekirse encoding parametresini değiştirirsin
        df = pd.read_csv(f, encoding="utf-8-sig")

        # İstersen tarih/saat kolonlarını düş
        if drop_date:
            df = drop_date_like_columns(df, extra_names=extra_date_cols)

        # Sadece virgül temizle + sayıya çevir
        df = _coerce_numeric_like_commas_only(df)

        out[f.stem] = df

    return out

if __name__ == "__main__":
    # Hızlı test
    data = read_all_csvs("data", drop_date=False)
    print(f"Okunan dosya sayısı: {len(data)}")
    for name, df in data.items():
        print(f"- {name}: shape={df.shape}")
        print(df.dtypes.head())
        break
