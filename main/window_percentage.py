# window_percentage.py
import numpy as np
import pandas as pd

def window_percentage(
    window_df: pd.DataFrame,
    absolute: bool = False,
    numeric_only: bool = True,
    prev_zero: str = "nan",   # "nan" veya "zero"
    eps: float = 1e-9,
) -> pd.DataFrame:
    if window_df.shape[0] < 2:
        return pd.DataFrame(columns=window_df.columns, index=[])

    # Sadece numerik kolonlar
    if numeric_only:
        cols = window_df.select_dtypes(include=[np.number]).columns
        df = window_df[cols].astype(float).copy()
    else:
        df = window_df.apply(pd.to_numeric, errors="coerce").copy()
        cols = df.columns

    prev = df.shift(1).iloc[1:]   # t-1
    curr = df.iloc[1:]            # t
    denom = prev

    # zero_mask'ı her durumda tanımla (lint sakinleşir)
    zero_mask = pd.DataFrame(False, index=denom.index, columns=denom.columns)

    if prev_zero == "nan":
        denom = denom.replace(0.0, np.nan)
    elif prev_zero == "zero":
        zero_mask = denom.eq(0.0)
        denom = denom + eps
    else:
        denom = denom + eps

    pct = (curr - prev) / denom

    if prev_zero == "zero":
        # Önceki 0 olan yerleri 0’a çek
        pct[zero_mask] = 0.0

    if absolute:
        pct = pct.abs()

    pct.index = window_df.index[1:]
    return pct
