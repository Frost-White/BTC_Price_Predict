# make_windows.py
import pandas as pd

def make_reverse_windows(df: pd.DataFrame, window: int = 30, step: int = 1, order: str = "old_to_new"):
    """
    order: 'old_to_new' -> windows[0] en eski
           'new_to_old' -> windows[0] en yeni (mevcut davranış)
    """
    n = len(df)
    windows = []

    # mevcut geriye kayan üretim (en yeni -> en eski)
    for end in range(n, window, -step):
        start = end - window
        if start < 0:
            break
        win = df.iloc[start:end].copy()
        windows.append(win)

    if order == "old_to_new":
        windows.reverse()
    return windows
