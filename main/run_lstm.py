import torch
import pandas as pd
import numpy as np
from model import LSTMRegressor


def run_lstm_inference(df: pd.DataFrame, model_path: str,
                       hidden: int = 512, layers: int = 2,
                       dropout: float = 0.0) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Checkpoint'i yükle ve eğitimdeki feature sırasını al ---
    ckpt = torch.load(model_path, map_location=device)

    # Eğitimde kaydedilen feature_names mutlaka olsun:
    ckpt_feature_names = ckpt.get("feature_names", None)
    if ckpt_feature_names is None:
        raise RuntimeError("Checkpoint içinde 'feature_names' yok; eğitim kodu onları yazmalıydı.")

    # --- Girdiyi eğitimdeki sıraya hizala ---
    missing = [c for c in ckpt_feature_names if c not in df.columns]
    if missing:
        raise RuntimeError(f"Inference df bu kolonları içermiyor: {missing}")

    # Fazla kolon varsa at:
    df_use = df.reindex(columns=ckpt_feature_names)

    # --- Tensöre çevir ---
    data = torch.tensor(df_use.to_numpy(dtype=np.float32), device=device).unsqueeze(0)  # (1, T, F)
    input_dim = data.shape[2]

    out_dim = 3
    output_names = ["close_ret", "high_delta", "low_delta"]

    # --- Modeli oluştur ve checkpoint ile aynı boyutlarda yükle ---
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        out_dim=out_dim
    ).to(device)

    # state_dict'i sıkı yükle
    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    with torch.no_grad():
        pred = model(data).cpu().numpy().squeeze(0)  # (3,)

    # --- Çıktıyı 3 kolonla döndür ---
    result = pd.DataFrame([pred], columns=output_names)
    return result
