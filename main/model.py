from torch import nn
import torch

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int = 5, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.0, out_dim: int = 3):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden)  # 🔸 eklendi
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.layer_norm(out)  # 🔸 tüm zaman adımları normalize edilir
        last = out[:, -1, :]
        y_pred = self.fc(last)
        return y_pred
