import torch
import torch.nn as nn

# --- TÜM MODEL PARAMETRELERİ BURADA ---
DEFAULT_INPUT_SIZE = 6
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_NUM_LAYERS = 5
DEFAULT_OUTPUT_SIZE = 10
DEFAULT_DROPOUT = 0.3

class LSTMForecast(nn.Module):
    def __init__(self,
                 input_size=DEFAULT_INPUT_SIZE,
                 hidden_size=DEFAULT_HIDDEN_SIZE,
                 num_layers=DEFAULT_NUM_LAYERS,
                 output_size=DEFAULT_OUTPUT_SIZE,
                 dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
