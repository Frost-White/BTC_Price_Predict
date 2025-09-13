import numpy as np
import torch
from torch.utils.data import Dataset

def create_sequences_multifeature_pctchange(data, window_size, pred_size, target_index):
    xs, ys = [], []
    for i in range(len(data) - window_size - pred_size + 1):
        x = data[i:i+window_size, :]                # shape: (window, feature)
        y = data[i+window_size:i+window_size+pred_size, target_index]  # hedef: Close'un değişimi
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class TimeseriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
