import torch
from torch.utils.data import Dataset

class datasetMF(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data.iloc[idx]
        label_sample = self.labels.iloc[idx]

        data_tensor = torch.tensor(data_sample.tolist(), dtype=torch.float)
        label_tensor = torch.tensor(label_sample.tolist(), dtype=torch.float)
        return data_tensor, label_tensor