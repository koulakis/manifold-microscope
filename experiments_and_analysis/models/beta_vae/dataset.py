"""Dataset utilities for the adapted beta-VAE implementation.

Copied and adapted from https://github.com/1Konny/Beta-VAE.
The upstream project is MIT licensed; see `LICENSE` and `NOTICE.md` in this
package for the original copyright and attribution.
"""
import torch
from torch.utils.data import Dataset, DataLoader



def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)



def training_loader(data, batch_size=64, num_workers=0, device="cpu"):
    if data.shape[1:] != (64, 64):
        raise ValueError("beta-VAE requires grayscale images of shape (64, 64).")
    images = torch.from_numpy(data).unsqueeze(1).float()
    return DataLoader(CustomTensorDataset(images), batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=device.startswith("cuda"))
