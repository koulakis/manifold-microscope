"""Dataset utilities for the adapted beta-VAE implementation.

Copied and adapted from https://github.com/1Konny/Beta-VAE.
The upstream project is MIT licensed; see `LICENSE` and `NOTICE.md` in this
package for the original copyright and attribution.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from experiment_scripts.model_configs import BetaVAEConfig
from microscope.datasets.generic_dataset_loader import DatasetName, load_dataset_fixed_test_split


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args: BetaVAEConfig) -> DataLoader:
    dataset_name = DatasetName[args.dataset.lower()]
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    number_of_dims = args.number_of_dims
    training_ratio = args.training_ratio
    ratio_per_dim = args.ratio_per_dim
    noise_sigma = args.noise_sigma

    # Determine device for volume computation
    if args.device is not None:
        # Use explicit device setting
        if args.device == "mps" and torch.backends.mps.is_available():
            device = "mps"
        elif args.device.startswith("cuda") and torch.cuda.is_available():
            device = args.device
        elif args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        # Backward compatibility: use cuda flag
        if args.cuda and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    data_train, _, _, _ = load_dataset_fixed_test_split(
        datasets_dir=args.exported_datasets_dir,
        dataset_name=dataset_name,
        number_of_dims=number_of_dims,
        ratio_per_dim=ratio_per_dim,
        training_ratio=training_ratio,
        noise_sigma=noise_sigma,
        weight_subsampling_by_manifold_volume=True
    )

    data_train = (data_train - data_train.min()) / (data_train.max() - data_train.min())
    data_train = torch.from_numpy(data_train).unsqueeze(1).float()
    train_kwargs = {'data_tensor': data_train}
    dset = CustomTensorDataset

    train_data = dset(**train_kwargs)
    # Optimize DataLoader for MPS compatibility
    use_pin_memory = device.startswith("cuda")  # Only use pin_memory for CUDA
    use_num_workers = 0 if device == "mps" else num_workers  # MPS works better with single process

    print(f"[DEBUG] DataLoader device: {device}")
    print(f"[DEBUG] DataLoader num_workers: {use_num_workers}")
    print(f"[DEBUG] DataLoader pin_memory: {use_pin_memory}")

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=use_num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )

    return train_loader
