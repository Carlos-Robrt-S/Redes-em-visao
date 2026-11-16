from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import torch
import torchvision.datasets as dt
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from .hyperParam import hyperParamsDict


def getMeanStd(dataset: ImgDataset):

    channelSum = torch.zeros(3)
    channelSqrdSum = torch.zeros(3)
    total_images = 0

    for img in dataset.data:
        total_images += 1
        channelSum += torch.sum(img, dim=[1, 2])
        channelSqrdSum += torch.sum(img**2, dim=[1, 2])

    num_pixels = total_images*224*224
    new_mean = channelSum/num_pixels
    new_std = torch.sqrt((channelSqrdSum/num_pixels) - (new_mean**2))
    print (f"Média: {new_mean}\nDesvio: {new_std}")
    return new_mean, new_std

#----------------------------------------------------------------------
CUSTOM_MEAN = [0.4914, 0.4822, 0.4465]
CUSTOM_STD = [0.2454, 0.2419, 0.2602]

BATCH_SIZE = hyperParamsDict["BATCH_SIZE"]
NUM_WORKERS = hyperParamsDict["NUM_WORKERS"]
#----------------------------------------------------------------------


transform_for_calculation = v2.Compose([
    v2.ToImage(),
    v2.Resize(hyperParamsDict["IMAGE_SIZE"], interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToDtype(torch.float32, scale=True),
])

transform1 = v2.Compose([
    v2.ToImage(),
    v2.Resize(hyperParamsDict["IMAGE_SIZE"], interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=CUSTOM_MEAN,
                 std=CUSTOM_STD),
])


class ImgDataset(dt.CIFAR10):
    def __init__(self, root: str | Path = "./dataset/data", train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        super().__init__(root, train, transform1, target_transform, download)
    
    def getLoader(self, batch_size: int, shuffle: bool, num_workers: int):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def getFolds(self, k_folds: int, batch_size: int, shuffle: bool, num_workers: int, seed: int):
        kfold = KFold(k_folds, shuffle=shuffle, random_state=seed)
        folds = []

        indxs = np.arange(len(self))

        for fold_idx, (train_idxs, val_idxs) in enumerate(kfold.split(indxs)):

            train_subset = Subset(self, list(train_idxs))
            val_subset = Subset(self, list(val_idxs))

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

            folds.append({
                "fold" : fold_idx + 1,
                "train_loader" : train_loader,
                "val_loader" : val_loader
            })

        return folds
