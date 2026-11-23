from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import torch
import torchvision.datasets as dt
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from .hyperParam import hyperParamsDict
from collections import Counter


def getMeanStd(dataset: ImgDataset):
    channelSum = torch.zeros(3, dtype=torch.float64)
    channelSqrdSum = torch.zeros(3, dtype=torch.float64)
    total_images = 0

    for img, _ in dataset:
        total_images += 1
        # img tem formato (C, H, W) em float32 [0.0, 1.0]
        channelSum += img.to(torch.float64).sum(dim=(1, 2))
        channelSqrdSum += (img.to(torch.float64) ** 2).sum(dim=(1, 2))

    _, h, w = dataset[0][0].shape
    num_pixels = total_images * h * w

    new_mean = (channelSum / num_pixels).to(torch.float32)
    variance = (channelSqrdSum / num_pixels) - ((channelSum / num_pixels) ** 2)
    new_std = torch.sqrt(torch.clamp(variance, min=0.0)).to(torch.float32)

    print(f"Média: {new_mean}\nDesvio: {new_std}")
    return new_mean, new_std

#----------------------------------------------------------------------
CUSTOM_MEAN = [0.4914, 0.4822, 0.4465]
CUSTOM_STD = [0.2454, 0.2419, 0.2602]

BATCH_SIZE = hyperParamsDict["BATCH_SIZE"]
NUM_WORKERS = hyperParamsDict["NUM_WORKERS"]
SEED = hyperParamsDict["SEED"]
#----------------------------------------------------------------------


transform_for_calculation = v2.Compose([
    v2.ToImage(),
    v2.Resize(hyperParamsDict["IMAGE_SIZE"], interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToDtype(torch.float32, scale=True),
])

transform224x224 = v2.Compose([
    v2.ToImage(),
    v2.Resize(hyperParamsDict["IMAGE_SIZE"], interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=CUSTOM_MEAN,
                 std=CUSTOM_STD),
])


class ImgDataset(dt.CIFAR10):
    def __init__(self, root: str | Path = "./dataset/data", train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        super().__init__(root, train, transform224x224, target_transform, download)

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

    def getTrainValSet(self, batch_size: int, shuffle: bool = True, num_workers: int = NUM_WORKERS, seed: int = SEED):
        targets = np.array(self.targets)
        indices = np.arange(len(self.data))
        train_ind, val_ind = train_test_split(indices, 
                                              test_size=0.2, 
                                              train_size=0.8, 
                                              random_state=seed, 
                                              shuffle=shuffle, 
                                              stratify=targets)

        train_dataset = Subset(self, train_ind)
        val_dataset = Subset(self, val_ind)

        persistent = num_workers > 0

        train_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, num_workers=num_workers, 
                                  pin_memory=True, 
                                  persistent_workers=persistent)
        
        val_loader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=num_workers, 
                                pin_memory=True, 
                                persistent_workers=persistent)

        return train_loader, val_loader

    def checkDistribution(self):
        train_set, val_set = self.getTrainValSet(batch_size=BATCH_SIZE, num_workers=0)

        ys_train = []
        ys_val = []

        for _, y in train_set:
            ys_train.extend(y.tolist())

        for _, y in val_set:
            ys_val.extend(y.tolist())

        print (f"Train size: {len(ys_train)} || Ammount per classes: {np.bincount(ys_train).tolist()}")
        print (f"val size: {len(ys_val)} || Ammount per classes: {np.bincount(ys_val).tolist()}")
        """
        Print: 
        Train size: 40000 || Ammount per classes: [4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000]
        val size: 10000 || Ammount per classes: [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        """

