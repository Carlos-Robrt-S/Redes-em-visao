import torch
import torchvision.datasets as dt
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from .hyperParam import hyperParamsDict

def getMeanStd():

    channelSum = torch.zeros(3)
    channelSqrdSum = torch.zeros(3)
    total_images = 0

    for img, label in getTrainset(transform_for_calculation):
        total_images += 1
        channelSum += torch.sum(img, dim=[1, 2])
        channelSqrdSum += torch.sum(img**2, dim=[1, 2])

    num_pixels = total_images*224*224
    new_mean = channelSum/num_pixels
    new_std = torch.sqrt((channelSqrdSum/num_pixels) - (new_mean**2))
    print (f"Média: {new_mean}\nDesvio: {new_std}")
    return new_mean, new_std

CUSTOM_MEAN = [0.4914, 0.4822, 0.4465]
CUSTOM_STD = [0.2454, 0.2419, 0.2602]

BATCH_SIZE = hyperParamsDict["BATCH_SIZE"]
NUM_WORKERS = hyperParamsDict["NUM_WORKERS"]


transform_for_calculation = v2.Compose([
    v2.ToImage(),
    v2.Resize(hyperParamsDict["IMAGE_SIZE"], interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToDtype(torch.float32, scale=True),
])

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(hyperParamsDict["IMAGE_SIZE"], interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=CUSTOM_MEAN,
                 std=CUSTOM_STD)
])


def getTrainset(transform: v2.Compose | None):
    return dt.CIFAR10("./dataset/data", train=True, transform=transform, download=True)

def getTestset(transform: v2.Compose | None):
    return dt.CIFAR10("./dataset/data", train=False, transform=transform, download=True)

def getTrainLoader():
    trainset = getTrainset(transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    return trainloader

def getTestLoader():
    testset = getTestset(transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)  
    return testloader