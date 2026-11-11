import torch
from torch import nn
hyperParamsDict = {
    "BATCH_SIZE" : 64,
    "EPOCHS" : 100,
    "IMAGE_SIZE": (224, 224),
    "LR" : 5e-4,
    "FOLDS" : 5,
    "WEIGHT_DECAY": 1e-4,
    "OPTIMIZER": "AdamW",
    "SEED": 13,
    "NUM_WORKERS": 2, 
    "LOSS_FUNC" : "CrossEntropy",
    "OPTIMIZER_NAME" : "SGD"
}