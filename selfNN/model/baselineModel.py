import sys
from pathlib import Path
from typing import Any
import torch
from torch import nn
import torchvision


# Adiciona a raiz do projeto ('Redes-em-visao') ao caminho de busca do Python
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import dataset.loader as loader
from dataset.hyperParam import hyperParamsDict

#------------------------------------------------------------------------------------------------
LR = hyperParamsDict["LR"]
WEIGHT_DECAY = hyperParamsDict["WEIGHT_DECAY"]
#------------------------------------------------------------------------------------------------


device = "cuda" if torch.cuda.is_available() == True else "cpu"
print (device)

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            
        )

    def forward(self, x):
        logits = self.model(x)
        return logits