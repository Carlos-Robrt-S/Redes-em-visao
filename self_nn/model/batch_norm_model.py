from torch import nn


class BatchNormNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = 10) -> None:
        super(BatchNormNetwork, self).__init__()
        # Igual ao baseline, com BatchNorm2d após cada convolução (antes do ReLU)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 10, 3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 20, 3),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 40, 3),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(40, 40, 5),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.AvgPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
