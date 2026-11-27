from torch import nn


class GapNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = 10) -> None:
        super(GapNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 40, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(40, 40, 5),
            nn.ReLU(),
            # Global Average Pooling no lugar do classificador denso, que concentra ~92% dos parâmetros do baseline
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(40, classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
