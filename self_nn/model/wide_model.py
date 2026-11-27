from torch import nn


class WideNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = 10) -> None:
        super(WideNetwork, self).__init__()
        # Igual ao baseline, com o dobro de filtros em todas as convoluções
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 40, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(40, 80, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(80, 80, 5),
            nn.ReLU(),
            nn.AvgPool2d(4, 4),
            nn.Flatten(),
            # 80 canais * 5 * 5 = 2000
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
