from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = 10) -> None:
        super(NeuralNetwork, self).__init__()
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
