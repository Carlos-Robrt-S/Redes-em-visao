from torch import nn
from torchvision import models
from core.trainer import count_parameters

ARCHITECTURES = ("resnet18", "mobilenet_v2")
STRATEGIES = ("frozen", "partial", "full", "scratch")


class TransferNet(nn.Module):
    def __init__(self, backbone: nn.Module, arch: str, strategy: str) -> None:
        super().__init__()
        self.model = backbone
        self.arch = arch
        self.strategy = strategy

    def forward(self, x):
        return self.model(x)

    def train(self, mode: bool = True):
        super().train(mode)
        # BatchNorm congelada continua atualizando running_mean/var em train(); mantém em eval
        if mode:
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
                    module.eval()
        return self


def _build_resnet18(num_classes: int, pretrained: bool):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net, net.fc, [net.layer4, net.fc]


def _build_mobilenet_v2(num_classes: int, pretrained: bool):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=weights)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    return net, net.classifier, [net.features[14:], net.classifier]


_BUILDERS = {
    "resnet18": _build_resnet18,
    "mobilenet_v2": _build_mobilenet_v2,
}


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def build_transfer_model(arch: str, strategy: str, num_classes: int = 10) -> TransferNet:
    if arch not in _BUILDERS:
        raise ValueError(f"Arquitetura desconhecida: {arch}. Opções: {ARCHITECTURES}")
    if strategy not in STRATEGIES:
        raise ValueError(f"Estratégia desconhecida: {strategy}. Opções: {STRATEGIES}")

    net, head, partial_blocks = _BUILDERS[arch](num_classes, pretrained=strategy != "scratch")

    if strategy == "frozen":
        _set_trainable(net, False)
        _set_trainable(head, True)
    elif strategy == "partial":
        _set_trainable(net, False)
        for block in partial_blocks:
            _set_trainable(block, True)
    else:  # full / scratch
        _set_trainable(net, True)

    return TransferNet(net, arch, strategy)


if __name__ == "__main__":
    print(f"{'Arquitetura':<14} {'Estratégia':<10} {'Total':>12} {'Treináveis':>12} {'%':>7}")
    for arch in ARCHITECTURES:
        for strategy in STRATEGIES:
            total, trainable = count_parameters(build_transfer_model(arch, strategy))
            print(f"{arch:<14} {strategy:<10} {total:>12,} {trainable:>12,} {100 * trainable / total:>6.1f}%")
