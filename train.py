"""
Ponto de entrada único para treinar qualquer modelo (na raiz do projeto):
    python -m train --model baseline
    python -m train --model resnet18 --strategy frozen
    python -m train --model mobilenet_v2 --strategy full --epochs 10
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torchvision.transforms import v2
from core.trainer import run_epochs, evaluate, count_parameters
from core.file_plot import save_metrics, save_run_metadata
from core.reproducibility import seed_everything
from core.paths import SELF_RESULTS_DIR, SELF_METADATA_DIR, TRANSFER_RESULTS_DIR, TRANSFER_METADATA_DIR
from dataset.loader import ImgDataset, transform_cifar_224, transform_imagenet_224
from dataset.hyper_param import HYPER_PARAMS, TRANSFER_HYPER_PARAMS
from self_nn.model.baseline_model import NeuralNetwork
from transfer_nn.model.transfer_models import ARCHITECTURES, STRATEGIES, build_transfer_model

# Registro das CNNs próprias: para um novo experimento da etapa 1, crie a classe e adicione aqui
SELF_MODELS = {
    "baseline": NeuralNetwork,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Experiment:
    name: str
    model: nn.Module
    transform: v2.Compose
    hyperparams: dict
    results_dir: Path
    metadata_dir: Path


def build_experiment(model_name: str, strategy: str | None, epochs: int | None, lr: float | None) -> Experiment:
    if model_name in SELF_MODELS:
        hp = dict(HYPER_PARAMS)
        model = SELF_MODELS[model_name]()
        name, transform = model_name, transform_cifar_224
        results_dir, metadata_dir = SELF_RESULTS_DIR / name, SELF_METADATA_DIR / name
    else:
        if strategy is None:
            raise SystemExit(f"--strategy é obrigatório para {model_name}. Opções: {STRATEGIES}")
        hp = dict(TRANSFER_HYPER_PARAMS)
        hp["LR"] = hp["LR_BY_STRATEGY"][strategy]
        model = build_transfer_model(model_name, strategy)
        name = f"{model_name}_{strategy}"
        # Pesos do ImageNet esperam a normalização do ImageNet; do zero usa a do CIFAR (como o baseline)
        transform = transform_cifar_224 if strategy == "scratch" else transform_imagenet_224
        results_dir, metadata_dir = TRANSFER_RESULTS_DIR / name, TRANSFER_METADATA_DIR / name

    if epochs is not None:
        hp["EPOCHS"] = epochs
    if lr is not None:
        hp["LR"] = lr

    return Experiment(name, model, transform, hp, results_dir, metadata_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Treino no CIFAR-10")
    parser.add_argument("--model", choices=[*SELF_MODELS, *ARCHITECTURES], required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, default=None, help="Só para modelos pré-treinados")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescreve o número de épocas padrão")
    parser.add_argument("--lr", type=float, default=None, help="Sobrescreve o LR padrão")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp = build_experiment(args.model, args.strategy, args.epochs, args.lr)
    hp = exp.hyperparams

    seed_everything(hp["SEED"])

    print(f"Device: {device} | Experimento: {exp.name} | LR: {hp['LR']} | Épocas: {hp['EPOCHS']}")

    train_set, val_set = ImgDataset(transform=exp.transform).get_train_val_set(batch_size=hp["BATCH_SIZE"])

    model = exp.model.to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Parâmetros: {total_params:,} total | {trainable_params:,} treináveis")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hp["LR"],
        weight_decay=hp["WEIGHT_DECAY"],
    )

    history = run_epochs(train_set, val_set, model, hp["EPOCHS"], device, loss_fn, optimizer)

    # Tempo de inferência medido com o melhor modelo sobre as 5.000 imagens de validação
    model.load_state_dict(history["best_model"])
    *_, val_inference_time = evaluate(model, val_set, device, loss_fn, desc="Inferência")

    run_number = save_metrics(history, base_dir=exp.results_dir)

    save_run_metadata(
        run_number=run_number,
        hyperparams=hp,
        model_arch=str(model),
        optimizer_name=optimizer.__class__.__name__,
        loss_name=loss_fn.__class__.__name__,
        device=device,
        best_val_acc=history["best_val_acc"],
        checkpoint_path=exp.results_dir / f"run{run_number}" / "best_model.pt",
        output_dir=exp.metadata_dir,
        extra_info={
            "experiment": exp.name,
            "model": args.model,
            "strategy": args.strategy,
            "best_epoch": history["best_epoch"],
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_train_time_s": sum(history["epoch_train_time"]),
            "epoch_train_time_s": history["epoch_train_time"],
            "val_inference_time_s": val_inference_time,
            "val_inference_ms_per_image": 1000 * val_inference_time / len(val_set.dataset),
        },
    )

    print(f"Melhor val acc: {history['best_val_acc']:.4f} (época {history['best_epoch']})")


if __name__ == "__main__":
    main()
