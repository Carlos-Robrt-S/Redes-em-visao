import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from core.trainer import evaluate, count_parameters
from core.file_plot import CLASS_NAMES, plot_confusion_matrix, save_summary_log
from core.paths import SELF_METADATA_DIR, SELF_RESULTS_DIR, TRANSFER_METADATA_DIR, TRANSFER_RESULTS_DIR, TEST_RESULTS_DIR
from core.reproducibility import seed_everything
from dataset.hyper_param import TRANSFER_HYPER_PARAMS
from dataset.loader import ImgDataset
from train import SELF_MODELS, build_experiment, device
from transfer_nn.model.transfer_models import ARCHITECTURES, STRATEGIES

GROUP_LABELS = {"self_nn": "CNN própria", "resnet18": "ResNet18", "mobilenet_v2": "MobileNetV2"}


def _parse_experiment(experiment: str) -> tuple[str, str, str | None]:
    if experiment in SELF_MODELS:
        return "self_nn", experiment, None
    for arch in ARCHITECTURES:
        strategy = experiment.removeprefix(f"{arch}_")
        if experiment.startswith(f"{arch}_") and strategy in STRATEGIES:
            return arch, arch, strategy
    raise ValueError(f"Experimento desconhecido: {experiment}")


def collect_runs() -> list[dict]:
    runs = []
    for metadata_root, results_root in [(SELF_METADATA_DIR, SELF_RESULTS_DIR), (TRANSFER_METADATA_DIR, TRANSFER_RESULTS_DIR)]:
        for meta_path in sorted(metadata_root.glob("*/run*_metadata.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            experiment = meta_path.parent.name
            run = meta["run"]
            try:
                group, model_name, strategy = _parse_experiment(experiment)
            except ValueError as err:
                print(f"Ignorando {meta_path}: {err}")
                continue

            checkpoint = results_root / experiment / run / "best_model.pt"
            if not checkpoint.exists():
                print(f"Ignorando {experiment}/{run}: checkpoint não encontrado")
                continue

            runs.append({
                "group": group, "experiment": experiment, "run": run,
                "model_name": model_name, "strategy": strategy,
                "checkpoint": checkpoint, "meta": meta,
                "val_acc": meta["best_val_accuracy"],
            })
    return runs


def select_best(runs: list[dict]) -> list[dict]:
    best = {}
    for run in runs:
        if run["group"] not in best or run["val_acc"] > best[run["group"]]["val_acc"]:
            best[run["group"]] = run
    return [best[g] for g in GROUP_LABELS if g in best]


def evaluate_on_test(run: dict) -> dict:
    exp = build_experiment(run["model_name"], run["strategy"], epochs=None, lr=None)
    model = exp.model
    model.load_state_dict(torch.load(run["checkpoint"], map_location="cpu"))
    model.to(device)

    test_set = ImgDataset.get_test_set(batch_size=exp.hyperparams["BATCH_SIZE"], transform=exp.transform)
    loss, acc, y_true, y_pred, inference_time = evaluate(model, test_set, device, nn.CrossEntropyLoss(), desc=f"Teste {run['experiment']}")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    total_params, trainable_params = count_parameters(model)

    out_dir = TEST_RESULTS_DIR / f"{run['group']}__{run['experiment']}_{run['run']}"
    title = f"Teste - {GROUP_LABELS[run['group']]} ({run['experiment']})"
    for ext in ("png", "pdf"):
        plot_confusion_matrix(cm, output_dir=out_dir, filename=f"confusion_matrix.{ext}", title=title)
    save_summary_log([[1, loss]], [[1, acc]], y_true, y_pred, phase="Test", output_dir=out_dir)

    return {
        "approach": GROUP_LABELS[run["group"]],
        "experiment": run["experiment"],
        "run": run["run"],
        "val_acc": run["val_acc"],
        "test_acc": acc,
        "test_loss": loss,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_time_s": run["meta"].get("total_train_time_s", ""),
        "test_inference_ms_per_image": 1000 * inference_time / len(test_set.dataset),
        "per_class_acc": (cm.diagonal() / cm.sum(axis=1)).tolist(),
    }


def save_comparison(results: list[dict]) -> None:
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    columns = ["approach", "experiment", "run", "val_acc", "test_acc", "test_loss",
               "total_params", "trainable_params", "train_time_s", "test_inference_ms_per_image"]
    with open(TEST_RESULTS_DIR / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    with open(TEST_RESULTS_DIR / "per_class_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", *[r["approach"] for r in results]])
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name, *[f"{r['per_class_acc'][i]:.4f}" for r in results]])

    names = [r["approach"] for r in results]
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(results)

    fig, (ax_total, ax_class) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 3]})
    bars = ax_total.bar(names, [r["test_acc"] for r in results], color="tab:blue")
    ax_total.bar_label(bars, fmt="%.4f")
    ax_total.set_ylim(0, 1)
    ax_total.set_title("Acurácia no teste")

    for i, r in enumerate(results):
        ax_class.bar(x + i * width, r["per_class_acc"], width, label=r["approach"])
    ax_class.set_xticks(x + width * (len(results) - 1) / 2, CLASS_NAMES, rotation=45)
    ax_class.set_ylim(0, 1)
    ax_class.set_title("Acurácia por classe no teste")
    ax_class.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(TEST_RESULTS_DIR / f"comparison.{ext}", dpi=150)
    plt.close(fig)


def main() -> None:
    seed_everything(TRANSFER_HYPER_PARAMS["SEED"])
    runs = collect_runs()
    if not runs:
        raise SystemExit("Nenhum run encontrado. Treine os modelos com 'python -m train' primeiro.")

    print("Runs encontrados (acurácia de validação):")
    for run in sorted(runs, key=lambda r: (r["group"], -r["val_acc"])):
        print(f"  {GROUP_LABELS[run['group']]:<12} {run['experiment']:<22} {run['run']:<6} {run['val_acc']:.4f}")

    selected = select_best(runs)
    print("\nSelecionados para o teste:")
    for run in selected:
        print(f"  {GROUP_LABELS[run['group']]:<12} {run['experiment']} / {run['run']}")

    results = [evaluate_on_test(run) for run in selected]
    save_comparison(results)

    print(f"\n{'Abordagem':<12} {'Experimento':<22} {'Val':>7} {'Teste':>7} {'Params':>12} {'ms/img':>7}")
    for r in results:
        print(f"{r['approach']:<12} {r['experiment']:<22} {r['val_acc']:>7.4f} {r['test_acc']:>7.4f} "
              f"{r['total_params']:>12,} {r['test_inference_ms_per_image']:>7.3f}")
    print(f"\nResultados salvos em: {TEST_RESULTS_DIR}")


if __name__ == "__main__":
    main()