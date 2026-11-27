import csv
import json
import os
from datetime import datetime
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support
from core.paths import SELF_RESULTS_DIR, SELF_METADATA_DIR

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def write_metrics_csv(
    loss_log: list[list[float]],
    acc_log: list[list[float]],
    phase: str,
    output_dir: str | Path = SELF_RESULTS_DIR / "CSVs",
    filename: str | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / (filename or f"{phase}_metrics.csv")
    file_exists = csv_path.exists()

    loss_by_epoch = {int(ep): float(val) for ep, val in loss_log}
    acc_by_epoch = {int(ep): float(val) for ep, val in acc_log}
    epochs = sorted(set(loss_by_epoch.keys()) | set(acc_by_epoch.keys()))

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "phase", "loss", "accuracy"])

        for ep in epochs:
            writer.writerow([
                ep,
                phase,
                loss_by_epoch.get(ep, ""),
                acc_by_epoch.get(ep, ""),
            ])

    return csv_path


def plot_loss_and_accuracy(
    train_loss_log: list[list[float]],
    train_acc_log: list[list[float]],
    val_loss_log: list[list[float]],
    val_acc_log: list[list[float]],
    output_dir: str | Path = SELF_RESULTS_DIR / "plots",
    filename: str = "loss_accuracy_curves.png",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename

    tr_loss_ep = [x[0] for x in train_loss_log]
    tr_loss_val = [x[1] for x in train_loss_log]
    tr_acc_ep = [x[0] for x in train_acc_log]
    tr_acc_val = [x[1] for x in train_acc_log]

    val_loss_ep = [x[0] for x in val_loss_log]
    val_loss_val = [x[1] for x in val_loss_log]
    val_acc_ep = [x[0] for x in val_acc_log]
    val_acc_val = [x[1] for x in val_acc_log]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(tr_loss_ep, tr_loss_val, marker="o", markersize=3, label="Train Loss")
    ax_loss.plot(val_loss_ep, val_loss_val, marker="s", markersize=3, label="Val Loss")
    ax_loss.set_title("Loss por Época")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend()

    ax_acc.plot(tr_acc_ep, tr_acc_val, marker="o", markersize=3, label="Train Accuracy")
    ax_acc.plot(val_acc_ep, val_acc_val, marker="s", markersize=3, label="Val Accuracy")
    ax_acc.set_title("Acurácia por Época")
    ax_acc.set_xlabel("Época")
    ax_acc.set_ylabel("Acurácia")
    ax_acc.grid(True, linestyle="--", alpha=0.4)
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | np.ndarray = CLASS_NAMES,
    output_dir: str | Path = SELF_RESULTS_DIR / "plots",
    filename: str = "confusion_matrix.png",
    title: str = "Matriz de Confusão",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename

    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d" if np.issubdtype(cm.dtype, np.integer) else ".2f")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def save_summary_log(
    loss_log: list[list[float]],
    acc_log: list[list[float]],
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str] | np.ndarray = CLASS_NAMES,
    phase: str = "validation",
    output_dir: str | Path = SELF_RESULTS_DIR / "logs",
    filename: str | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / (filename or f"{phase}_summary.log")

    mean_loss = float(np.mean([x[1] for x in loss_log])) if loss_log else float("nan")
    mean_acc = float(np.mean([x[1] for x in acc_log])) if acc_log else float("nan")

    labels_idx = np.arange(len(class_names))
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average=None, zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="weighted", zero_division=0
    )

    lines = [
        f"=== Resumo de Métricas ({phase}) ===",
        f"Média Loss (épocas):     {mean_loss:.4f}",
        f"Média Acurácia (épocas): {mean_acc:.4f}",
        "",
        "--- Métricas Globais ---",
        f"Precision (macro):    {prec_macro:.4f} | (weighted): {prec_weighted:.4f}",
        f"Recall    (macro):    {rec_macro:.4f} | (weighted): {rec_weighted:.4f}",
        f"F1-Score  (macro):    {f1_macro:.4f} | (weighted): {f1_weighted:.4f}",
        "",
        "--- Métricas por Classe ---",
        f"{'Classe':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Suporte':>8}",
    ]

    prec_per_class = np.array(prec_per_class)
    rec_per_class = np.array(rec_per_class)
    f1_per_class = np.array(f1_per_class)
    support = np.array(support)

    for idx, name in enumerate(class_names):
        lines.append(
            f"{str(name):<12} "
            f"{prec_per_class[idx]:>10.4f} "
            f"{rec_per_class[idx]:>10.4f} "
            f"{f1_per_class[idx]:>10.4f} "
            f"{int(support[idx]):>8d}"
        )

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path



def save_run_metadata(
    run_number: int,
    hyperparams: dict,
    model_arch: str,
    optimizer_name: str,
    loss_name: str,
    device: str,
    best_val_acc: float,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path = SELF_METADATA_DIR,
    extra_info: dict | None = None,
) -> Path:
    meta_dir = Path(output_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    meta_path = meta_dir / f"run{run_number}_metadata.json"

    serializable_hyperparams = {}
    for k, v in hyperparams.items():
        if isinstance(v, torch.Tensor):
            serializable_hyperparams[k] = v.tolist()
        elif isinstance(v, np.ndarray):
            serializable_hyperparams[k] = v.tolist()
        else:
            serializable_hyperparams[k] = v

    metadata = {
        "run": f"run{run_number}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "hyperparameters": serializable_hyperparams,
        "optimizer": optimizer_name,
        "loss_function": loss_name,
        "best_val_accuracy": float(best_val_acc),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "architecture": model_arch,
    }
    if extra_info:
        metadata.update(extra_info)

    meta_path.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return meta_path


def save_metrics(    
    train_loss_log: list[list[float]],
    train_acc_log: list[list[float]],
    val_loss_log: list[list[float]],
    val_acc_log: list[list[float]],
    cm: np.ndarray,
    val_y_true: list,
    val_y_pred: list,
    best_model,
    base_dir: str | Path = SELF_RESULTS_DIR) -> int:

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = [
        int(p.name.removeprefix("run"))
        for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("run") and p.name.removeprefix("run").isdigit()
    ]
    run_nmbr = max(existing_runs, default=0) + 1
    out_dir = base_dir / f"run{run_nmbr}"
    out_dir.mkdir(parents=True, exist_ok=True)

    path_csv_train = write_metrics_csv(output_dir=f"{out_dir}/CSVs", loss_log=train_loss_log, acc_log=train_acc_log, phase="Train")
    path_csv_val = write_metrics_csv(output_dir=f"{out_dir}/CSVs", loss_log=val_loss_log, acc_log=val_acc_log, phase="Val")

    path_graphic = plot_loss_and_accuracy(output_dir=f"{out_dir}/graphics",
                            train_loss_log=train_loss_log, 
                            train_acc_log=train_acc_log,
                            val_loss_log=val_loss_log,
                            val_acc_log=val_acc_log)

    path_cm = plot_confusion_matrix(output_dir=f"{out_dir}/confusion",
                          cm=cm,
                          title="ConfusionMatrix")

    path_sum = save_summary_log(output_dir=f"{out_dir}/metricsLog",
                     loss_log=val_loss_log,
                     acc_log=val_acc_log,
                     y_true=val_y_true,
                     y_pred=val_y_pred,
                     phase="Val")

    torch.save(best_model, out_dir / "best_model.pt")

    print(f"CSV Train saved in: {path_csv_train}\nCSV Val saved in: {path_csv_val}\nGráfico saved in: {path_graphic}\nConfusion matrix saved in:{path_cm}\nMetrics log saved in:{path_sum}")

    return run_nmbr