import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
from core.paths import ROOT, SELF_METADATA_DIR, SELF_RESULTS_DIR, TRANSFER_METADATA_DIR, TRANSFER_RESULTS_DIR

ANALYSIS_DIR = ROOT / "analysis"

ARCHITECTURES = ("resnet18", "mobilenet_v2")
LABELS = {
    "baseline": "Baseline",
    "batch_norm": "BatchNorm",
    "wide": "Mais filtros",
    "gap": "GAP",
    "resnet18": "ResNet18",
    "mobilenet_v2": "MobileNetV2",
}


def _read_curve(csv_path: Path) -> dict[int, tuple[float, float]]:
    """Retorna {época: (loss, acurácia)}."""
    with open(csv_path, encoding="utf-8") as f:
        return {int(r["epoch"]): (float(r["loss"]), float(r["accuracy"])) for r in csv.DictReader(f)}


def _describe(family: str, experiment: str) -> tuple[str, str, str]:
    """Retorna (grupo, modelo, estratégia) a partir do nome da pasta do experimento."""
    if family == "self_nn":
        return "CNN própria", experiment, "-"
    for arch in ARCHITECTURES:
        if experiment.startswith(f"{arch}_"):
            return LABELS[arch], arch, experiment.removeprefix(f"{arch}_")
    return experiment, experiment, "-"


def label_of(run: dict) -> str:
    if run["family"] == "self_nn":
        return LABELS.get(run["model"], run["model"])
    return f"{LABELS.get(run['model'], run['model'])} ({run['strategy']})"


def collect_runs() -> list[dict]:
    runs = []
    sources = [("self_nn", SELF_METADATA_DIR, SELF_RESULTS_DIR), ("transfer_nn", TRANSFER_METADATA_DIR, TRANSFER_RESULTS_DIR)]
    for family, metadata_root, results_root in sources:
        for meta_path in sorted(metadata_root.glob("*/run*_metadata.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            experiment, run_name = meta_path.parent.name, meta["run"]
            csv_dir = results_root / experiment / run_name / "csv"
            if not (csv_dir / "Val_metrics.csv").exists():
                print(f"Ignorando {experiment}/{run_name}: CSVs não encontrados")
                continue

            train, val = _read_curve(csv_dir / "Train_metrics.csv"), _read_curve(csv_dir / "Val_metrics.csv")
            last_epoch = max(val)
            # Metadados antigos não têm best_epoch: recupera pela curva
            best_epoch = meta.get("best_epoch") or max(val, key=lambda e: val[e][1])
            group, model, strategy = _describe(family, experiment)
            epoch_times = meta.get("epoch_train_time_s") or []

            runs.append({
                "family": family, "group": group, "experiment": experiment, "run": run_name,
                "model": model, "strategy": strategy,
                "epochs": last_epoch,
                "lr": meta["hyperparameters"].get("LR"),
                "best_epoch": best_epoch,
                "val_acc": meta["best_val_accuracy"],
                "train_acc_at_best": train[best_epoch][1],
                # quanto maior a diferença de treino e validação, mais overfitting
                "gap_at_best": train[best_epoch][1] - val[best_epoch][1],
                "gap_at_end": train[last_epoch][1] - val[last_epoch][1],
                "min_val_loss_epoch": min(val, key=lambda e: val[e][0]),
                "total_params": meta.get("total_params"),
                "trainable_params": meta.get("trainable_params"),
                "train_time_min": meta["total_train_time_s"] / 60 if "total_train_time_s" in meta else None,
                "epoch_time_s": sum(epoch_times) / len(epoch_times) if epoch_times else None,
                "inference_ms_per_image": meta.get("val_inference_ms_per_image"),
                "device": meta.get("device"),
                "train_curve": train, "val_curve": val,
            })
    return runs


def sort_key(run: dict) -> tuple:
    """Ordem das tabelas: CNN própria, ResNet18, MobileNetV2; dentro de cada uma, a ordem de LABELS/estratégias."""
    group_order = ["CNN própria", "ResNet18", "MobileNetV2"]
    strategy_order = ["frozen", "partial", "full", "scratch"]
    model_order = list(LABELS)
    return (
        group_order.index(run["group"]) if run["group"] in group_order else 9,
        model_order.index(run["model"]) if run["model"] in model_order else 9,
        strategy_order.index(run["strategy"]) if run["strategy"] in strategy_order else 9,
        run["experiment"],
    )


def best_per_experiment(runs: list[dict]) -> list[dict]:
    best = {}
    for run in runs:
        key = (run["family"], run["experiment"])
        if key not in best or run["val_acc"] > best[key]["val_acc"]:
            best[key] = run
    return list(best.values())



# Tabelas


TABLE_COLUMNS = ["group", "experiment", "run", "strategy", "epochs", "lr", "best_epoch", "val_acc", "train_acc_at_best",
                 "gap_at_best", "gap_at_end", "min_val_loss_epoch", "total_params", "trainable_params",
                 "train_time_min", "epoch_time_s", "inference_ms_per_image", "device"]


def save_csv(runs: list[dict]) -> Path:
    path = ANALYSIS_DIR / "validation_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    return path


def _fmt(value, pattern: str) -> str:
    return "--" if value is None else pattern.format(value)


def _pt(text: str) -> str:
    """Formato brasileiro: 1,234.5 -> 1.234,5."""
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def save_latex(runs: list[dict]) -> Path:
    """Tabela pronta para colar no relatório (usa o pacote booktabs)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Resultados no conjunto de validação (melhor run de cada experimento).}",
        r"\label{tab:validacao}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Abordagem & Configuração & Acc. val. (\%) & Treino$-$val. (p.p.) & Parâmetros & Treináveis & Tempo (min) \\",
        r"\midrule",
    ]
    for group in dict.fromkeys(r["group"] for r in runs):
        for r in [r for r in runs if r["group"] == group]:
            config = LABELS.get(r["model"], r["model"]) if r["family"] == "self_nn" else r["strategy"]
            val_acc = _pt(f"{100 * r['val_acc']:.2f}")
            gap = _pt(f"{100 * r['gap_at_best']:.2f}")
            params = _pt(_fmt(r["total_params"], "{:,}"))
            trainable = _pt(_fmt(r["trainable_params"], "{:,}"))
            minutes = _pt(_fmt(r["train_time_min"], "{:.1f}"))
            lines.append(f"{group} & {config} & {val_acc} & {gap} & {params} & {trainable} & {minutes} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]

    path = ANALYSIS_DIR / "validation_summary.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path



# Gráficos


def _save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ANALYSIS_DIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(runs: list[dict], name: str, title: str) -> None:
    """Acurácia e loss de várias execuções sobrepostas: validação em linha cheia, treino tracejado."""
    if not runs:
        return
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, color in zip(runs, plt.cm.tab10.colors):
        label = label_of(run)
        for curve, style, suffix in ((run["val_curve"], "-", "val."), (run["train_curve"], "--", "treino")):
            epochs = sorted(curve)
            ax_acc.plot(epochs, [curve[e][1] for e in epochs], style, color=color, label=f"{label} – {suffix}")
            ax_loss.plot(epochs, [curve[e][0] for e in epochs], style, color=color)

    ax_acc.set(title=f"Acurácia – {title}", xlabel="Época", ylabel="Acurácia")
    ax_loss.set(title=f"Loss – {title}", xlabel="Época", ylabel="Loss")
    for ax in (ax_acc, ax_loss):
        ax.grid(True, linestyle="--", alpha=0.4)
    ax_acc.legend(fontsize=8)
    _save(fig, name)


def plot_complexity(runs: list[dict]) -> None:
    """Número de parâmetros (escala log) x acurácia de validação."""
    runs = [r for r in runs if r["total_params"]]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"CNN própria": "o", "ResNet18": "s", "MobileNetV2": "^"}
    for group in dict.fromkeys(r["group"] for r in runs):
        group_runs = [r for r in runs if r["group"] == group]
        ax.scatter([r["total_params"] for r in group_runs], [r["val_acc"] for r in group_runs],
                   marker=markers.get(group, "o"), s=60, label=group)
        for r in group_runs:
            text = LABELS.get(r["model"], r["model"]) if r["family"] == "self_nn" else r["strategy"]
            ax.annotate(text, (r["total_params"], r["val_acc"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set(title="Complexidade x desempenho", xlabel="Número total de parâmetros (escala log)", ylabel="Acurácia de validação")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    _save(fig, "complexity_vs_accuracy")


def main() -> None:
    all_runs = collect_runs()
    if not all_runs:
        raise SystemExit("Nenhum run encontrado. Treine os modelos com 'python -m train' primeiro.")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    best = sorted(best_per_experiment(all_runs), key=sort_key)

    save_csv(sorted(all_runs, key=lambda r: (*sort_key(r), r["run"])))
    save_latex(best)

    plot_curves([r for r in best if r["family"] == "self_nn"], "curves_self_nn", "CNN própria")
    for arch in ARCHITECTURES:
        plot_curves([r for r in best if r["model"] == arch], f"curves_{arch}", LABELS[arch])
    plot_complexity(best)

    devices = {r["device"] for r in best}
    if len(devices) > 1:
        print(f"Atenção: os runs foram treinados em dispositivos diferentes ({devices}); tempos não são comparáveis.")

    print(f"{'Abordagem':<12} {'Experimento':<22} {'Val':>7} {'Treino-val':>11} {'Params':>11} {'Tempo(min)':>11}")
    for r in best:
        print(f"{r['group']:<12} {r['experiment']:<22} {r['val_acc']:>7.4f} {r['gap_at_best']:>11.4f} "
              f"{_fmt(r['total_params'], '{:,}'):>11} {_fmt(r['train_time_min'], '{:.1f}'):>11}")
    print(f"\nTabelas e gráficos salvos em: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()