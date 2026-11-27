import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

NUM_CLASSES = 10


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _sync(device) -> None:
    # Garante que o tempo medido inclua o trabalho assíncrono da GPU
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def train_one_epoch(model: nn.Module, loader: DataLoader, device, loss_fn, optimizer, desc: str = "Train"):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0

    progress = tqdm(loader, desc=desc, leave=False)
    for X, y in progress:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / seen:.4f}")

    return total_loss / len(loader), correct / seen


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, loss_fn, desc: str = "Eval"):
    """Retorna loss média, acurácia, rótulos, predições e tempo de inferência (s)."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    y_true, y_pred = [], []

    _sync(device)
    start = time.perf_counter()
    for X, y in tqdm(loader, desc=desc, leave=False):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        logits = model(X)
        preds = logits.argmax(1)

        total_loss += loss_fn(logits, y).item()
        correct += (preds == y).sum().item()
        seen += len(y)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    _sync(device)
    elapsed = time.perf_counter() - start

    return total_loss / len(loader), correct / seen, y_true, y_pred, elapsed


def run_epochs(train_set: DataLoader, val_set: DataLoader, model: nn.Module, epochs: int, device, loss_fn, optimizer, scheduler=None) -> dict:
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epoch_train_time": [],
        "best_val_acc": 0.0, "best_epoch": 0,
        "val_y_true": [], "val_y_pred": [],
        "cm": np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int),
        "best_model": {},
    }

    for epoch in range(1, epochs + 1):
        print(f"------------------------------ Epoch {epoch}/{epochs} ------------------------------")

        _sync(device)
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_set, device, loss_fn, optimizer, desc=f"Train {epoch}/{epochs}")
        _sync(device)
        history["epoch_train_time"].append(time.perf_counter() - start)

        val_loss, val_acc, y_true, y_pred, _ = evaluate(model, val_set, device, loss_fn, desc=f"Val {epoch}/{epochs}")

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append([epoch, train_loss])
        history["train_acc"].append([epoch, train_acc])
        history["val_loss"].append([epoch, val_loss])
        history["val_acc"].append([epoch, val_acc])

        print(f"Train acc: {train_acc:.4f} | loss: {train_loss:.4f} | tempo: {history['epoch_train_time'][-1]:.1f}s")
        print(f"Val   acc: {val_acc:.4f} | loss: {val_loss:.4f}")

        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch
            history["val_y_true"] = y_true
            history["val_y_pred"] = y_pred
            history["cm"] = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
            history["best_model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"Novo melhor modelo (val acc {val_acc:.4f})")

    return history
