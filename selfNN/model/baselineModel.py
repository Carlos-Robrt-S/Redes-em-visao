import sys
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from tqdm import tqdm
from dataset.loader import ImgDataset
from dataset.hyperParam import hyperParamsDict
from sklearn.metrics import confusion_matrix

#------------------------------------------------------------------------------------------------
LR = hyperParamsDict["LR"]
WEIGHT_DECAY = hyperParamsDict["WEIGHT_DECAY"]
K = hyperParamsDict["FOLDS"]
# {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
classes = {
    0 : "airplane",
    1 : "automobile", 
    2 : "bird",
    3 : "cat",
    4 : "deer",
    5 : "dog",
    6 : "frog",
    7 : "horse",
    8 : "ship",
    9 : "truck"
}

device = "cuda" if torch.cuda.is_available() == True else "cpu"
#------------------------------------------------------------------------------------------------



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
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits



def run_epochs(dataloader: DataLoader, model: NeuralNetwork, device, loss_fn, optimizer, is_eval = False, epochs: int = 1):

    if is_eval:
        model.eval()
        grad_context = torch.no_grad()
        epochs = 1
    else:
        model.train()
        grad_context = torch.enable_grad()
        
    total_loss = 0
    loss_log = []
    accuracy_log = []
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []

    phase = "Val" if is_eval else "Train"

    for epoch in range(epochs):
        progress = tqdm(
            enumerate(dataloader, start=1),
            total=len(dataloader),
            desc=f"{phase} | Epoch {epoch + 1}/{epochs}",
            leave=False,
        )
        with grad_context:
            for batch_idx, (X, y) in progress:
                batch_size = len(y)
                total_samples += batch_size
                X, y = X.to(device), y.to(device)

                pred = model(X)
                preds = pred.argmax(1)

                loss = loss_fn(pred, y)
                batch_loss = loss.item()
                total_loss += batch_loss
                avg_loss_epoch = total_loss / (batch_idx + epoch*len(dataloader))

                if not is_eval:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                else:
                    y_pred.extend(preds.detach().cpu().tolist())
                    y_true.extend(y.tolist())

                batch_correct = (preds == y).type(torch.float).sum().item()
                batch_acc = batch_correct / batch_size
                total_correct += batch_correct
                avg_acc_epoch = total_correct / total_samples

                
                progress.set_postfix(
                    batch=f"{batch_idx}/{len(dataloader)}",
                    loss=f"{batch_loss:.4f}",
                    avg_loss=f"{avg_loss_epoch:.4f}",
                    acc=f"{batch_acc:.4f}",
                    avg_acc=f"{avg_acc_epoch:.4f}",
                )

        accuracy = total_correct/total_samples
        accuracy_log.append(accuracy)
        loss = total_loss / ((epoch+1)*len(dataloader))
        loss_log.append(loss)

        print(f"[{phase} | Epoch {epoch + 1}/{epochs}]"
            f"Accumulated accuracy: {accuracy:.4f} | Accumulated loss: {loss:.4f}")

    avrg_loss = total_loss/(len(dataloader) * epochs)
    final_acc = total_correct/total_samples

    if is_eval:
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    else:
        cm = None

    return loss_log, accuracy_log, avrg_loss, final_acc, cm


        


    """ else:
        model.eval()

        total_loss = 0
        loss_log = []
        accuracy_log = []
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X, y in dataloader:
                total_samples += len(y)
                X, y = X.to(device), y.to(device)

                pred = model(X)

                loss = loss_fn(pred, y)
                total_loss += loss.item()
                total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                loss_log.append(loss.item())
                accuracy = total_correct/total_samples
                accuracy_log.append(accuracy)

        avrg_loss = total_loss/len(dataloader)
        final_acc = total_correct/total_samples

        return loss_log, accuracy_log, avrg_loss, final_acc """


def main() -> None:

    print (f"Using device: {device}")

    

    dataset = ImgDataset()

    folds = dataset.getFolds(k_folds=hyperParamsDict["FOLDS"], 
                            batch_size=hyperParamsDict["BATCH_SIZE"], 
                            shuffle=True, 
                            num_workers=hyperParamsDict["NUM_WORKERS"], 
                            seed=hyperParamsDict["SEED"])


    for fold in folds:
        print (f"------------------------Fold {fold["fold"]}------------------------\n")

        model = NeuralNetwork().to(device=device)
        train_set = fold["train_loader"]
        val_set = fold["val_loader"]

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), 
                                     hyperParamsDict["LR"], 
                                     weight_decay=hyperParamsDict["WEIGHT_DECAY"])

        loss_train, acc_train, avrg_train_loss, final_train_acc, _ = run_epochs(train_set, model, device, loss_fn, optimizer, is_eval=False, epochs=hyperParamsDict["EPOCHS"])
        loss_val, acc_val, avrg_val_loss, final_val_acc, cm = run_epochs(val_set, model, device, loss_fn, optimizer, is_eval=True)

        print (cm)





































if __name__ == "__main__":
    main()