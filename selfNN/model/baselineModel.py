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
BATCH_SIZE = hyperParamsDict["BATCH_SIZE"]
EPOCHS = hyperParamsDict["EPOCHS"]
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
labels = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])

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
            nn.Dropout(p=0.2),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits



def run_epochs(train_set: DataLoader, val_set: DataLoader, model: NeuralNetwork, epochs: int, device, loss_fn, optimizer):

    #-----------------Train variables----------------------------------------
    train_total_loss_per_epoch = 0
    train_avg_loss_per_epoch = 0
    train_loss_log = []
    train_accuracy_log = []
    train_total_correct_per_epoch = 0
    train_total_samples_seen_per_epoch = 0
    train_acc_per_epoch = 0
    #---------------------------------------------------------

    #-----------------Validation variables----------------------------------------
    val_total_loss_per_epoch = 0
    val_avg_loss_per_epoch = 0
    val_loss_log = []
    val_accuracy_log = []
    val_total_correct_per_epoch = 0
    val_total_samples_seen_per_epoch = 0
    val_acc_per_epoch = 0
    val_y_true = []
    val_y_pred = []
    val_best_acc = 0
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    #---------------------------------------------------------

#----------------------------Main loop-------------------------------------------------------------------
    

    for epoch in range(epochs):
        train_total_loss_per_epoch = 0
        train_total_correct_per_epoch = 0
        train_total_samples_seen_per_epoch = 0
        train_acc_per_epoch = 0

        print (f"------------------------------------Running epoch {epoch + 1}------------------------------------\n")

        #----------------------Train loop-------------------------------------------------------------------------
        phase = "Train"
        progress = tqdm(
            enumerate(train_set, start=1),
            total=len(train_set),
            desc=f"{phase} | Epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        model.train()

        for batch_idx, (X, y) in progress:
            batch_size = len(y)
            train_total_samples_seen_per_epoch += batch_size
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            pred = model(X)
            preds = pred.argmax(1)

            loss = loss_fn(pred, y)
            batch_loss = loss.item()
            train_total_loss_per_epoch += batch_loss
            train_avg_loss_per_epoch = train_total_loss_per_epoch / (batch_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_correct = (preds == y).type(torch.float).sum().item()
            train_total_correct_per_epoch += batch_correct
            train_acc_per_epoch = train_total_correct_per_epoch / train_total_samples_seen_per_epoch

            
            progress.set_postfix(
                batch=f"{batch_idx}/{len(train_set)}",
                loss=f"{batch_loss:.4f}",
                avg_loss=f"{train_avg_loss_per_epoch:.4f}",
                acc=f"{train_acc_per_epoch:.4f}",
            )

        accuracy = train_total_correct_per_epoch/train_total_samples_seen_per_epoch
        train_accuracy_log.append([epoch + 1, accuracy])
        train_avg_loss_per_epoch = train_total_loss_per_epoch / (len(train_set))
        train_loss_log.append([epoch + 1, train_avg_loss_per_epoch])

        print(f"[{phase} | Epoch {epoch + 1}/{epochs}] "
            f"Epoch accuracy: {accuracy:.4f} | Epoch loss: {train_avg_loss_per_epoch:.4f}")
        
        #-----------------------------------------------------------------------------------------------
        #--------------------------------Validation loop---------------------------------------------------------------

        val_total_loss_per_epoch = 0
        val_avg_loss_per_epoch = 0
        val_total_correct_per_epoch = 0
        val_total_samples_seen_per_epoch = 0
        val_acc_per_epoch = 0
        val_y_pred = []
        val_y_true = []

        model.eval()

        phase = "Validation"
        print (f"{phase} in progress...")
        progress = tqdm(
                    enumerate(val_set, start=1),
                    total=len(val_set),
                    desc=f"{phase} | Epoch {epoch + 1}/{epochs}",
                    leave=False,
                )

        for batch_idx, (X, y) in progress:
            batch_size = len(y)
            val_total_samples_seen_per_epoch += batch_size
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.no_grad():

                pred = model(X)
                preds = pred.argmax(1)

                loss = loss_fn(pred, y)
                batch_loss = loss.item()
                val_total_loss_per_epoch += batch_loss
                val_avg_loss_per_epoch = val_total_loss_per_epoch / (batch_idx)

                batch_correct = (preds == y).type(torch.float).sum().item()
                val_total_correct_per_epoch += batch_correct
                val_acc_per_epoch = val_total_correct_per_epoch / val_total_samples_seen_per_epoch


            val_y_true.extend(y.detach().cpu().tolist())
            val_y_pred.extend(preds.detach().cpu().tolist())

            progress.set_postfix(
                batch=f"{batch_idx}/{len(val_set)}",
                loss=f"{batch_loss:.4f}",
                avg_loss=f"{val_avg_loss_per_epoch:.4f}",
                acc=f"{val_acc_per_epoch:.4f}",
            )

        val_accuracy = val_total_correct_per_epoch / val_total_samples_seen_per_epoch
        val_accuracy_log.append([epoch + 1, val_accuracy])
        val_avg_loss_per_epoch = val_total_loss_per_epoch / (len(val_set))
        val_loss_log.append([epoch + 1, val_avg_loss_per_epoch])


        print(f"[{phase} | Epoch {epoch + 1}/{epochs}] "
            f"Epoch accuracy: {val_accuracy:.4f} | Epoch loss: {val_avg_loss_per_epoch:.4f}")

        
        if val_accuracy > val_best_acc:
            val_best_acc = val_accuracy
            print (f"New best validation accuracy: {val_best_acc:.4}")
            cm = confusion_matrix(y_true=val_y_true, y_pred=val_y_pred, labels=np.arange(len(labels)))
        #-----------------------------------------------------------------------------------------------

        print ("---------------------------------------------------------------------------------------\n")
                

    return train_loss_log, train_accuracy_log, val_loss_log, val_accuracy_log, cm, val_y_true, val_y_true




def main() -> None:

    print (f"Using device: {device}")

    

    dataset = ImgDataset()

    train_set, val_set = dataset.getTrainValSet(batch_size=BATCH_SIZE)

    model = NeuralNetwork().to(device=device)

    print (f"Model:\n {model.model}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                    hyperParamsDict["LR"], 
                                    weight_decay=hyperParamsDict["WEIGHT_DECAY"])

    train_loss_log, train_acc_log, val_loss_log, val_acc_log, cm, val_y_true, val_y_pred = run_epochs(train_set=train_set, val_set=val_set, model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, epochs=EPOCHS)

    print (cm)

    


if __name__ == "__main__":
    main()