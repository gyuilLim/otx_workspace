import torch
import torch.nn as nn
from tqdm import tqdm

class EarlyStoppingWithWarmup:
    def __init__(self, monitor="val_acc", mode="max", patience=5, warmup=5, verbose=True):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.warmup = warmup
        self.verbose = verbose

        self.wait = 0
        self.best = None
        self.stopped_epoch = 0
        self.stop_training = False
        self.epoch = 0

        if mode == "min":
            self.monitor_op = lambda current, best: current < best
            self.best = float('inf')
        elif mode == "max":
            self.monitor_op = lambda current, best: current > best
            self.best = -float('inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def step(self, current_value):
        self.epoch += 1

        if self.epoch <= self.warmup:
            if self.verbose:
                print(f"Epoch {self.epoch}: Warmup period ({self.warmup} epochs) - skipping early stopping check.")
            return

        if self.best is None or self.monitor_op(current_value, self.best):
            self.best = current_value
            self.wait = 0
            if self.verbose:
                print(f"Epoch {self.epoch}: {self.monitor} improved to {current_value:.4f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"Epoch {self.epoch}: {self.monitor} did not improve. Wait {self.wait}/{self.patience}")

            if self.wait >= self.patience:
                self.stop_training = True
                self.stopped_epoch = self.epoch
                if self.verbose:
                    print(f"Early stopping triggered at epoch {self.stopped_epoch}.")


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    acc = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    acc = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), acc