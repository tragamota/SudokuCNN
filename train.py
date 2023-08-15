import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import preprocessing
from model import SudokuCNN
from sudoku_dataset import SudokuDataset


def train(model, loader, optimizer, criterion, epoch):
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    losses = []
    accuracies = []

    model.train()

    for inputs, labels in tqdm(loader):
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)

            outputs_argmax = torch.argmax(outputs, dim=3) + 1
            labels_argmax = torch.argmax(labels, dim=3) + 1

            loss = criterion(outputs, labels)

        losses.append(loss.item())

        accuracy = (outputs_argmax == labels_argmax).float().mean()
        accuracies.append(accuracy.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(accuracies)

    print("\n\n" + "=" * 20)

    print("\nEpoch {}: Loss {:.4f}, Accuracy {:.4f} \n".format(
        epoch + 1, mean_loss, mean_accuracy))

    print("=" * 20 + "\n\n")

    return model


def test(model, loader, epoch):
    correct = 0
    total = 0

    accuracies = []

    model.eval()

    for inputs, labels in tqdm(loader):
        inputs = inputs.cuda()

        outputs = model(inputs)

        outputs = torch.argmax(outputs, dim=3)
        labels = torch.argmax(labels.cuda(), dim=3)

        results = zip(*(outputs, labels))

        accuracy = (outputs == labels).float().mean()
        accuracies.append(accuracy.item())

        for output, label in results:
            if torch.eq(output, label).all():
                correct += 1

            total += 1

    print("\n\n" + "=" * 20)
    print("\nVALIDATION: Epoch {}: Sudoku Accuracy {:.4f} Digit Accuracy {:.4f} \n".format(
        epoch + 1, correct / total, np.mean(accuracies)))
    print("=" * 20 + "\n\n")


if __name__ == "__main__":
    train_X, test_X, train_Y, test_Y = preprocessing.preprocess_sudoku('./data/sudoku.csv', 1_000_000)

    train_dataset = SudokuDataset(train_X, train_Y)
    test_dataset = SudokuDataset(test_X, test_Y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64, num_workers=2)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=2)

    model = SudokuCNN().cuda()

    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(10):
        model = train(model, train_loader, optimizer, criterion, epoch)
        test(model, test_loader, epoch)

    torch.save(model.state_dict(), "sudoku_CNN.pth")
