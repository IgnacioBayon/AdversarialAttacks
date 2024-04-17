from torch.utils.data import Dataset, DataLoader, random_split
from torch.jit import RecursiveScriptModule
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.rc('font', **{'sans-serif': 'Arial',
                         'family': 'sans-serif'})


def save_model(model, name):
    """Save the trained SkipGram model to a file, creating the directory if it does not exist.

    Args:
        model: The trained SkipGram model.
        model_path: The path to save the model file, including directory and filename.

    Returns:
        The path where the model was saved.
    """
    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cuda())
    model_scripted.save(f"models/{name}.pt")


def load_model(name: str) -> RecursiveScriptModule:
    """Load a PyTorch model from a file.

    Args:
        name (str): The name of the model file to load.

    Returns:
        The loaded PyTorch model.
    """
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def save_model(model: torch.nn.Module, path: str) -> None:
    """Saves the model to the given path

    Args:
        model (torch.nn.Module): The model to save
        path (str): Path to save the model
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """Loads the model from the given path

    Args:
        model (torch.nn.Module): The model to load
        path (str): Path to load the model

    Returns:
        torch.nn.Module: The loaded model
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    """
    Train a Pytorch model.

    Args:
        model (torch.nn.Module): Pytorch model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        epochs (int): The number of epochs to train the model.
        device (str): device where to train the model.
        learning_rate (float): The learning rate for the optimizer.
        print_every (int): Frequency of epochs to print training and test loss.
        patience (int): The number of epochs to wait for improvement on the test loss before stopping training early.
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        # Print an element of the train_loader
        for i, (reviews, labels) in enumerate(train_loader):
            reviews, labels = reviews.to(device), labels.to(device)
            reviews, labels = reviews.long(), labels.float()

            optimizer.zero_grad()

            output = model(reviews)
            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            # predictions = torch.round(output)
            predictions = (output > 0).int()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if i % 100 == 0:
                print(
                    f"Epoch: {epoch}, Step: {i}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.4f}%"
                )

        # Validation loop
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for reviews, labels in val_loader:
                reviews, labels = reviews.to(device), labels.to(device)
                reviews, labels = reviews.long(), labels.float()

                output = model(reviews)
                predictions = (output > 0).int()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            print(
                f"Epoch: {epoch}, Validation Accuracy: {100 * correct / total}%")


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, device: torch.device):
    """
    Evaluate a Pytorch model.

    Args:
        model (torch.nn.Module): Pytorch model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (str): device where to evaluate the model.

    Returns:
        float: The accuracy of the model on the test set.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for reviews, labels in test_loader:
            reviews, labels = reviews.to(device), labels.to(device)
            reviews, labels = reviews.long(), labels.float()

            output = model(reviews)
            predictions = (output > 0).int()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total
