import torch
import numpy as np

from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Tuple


def train_model(
    epochs: int,
    model: Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    batch_size: int,
    criterion: Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Module:
    """Trains the model (RNN or LSTM)

    Args:
        epochs (int): Number of epochs to train the model
        model (Module): The model to train
        train_loader (DataLoader): DataLoader for the training set
        valid_loader (DataLoader): DataLoader for the validation set
        batch_size (int): Batch size
        criterion (Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on

    Returns:
        Module: The trained model

    """

    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    model.to(device)

    model.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            inputs, labels = inputs.to(device), labels.to(device)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, labels = inputs.to(device), labels.to(device)

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output, labels.float())

                    val_losses.append(val_loss.item())

                model.train()
                print(
                    "Epoch: {}/{}...".format(e + 1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)),
                )

    return model


def evaluate_model(
    model: Module,
    test_loader: DataLoader,
    batch_size: int,
    criterion: Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on the test set

    Args:
        model (Module): Model to evaluate
        test_loader (DataLoader): DataLoader for the test set
        batch_size (int): Batch size
        criterion (Module): Loss function
        device (torch.device): Device to evaluate on

    Returns:
        Tuple[float, float]: Test loss and accuracy
    """
    model.to(device)

    # Make test loop for multiclass classification

    test_losses = []  # track loss
    accuracies = []  # track accuracy

    # init hidden state
    h = model.init_hidden(batch_size)

    model.eval()
    # iterate over test data

    for inputs, labels in test_loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        inputs, labels = inputs.to(device), labels.to(device)

        # get predicted outputs
        output, h = model(inputs, h)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        prediction = torch.argmax(output, 1)

        # compare predictions to true label
        true_label = torch.argmax(labels, 1)

        accuracies.append(
            sum(prediction == true_label) / output.shape[0]
        )

    # accuracy over all test data
    test_acc = sum(accuracies) / len(accuracies)
    test_loss = np.mean(test_losses)

    return test_loss, test_acc
