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
    train_on_gpu: bool,
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
        train_on_gpu (bool): Whether to train on GPU

    Returns:
        Module: The trained model

    """

    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    if train_on_gpu:
        model.cuda()

    model.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
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

                    if train_on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

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
    train_on_gpu: bool,
) -> Tuple[float, float]:
    """Evaluate the model on the test set

    Args:
        model (Module): Model to evaluate
        test_loader (DataLoader): DataLoader for the test set
        batch_size (int): Batch size
        criterion (Module): Loss function
        train_on_gpu (bool): Whether to train on GPU

    Returns:
        Tuple[float, float]: Test loss and accuracy
    """
    # Make test loop for multiclass classification
    test_losses = []  # track loss
    num_correct = 0

    # init hidden state
    h = model.init_hidden(batch_size)

    model.eval()
    # iterate over test data

    for inputs, labels in test_loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        # get predicted outputs
        output, h = model(inputs, h)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        prediction = torch.argmax(output, 1)

        # compare predictions to true label
        true_label = torch.argmax(labels, 1)

        correct_tensor = prediction.eq(true_label.data.view_as(prediction))
        correct = (
            np.squeeze(correct_tensor.numpy())
            if not train_on_gpu
            else np.squeeze(correct_tensor.cpu().numpy())
        )
        num_correct += np.sum(correct)

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)

    return np.mean(test_losses), test_acc
