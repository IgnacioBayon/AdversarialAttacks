import torch

from utils.train_functions import train_model
from utils.utils import create_vocab, save_model
from utils.data import (
    load_classification_data,
    process_texts,
    prepare_data_for_training,
)
from src.newsClassification.models import ClassificationRNN

from typing import List, Dict


def main():
    # HYPERPARAMETERS -------------------------------------------------------------------------
    seq_len: int = 200
    train_on_gpu: bool = torch.cuda.is_available()
    print(f"Training on GPU: {train_on_gpu}")
    lr: float = 0.001
    epochs: int = 4
    batch_size: int = 64
    split: float = 0.8

    path_to_data: str = "data/newsClassification/train.csv"

    path_to_save: str = "models/newsClassification/classification_rnn.pth"

    embedding_dim: int = 400
    hidden_dim: int = 256
    n_layers: int = 2
    output_size: int = 4
    # -----------------------------------------------------------------------------------------

    # Data loading and processing
    headlines: List[List[str]]
    labels: List[List[int]]
    headlines, labels = load_classification_data(path_to_data)

    word2idx: Dict[str, int]
    word2idx, _ = create_vocab(headlines)

    features: List[List[int]] = process_texts(headlines, seq_len, word2idx)

    # Define the model, criterion, and optimizer
    model = ClassificationRNN(
        vocab_size=len(word2idx) + 1,
        output_size=output_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )

    if train_on_gpu:
        model.cuda()

    train_loader, valid_loader = prepare_data_for_training(
        features, labels, batch_size, split
    )

    # Define criterion and optimizer (CrossEntropyLoss for newsClassification)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model = train_model(
        epochs=epochs,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        batch_size=batch_size,
        criterion=criterion,
        optimizer=optimizer,
        train_on_gpu=train_on_gpu,
    )

    save_model(model, path_to_save)

    return None


if __name__ == "__main__":
    main()
