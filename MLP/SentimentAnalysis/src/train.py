import torch
import numpy as np
import random
import os

try:
    from data_processing import generate_data_loader
    from utils import save_model, train_model
    from models import MultiLayerPerceptron
except ImportError:
    from src.data_processing import generate_data_loader
    from src.utils import save_model, train_model
    from src.models import MultiLayerPerceptron


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    data_path = "data/IMDB_Dataset.csv"
    model_path = "models/best_model.pt"

    embedding_dim = 100
    epochs = 25
    lr = 0.0001
    batch_size = 64
    patience = 100
    hidden_sizes = [32, 16]
    output_dim = 1

    train_loader, val_loader, _, vocab_size, vocab_to_int, int_to_vocab, sentence_length = generate_data_loader(
        data_path, batch_size
    )

    model = MultiLayerPerceptron(
        vocab_size, sentence_length, embedding_dim, hidden_sizes, output_dim
    ).to(device)

    train_model(
        model, train_loader, val_loader, epochs, lr, device
    )

    save_model(model, model_path)

    # plot_embeddings(model.embedding, int_to_vocab)

    print("Training complete.")
