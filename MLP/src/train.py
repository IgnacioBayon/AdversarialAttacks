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

    task = "binary"
    # task = "mutliclass"
    if task == "binary":
        data_path = "data/IMDB_Dataset.csv"
        model_path = "models/sentiment_analysis.pt"
        output_dim = 2
    elif task == "multiclass":
        data_path = "data/News_Dataset.csv"
        model_path = "models/news_classification.pt"
        output_dim = 4
    data_type = task

    epochs = 6
    lr = 0.0001
    batch_size = 64
    hidden_sizes = [32, 16]
    embedding_dim = 100

    train_loader, val_loader, _, vocab_size, vocab_to_int, int_to_vocab, sentence_length = \
        generate_data_loader(data_path, data_type, batch_size)

    model = MultiLayerPerceptron(
        vocab_size, embedding_dim, hidden_sizes, output_dim
    ).to(device)

    train_model(
        model, train_loader, val_loader, epochs, lr, device, task
    )

    save_model(model, model_path)

    print("\nTraining complete.")
