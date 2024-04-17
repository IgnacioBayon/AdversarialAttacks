import torch
import numpy as np
import random

try:
    from src.utils import load_model, evaluate_model
    from src.data_processing import generate_data_loader
    from src.models import MultiLayerPerceptron
except ImportError:
    from utils import load_model, evaluate_model
    from data_processing import generate_data_loader
    from models import MultiLayerPerceptron


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


if __name__ == "__main__":

    data_path = "data/IMDB_Dataset.csv"
    model_path = "models/best_model.pt"

    batch_size = 64
    embedding_dim = 100
    epochs = 25
    lr = 0.0001
    batch_size = 64
    patience = 100
    hidden_sizes = [32, 16]
    output_dim = 1

    _, _, test_loader, vocab_size, _, _, sentence_length = generate_data_loader(
        data_path, batch_size
    )
    model = MultiLayerPerceptron(
        vocab_size, sentence_length, embedding_dim, hidden_sizes, output_dim
    ).to(device)

    model = load_model(model, model_path, device)
    model.to(device)

    acc = evaluate_model(model, test_loader, device)

    print(f"Test Accuracy: {acc}%")
