import torch
import numpy as np
import random

from utils.mlp_utils import load_model, evaluate_model
from utils.mlp_data import generate_data_loader
from src.ownModels.models import MultiLayerPerceptron


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


if __name__ == "__main__":

    task = "binary"

    data_path = "data/IMDB_Dataset.csv"
    model_path = "models/sentimentAnalysis/mlp_sentiment_analysis_high.pt"
    # model_path = "models/sentimentAnalysis/mlp_sentiment_analysis_low.pt"
    output_dim = 2

    data_type = task

    lr = 0.0001
    batch_size = 64
    hidden_sizes = [32, 16]
    embedding_dim = 100

    _, _, test_loader, vocab_size, _, _, sentence_length = generate_data_loader(
        data_path, data_type, batch_size, one_hot_labels=True
    )

    model = MultiLayerPerceptron(
        vocab_size + 1, embedding_dim, hidden_sizes, output_dim
    ).to(device)

    model = load_model(model, model_path, device)
    model.to(device)

    acc = evaluate_model(model, test_loader, device, task)

    print(f"Test Accuracy: {acc:.4f} %")
