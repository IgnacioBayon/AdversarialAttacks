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

    lr = 0.0001
    batch_size = 64
    hidden_sizes = [32, 16]
    embedding_dim = 100

    _, _, test_loader, vocab_size, _, _, sentence_length = \
        generate_data_loader(data_path, data_type, batch_size)

    model = MultiLayerPerceptron(
        vocab_size, embedding_dim, hidden_sizes, output_dim
    ).to(device)

    model = load_model(model, model_path, device)
    model.to(device)

    acc = evaluate_model(model, test_loader, device, task)

    print(f"Test Accuracy: {acc:.4f} %")
