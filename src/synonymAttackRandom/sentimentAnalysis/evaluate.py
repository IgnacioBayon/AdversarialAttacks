import torch

from utils.train_functions import evaluate_model
from utils.utils import load_model, create_vocab
from utils.data import load_sentiment_data, process_texts, prepare_data_for_testing

from src.synonymAttackRandom.newsClassification.models import ClassificationRNN

from typing import List, Dict


def main():
    # HYPERPARAMETERS -------------------------------------------------------------------------
    path_to_model: str = "models/sentimentAnalysis/sentiment_rnn.pt"

    path_to_test_reviews: str = "data/sentimentAnalysis/test_data/reviews.txt"
    path_to_test_labels: str = "data/sentimentAnalysis/test_data/labels.txt"

    path_to_train_reviews: str = "data/sentimentAnalysis/train_data/reviews.txt"
    path_to_train_labels: str = "data/sentimentAnalysis/train_data/labels.txt"

    batch_size: int = 64
    embedding_dim: int = 400
    hidden_dim: int = 256
    n_layers: int = 2
    output_size: int = 1
    train_on_gpu: bool = torch.cuda.is_available()
    print(f"Training on GPU: {train_on_gpu}")
    # -----------------------------------------------------------------------------------------
    # Create vocab from training data, in order to load the model correctly (same vocab size)
    train_headlines: List[List[str]]
    train_headlines, _ = load_sentiment_data(
        path_to_train_reviews, path_to_train_labels
    )

    word2idx: Dict[str, int]
    word2idx, _ = create_vocab(train_headlines)

    model = ClassificationRNN(
        vocab_size=len(word2idx) + 1,
        output_size=output_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )
    model = load_model(model, path_to_model, train_on_gpu)

    # Load test data
    headlines: List[List[str]]
    labels: List[List[int]]
    headlines, labels = load_sentiment_data(path_to_test_reviews, path_to_test_labels)

    features: List[List[int]] = process_texts(headlines, seq_len=200, word2idx=word2idx)

    test_loader = prepare_data_for_testing(features, labels, batch_size)

    # Define criterion
    criterion = torch.nn.BCELoss()

    test_loss, test_acc = evaluate_model(
        model=model,
        test_loader=test_loader,
        batch_size=batch_size,
        criterion=criterion,
        train_on_gpu=train_on_gpu,
    )

    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

    return None


if __name__ == "__main__":
    main()
