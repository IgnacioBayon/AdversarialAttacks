import torch

from utils.train_functions import evaluate_model
from utils.utils import load_model, create_vocab
from utils.data import load_sentiment_data_txt, process_texts, prepare_data_for_testing

from src.ownModels.models import SentimentRNN
from src.synonymAttackRandom.attacks.adversarialAttacks import generate_synonym_texts

from typing import List, Dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # HYPERPARAMETERS -------------------------------------------------------------------------
    path_to_model: str = "models/sentimentAnalysis/lstm_sentiment.pth"
    path_to_test_reviews: str = "data/sentimentAnalysis/test_data/reviews.txt"
    path_to_test_labels: str = "data/sentimentAnalysis/test_data/labels.txt"
    path_to_train_reviews: str = "data/sentimentAnalysis/train_data/reviews.txt"
    path_to_train_labels: str = "data/sentimentAnalysis/train_data/labels.txt"

    seq_len = 200
    batch_size: int = 64
    embedding_dim: int = 400
    hidden_dim: int = 256
    n_layers: int = 2
    output_size: int = 2
    # -----------------------------------------------------------------------------------------

    # Create vocab from training data, in order to load the model correctly (same vocab size)
    train_headlines: List[List[str]]
    train_headlines, _ = load_sentiment_data_txt(
        path_to_train_reviews, path_to_train_labels
    )

    word2idx: Dict[str, int]
    word2idx, _ = create_vocab(train_headlines)

    model = SentimentRNN(
        vocab_size=len(word2idx),
        output_size=output_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )
    model = load_model(model, path_to_model, device)

    # Load test data
    headlines: List[List[str]]
    labels: List[List[int]]
    headlines, labels = load_sentiment_data_txt(
        path_to_test_reviews, path_to_test_labels)

    features: List[List[int]] = process_texts(
        headlines, seq_len=seq_len, word2idx=word2idx)

    test_loader = prepare_data_for_testing(features, labels, batch_size)

    # Define criterion
    criterion = torch.nn.CrossEntropyLoss()

    # ORIGINAL DATA
    test_loss, test_acc = evaluate_model(
        model=model,
        test_loader=test_loader,
        batch_size=batch_size,
        criterion=criterion,
        device=device,
    )

    print("\nOriginal test data")
    print(f"\tTest loss: {test_loss}")
    print(f"\tTest accuracy: {test_acc}")

    # SYNONYM ATTACK
    synonym_headlines = generate_synonym_texts(headlines)
    
    features: List[List[int]] = process_texts(
        synonym_headlines, seq_len=seq_len, word2idx=word2idx
    )
    print("Synonym attack features: ", features[:2])

    test_synonym_loader = prepare_data_for_testing(
        features, labels, batch_size)
    print("Test Synonym Loader: ", test_synonym_loader[:2])

    test_loss, test_acc = evaluate_model(
        model=model,
        test_loader=test_synonym_loader,
        batch_size=batch_size,
        criterion=criterion,
        device=device,
    )

    print("\nSynonym attack")
    print(f"\tTest loss: {test_loss}")
    print(f"\tTest accuracy: {test_acc}")

    return None


if __name__ == "__main__":
    main()
