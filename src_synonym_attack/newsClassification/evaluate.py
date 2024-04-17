import torch
import numpy as np

from utils.train_functions import evaluate_model
from utils.utils import load_model, create_vocab, tokenize_sentence
from utils.data import load_classification_data, process_texts, prepare_data_for_testing

from src.newsClassification.models import ClassificationRNN
from src.attacks.adversarialAttacks import generate_synonym_texts, synonym_attack

from typing import List, Dict


def predict_news_category(
    model: torch.nn.Module,
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    headline: str,
    seq_len: int = 200,
    train_on_gpu: bool = False,
) -> str:
    """Predicts the category of a news headline

    Args:
        model (torch.nn.Module): _description_
        word2idx (Dict[str, int]): _description_
        idx2word (Dict[int, str]): _description_
        headline (str): _description_
        seq_len (int, optional): _description_. Defaults to 200.
        train_on_gpu (bool, optional): _description_. Defaults to False.

    Returns:
        str: _description_
    """
    headline = headline.lower()
    headline = tokenize_sentence(headline)
    synonym_headline = synonym_attack(headline)

    headline = process_texts([headline], seq_len, word2idx)
    synonym_headline = process_texts([synonym_headline], seq_len, word2idx)

    headline = np.array(headline)
    synonym_headline = np.array(synonym_headline)

    headline = torch.from_numpy(headline).to(torch.int64)
    synonym_headline = torch.from_numpy(synonym_headline).to(torch.int64)

    if train_on_gpu:
        headline = headline

    model.eval()
    h = model.init_hidden(64)
    with torch.no_grad():
        output, h = model(headline, h)
        synonym_output = model(synonym_headline)

    _, predicted = torch.max(output, 1)
    _, synonym_predicted = torch.max(synonym_output, 1)

    return idx2word[predicted.item()], idx2word[synonym_predicted.item()]


def main():
    # HYPERPARAMETERS -------------------------------------------------------------------------
    path_to_model: str = "models/newsClassification/classification_rnn.pt"
    path_to_test_data: str = "data/newsClassification/test.csv"
    path_to_train_data: str = "data/newsClassification/train.csv"

    batch_size: int = 64
    embedding_dim: int = 400
    hidden_dim: int = 256
    n_layers: int = 2
    output_size: int = 4
    train_on_gpu: bool = torch.cuda.is_available()
    print(f"Training on GPU: {train_on_gpu}")
    # -----------------------------------------------------------------------------------------
    # Create vocab from training data, in order to load the model correctly (same vocab size)
    train_headlines: List[List[str]]
    train_headlines, _ = load_classification_data(path_to_train_data)

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
    headlines, labels = load_classification_data(path_to_test_data)

    # SYNONYM ATTACK
    synonym_headlines = generate_synonym_texts(headlines)

    features: List[List[int]] = process_texts(headlines, seq_len=200, word2idx=word2idx)

    test_loader = prepare_data_for_testing(features, labels, batch_size)

    # Define criterion
    criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate_model(
        model=model,
        test_loader=test_loader,
        batch_size=batch_size,
        criterion=criterion,
        train_on_gpu=train_on_gpu,
    )

    print("Original test data")
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

    # Evaluate synonym attack
    features: List[List[int]] = process_texts(
        synonym_headlines, seq_len=200, word2idx=word2idx
    )

    test_synonym_loader = prepare_data_for_testing(features, labels, batch_size)

    test_loss, test_acc = evaluate_model(
        model=model,
        test_loader=test_synonym_loader,
        batch_size=batch_size,
        criterion=criterion,
        train_on_gpu=train_on_gpu,
    )

    print("Synonym attack")
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

    return None


if __name__ == "__main__":
    main()
