import torch

from typing import List, Dict

from utils.utils import save_model, load_model
from utils.custom_loss_function import custom_loss_function
from utils.data import load_sentiment_data, create_vocab, process_texts

from src.synonymAttackModel.models import SynonymAttackModelMLP
from src.sentimentAnalysis.models import SentimentRNN


def train():
    # HYPERPARAMETERS -------------------------------------------------------------------------
    path_to_reviews: str = "data/sentimentAnalysis/train_data/reviews.txt"
    path_to_labels: str = "data/sentimentAnalysis/train_data/labels.txt"

    path_to_sentiment_model: str = "models/sentimentAnalysis/sentiment_rnn.pt"

    seq_len = 200
    lr = 0.001

    train_on_gpu: bool = torch.cuda.is_available()

    # Sentiment model hyperparameters
    sentiment_embedding_dim: int = 400
    sentiment_hidden_dim: int = 256
    sentiment_n_layers: int = 2
    sentiment_output_size: int = 1

    # Synonym model hyperparameters
    synonym_embedding_dim: int = 400
    synonym_input_dim: int = seq_len
    synonym_hidden_dims: List[int] = [256, 128]
    synonym_output_dim: int = seq_len

    # -----------------------------------------------------------------------------------------

    # Data loading and processing
    reviews: List[List[str]]
    labels: List[List[int]]
    reviews, labels = load_sentiment_data(path_to_reviews, path_to_labels)

    word2idx: Dict[str, int]
    word2idx, _ = create_vocab(reviews)

    features: List[List[int]] = process_texts(reviews, seq_len, word2idx)

    sentiment_model = SentimentRNN(
        vocab_size=len(word2idx) + 1,
        output_size=sentiment_output_size,
        embedding_dim=sentiment_embedding_dim,
        hidden_dim=sentiment_hidden_dim,
        n_layers=sentiment_n_layers,
    )

    sentiment_model = load_model(sentiment_model, path_to_sentiment_model, train_on_gpu)

    if train_on_gpu:
        sentiment_model.cuda()

    synonym_model = SynonymAttackModelMLP(
        vocab_size=len(word2idx) + 1,
        embedding_dim=synonym_embedding_dim,
        input_dim=synonym_input_dim,
        hidden_dims=synonym_hidden_dims,
    )

    if train_on_gpu:
        synonym_model.cuda()

    # Define the loss function and optimizer
    criterion = custom_loss_function
    optimizer = torch.optim.Adam(synonym_model.parameters(), lr=lr)

    # Train the model
    for epoch in range(10):
        for i, (inputs, labels) in enumerate(zip(features, features)):
            inputs = torch.tensor(inputs).long()
            labels = torch.tensor(labels).long()

            if train_on_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            output = synonym_model(inputs)

            # Con este output, calcular qué palabras cambiar por sinónimos.
            # Luego pasar la frase cambiada por SentimentRNN y calcular el output_prob
            # Con el output_prob, la frase original, la frase cambiada y la label correcta(del output_prob),
            # calcular la loss

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
