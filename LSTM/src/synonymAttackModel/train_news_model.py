import torch

from typing import List, Dict

from utils.utils import save_model, load_model, change_synonyms
from utils.custom_loss_function import AdversarialLoss
from utils.data import (
    load_classification_data,
    create_vocab,
    process_texts,
    prepare_data_for_training,
)

from src.synonymAttackModel.models import SynonymAttackModelMLP
from src.sentimentAnalysis.models import SentimentRNN


def train():
    # HYPERPARAMETERS -------------------------------------------------------------------------
    path_to_headlines: str = "data/newsClassification/train.csv"
    path_to_labels: str = "data/newsClassification/test.csv"

    path_to_sentiment_model: str = (
        "models/sentimentAnalysis/sentiment_rnn_outputdim2.pt"
    )

    seq_len = 200
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For M1 Mac
    # device = torch.device("mps") if torch.backends.mps.is_available() else device

    # Sentiment model hyperparameters
    sentiment_embedding_dim: int = 400
    sentiment_hidden_dim: int = 256
    sentiment_n_layers: int = 2
    sentiment_output_size: int = 4

    # Synonym model hyperparameters
    synonym_embedding_dim: int = 400
    synonym_input_dim: int = seq_len
    synonym_hidden_dims: List[int] = [256, 128]
    synonym_output_dim: int = seq_len

    # Loss function hyperparameters
    coeffs = [0.8, 0.5, 1]

    # -----------------------------------------------------------------------------------------

    # Data loading and processing
    headlines: List[List[str]]
    labels: List[List[int]]
    headlines, labels = load_classification_data(path_to_headlines, path_to_labels)

    word2idx: Dict[str, int]
    word2idx, idx2word = create_vocab(headlines)

    features: List[List[int]] = process_texts(headlines, seq_len, word2idx)

    train_loader, valid_loader = prepare_data_for_training(features, labels, 50, 0.8)

    sentiment_model = SentimentRNN(
        vocab_size=len(word2idx) + 1,
        output_size=sentiment_output_size,
        embedding_dim=sentiment_embedding_dim,
        hidden_dim=sentiment_hidden_dim,
        n_layers=sentiment_n_layers,
    )

    sentiment_model = load_model(sentiment_model, path_to_sentiment_model, device)

    sentiment_model.to(device)

    synonym_model = SynonymAttackModelMLP(
        vocab_size=len(word2idx) + 1,
        embedding_dim=synonym_embedding_dim,
        input_dim=synonym_input_dim,
        hidden_dims=synonym_hidden_dims,
        output_dim=synonym_output_dim,
    )

    synonym_model.to(device)

    # Define the loss function and optimizer
    criterion = AdversarialLoss()
    optimizer = torch.optim.Adam(synonym_model.parameters(), lr=lr)

    # Train the model
    for epoch in range(5):
        sentiment_h = sentiment_model.init_hidden(50)
        synonym_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.long()
            labels = labels.long()

            inputs = inputs.to(device)
            labels = labels.to(device)

            output = synonym_model(inputs)

            # Con este output, calcular qué palabras cambiar por sinónimos.
            synonym_outputs: List[List[int]]
            synonym_outputs = change_synonyms(output, inputs, word2idx, idx2word)

            sentiment_h = tuple([h.data for h in sentiment_h])

            # first_sentence = inputs[0]
            # print(first_sentence[0].item())
            # input_sentence = [idx2word[idx.item()] for idx in first_sentence]
            # synonym_sentence = [idx2word[idx] for idx in synonym_outputs[0]]

            # print(f"Frase original: {input_sentence}")
            # print(f"Frase cambiada: {synonym_sentence}")

            # Luego pasar la frase cambiada por SentimentRNN y calcular el output_prob
            output_probs, sentiment_h = sentiment_model(
                torch.tensor(synonym_outputs), sentiment_h
            )

            output_probs.to(device)
            # Con el output_prob, la frase original, la frase cambiada y la label correcta(del output_prob),
            # calcular la loss
            loss, adv_loss, cosine_similarity, sum_loss = criterion(
                output_probs,
                labels,
                inputs,
                synonym_outputs,
                sentiment_model,
                3,
                coeffs,
            )

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
                print(
                    f"--> Adversarial loss (pre-coeff): {adv_loss.item() / coeffs[0]}"
                )
                print(f"--> Cosine similarity loss: {cosine_similarity.item()}")
                print(f"--> Synonym output loss: {sum_loss.item()}\n")

        # validation
        val_h = sentiment_model.init_hidden(50)
        synonym_model.eval()
        val_losses = []
        accuracies = []
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.long()
            labels = labels.long()

            inputs = inputs.to(device)
            labels = labels.to(device)

            output = synonym_model(inputs)

            synonym_outputs = change_synonyms(output, inputs, word2idx, idx2word)

            val_h = tuple([h.data for h in val_h])
            synonym_outputs = torch.tensor(synonym_outputs).to(device)

            output_probs, val_h = sentiment_model(synonym_outputs, val_h)
            loss, _, _, _ = criterion(
                output_probs,
                labels,
                inputs,
                synonym_outputs,
                sentiment_model,
                3,
                coeffs,
            )

            val_losses.append(loss.item())

            # Calculate accuracy between output_probs and labels
            predicted_outputs = torch.argmax(output_probs, dim=1)
            correct_labels = torch.argmax(labels, dim=1)
            accuracies.append(
                sum(predicted_outputs == correct_labels) / output_probs.shape[0]
            )

        print(f"Validation loss: {sum(val_losses) / len(val_losses)}")
        print(f"Validation accuracy: {sum(accuracies) / len(accuracies)}")

    save_model(synonym_model, "models/synonymAttackModel/synonym_attack_model.pt")


if __name__ == "__main__":
    train()
