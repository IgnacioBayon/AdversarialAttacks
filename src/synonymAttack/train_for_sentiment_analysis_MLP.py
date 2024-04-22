import torch

import pickle

from typing import List, Dict

from utils.utils import save_model, load_model, change_synonyms
from utils.custom_loss_function import AdversarialLoss
from utils.mlp_data import generate_data_loader
from utils.mlp_utils import save_model, train_model
from src.ownModels.models import MultiLayerPerceptron

from src.ownModels.attack_models import SynonymAttackModelMLP, SynonymAttackRNN
from src.ownModels.models import MultiLayerPerceptron


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # HYPERPARAMETERS -------------------------------------------------------------------------
    path_to_data: str = "data/IMDB_Dataset.csv"

    path_to_sentiment_model: str = (
        "models/sentimentAnalysis/mlp_sentiment_analysis_low.pt"
    )

    data_type = "binary"

    seq_len = 200
    lr = 0.001

    # For M1 Mac
    # device = torch.device("mps") if torch.backends.mps.is_available() else device

    batch_size = 64
    sentiment_hidden_sizes = [32, 16]
    sentiment_embedding_dim = 100
    sentiment_output_dim = 2

    # Synonym model hyperparameters
    synonym_embedding_dim: int = 400
    synonym_input_dim: int = seq_len
    synonym_hidden_dim: List[int] = 256
    synonym_output_dim: int = seq_len
    synonym_n_layers: int = 2

    # Loss function hyperparameters
    coeffs = [7, 0.1, 0.005]

    # -----------------------------------------------------------------------------------------

    (
        train_loader,
        val_loader,
        _,
        vocab_size,
        word2idx,
        idx2word,
        sentence_length,
    ) = generate_data_loader(path_to_data, data_type, batch_size)

    sentiment_model = MultiLayerPerceptron(
        vocab_size + 1,
        sentiment_embedding_dim,
        sentiment_hidden_sizes,
        sentiment_output_dim,
    )

    sentiment_model = load_model(
        sentiment_model, path_to_sentiment_model, device)

    sentiment_model.to(device)

    synonym_model = SynonymAttackRNN(
        vocab_size=vocab_size,
        output_size=synonym_output_dim,
        embedding_dim=synonym_embedding_dim,
        hidden_dim=synonym_hidden_dim,
        n_layers=synonym_n_layers,
    )
    synonym_model.to(device)

    # Define the loss function and optimizer
    criterion = AdversarialLoss()
    optimizer = torch.optim.Adam(synonym_model.parameters(), lr=lr)

    print(f"Synonym model: \n{synonym_model}")
    print(f"Training on: {device}")

    sentiment_model.eval()

    total_accuracies = []
    # Train the model
    for epoch in range(2):
        synonym_h = synonym_model.init_hidden(batch_size)
        synonym_model.train()
        accuracies = []
        for i, (inputs, labels) in enumerate(train_loader):
            batch_accuracy = 0
            inputs = inputs.long()
            labels = labels.long()

            inputs = inputs.to(device)
            labels = labels.to(device)

            synonym_h = tuple([h.data for h in synonym_h])

            synonym_output, synonym_h = synonym_model(inputs, synonym_h)

            # With this output, calculate which words to change for synonyms.
            synonym_inputs: List[List[int]]
            synonym_inputs = change_synonyms(
                synonym_output, inputs, word2idx, idx2word)
            synonym_inputs = torch.tensor(synonym_inputs).to(device)

            # Luego pasar la frase cambiada por sentimentRNN y calcular el output_prob

            output_probs = sentiment_model(synonym_inputs)

            output_probs.to(device)
            # Con el output_prob, la frase original, la frase cambiada y la label correcta(del output_prob),
            # calcular la loss
            loss, adv_loss, cosine_similarity_loss, sum_loss = criterion(
                synonym_outputs=synonym_output,
                predictions=output_probs,
                labels=labels,
                original_sentence=inputs,
                perturbed_sentence=synonym_inputs,
                model=sentiment_model,
                kappa=3,
                coeffs=coeffs,
            )

            # Calculate accuracy between output_probs and labels
            predicted_outputs = torch.argmax(output_probs, dim=1)
            correct_labels = torch.argmax(labels, dim=1)
            batch_accuracy = (
                sum(predicted_outputs == correct_labels) /
                output_probs.shape[0]
            )
            accuracies.append(batch_accuracy)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
                print(
                    f"--> Adversarial loss (pre-coeff): {adv_loss.item() / coeffs[0]}"
                )
                print(
                    f"--> Cosine similarity loss (pre-coeff): {cosine_similarity_loss.item() / coeffs[1]}"
                )
                print(
                    f"--> Synonym output loss (pre-coeff): {sum_loss.item() / coeffs[2]}\n"
                )
                # first_sentence = inputs[0]
                # input_sentence = [idx2word[idx.item()] for idx in first_sentence]
                # synonym_sentence = [idx2word[idx] for idx in synonym_inputs[0]]

                # print(f"Frase original: {input_sentence}")
                # print(f"Frase cambiada: {synonym_sentence}\n\n")

                # # Print the number of times the synonym model has changed words (synonym_output > 0.5)
                print(
                    f"Number of words changed: {sum(sum([1 for i in output if i > 0.5]) for output in synonym_output)}\n\n"
                )
                print(f"Accuracy: {batch_accuracy}\n\n")

        print(
            f"Accuracy (avg over all epochs): {sum(accuracies) / len(accuracies)}\n\n"
        )
        total_accuracies += accuracies

        # validation
        val_synonym_h = synonym_model.init_hidden(batch_size)
        synonym_model.eval()
        val_losses = []
        accuracies = []
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.long()
            labels = labels.long()

            inputs = inputs.to(device)
            labels = labels.to(device)

            val_synonym_h = tuple([h.data for h in val_synonym_h])
            synonym_outputs, val_synonym_h = synonym_model(
                inputs, val_synonym_h)

            synonym_inputs: List[List[int]]
            synonym_inputs = change_synonyms(
                synonym_outputs, inputs, word2idx, idx2word
            )

            synonym_inputs = torch.tensor(synonym_inputs).to(device)

            output_probs = sentiment_model(synonym_inputs)
            loss, _, _, _ = criterion(
                synonym_outputs=synonym_outputs,
                predictions=output_probs,
                labels=labels,
                original_sentence=inputs,
                perturbed_sentence=synonym_inputs,
                model=sentiment_model,
                kappa=3,
                coeffs=coeffs,
            )

            val_losses.append(loss.item())

            # Calculate accuracy between output_probs and labels
            predicted_outputs = torch.argmax(output_probs, dim=1)
            correct_labels = torch.argmax(labels, dim=1)
            accuracies.append(
                sum(predicted_outputs == correct_labels) /
                output_probs.shape[0]
            )

        print(f"Validation loss: {sum(val_losses) / len(val_losses)}")
        print(f"Validation accuracy: {sum(accuracies) / len(accuracies)}")

    with open("accuracies_list_sentiment_mlp.pkl", "wb") as f:
        pickle.dump(total_accuracies, f)

    save_model(
        synonym_model,
        "models/synonymAttack/synonym_attack_mlp_sentiment.pt",
    )


if __name__ == "__main__":
    train()
