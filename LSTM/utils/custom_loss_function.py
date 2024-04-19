import torch
from torch import Tensor

from typing import List


# loss = cosine_similarity + perplexity + euclidean_distance


class AdversarialLoss(torch.nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(
        self,
        predictions: Tensor,
        labels: Tensor,
        original_sentence: Tensor,
        perturbed_sentence: Tensor,
        model: torch.nn.Module,
        kappa: float = 5,
        coeffs: List[float] = [1, 1, 1],
    ) -> float:

        adversarial_coeff, cosine_similarity_coeff, sum_coeff = coeffs

        adv_loss = adversarial_loss(predictions, labels, kappa) * adversarial_coeff

        cosine_similarity_loss = (
            cosine_similarity(original_sentence, perturbed_sentence, model)
            * cosine_similarity_coeff
        )

        synonym_output_loss = synonym_output_sum(predictions) * sum_coeff

        loss = adv_loss + cosine_similarity_loss + synonym_output_loss

        return loss, adv_loss, cosine_similarity_loss, synonym_output_loss


def synonym_output_sum(synonym_outputs: Tensor) -> Tensor:
    """Given a tensor of synonym outputs, return the sum of the outputs

    Args:
        synonym_outputs (Tensor) [batch_size, seq_len]: Tensor of synonym outputs

    Returns:
        Tensor: Sum of the synonym outputs
    """
    synonym_sum = torch.sum(synonym_outputs, dim=1)
    return synonym_sum.mean()


# def adversarial_loss(predictions: Tensor, labels: Tensor, kappa: float = 5) -> Tensor:
#     """Adversarial loss for sentiment analysis

#     Args:
#         predictions (Tensor) [batch_size, output_dim]: Predictions from the model
#         labels (Tensor) [batch_size, output_dim]: Labels

#     Returns:
#         float: Adversarial loss
#     """
#     adv_loss = torch.zeros_like(predictions)

#     for i, label in enumerate(labels):
#         label_int = label.argmax()
#         prediction = predictions[i].tolist()
#         prediction.pop(label_int)
#         incorrect_pred = prediction

#         max_incorrect_pred = max(incorrect_pred)

#         adv_loss[i] = predictions[i][label_int].item() - max_incorrect_pred + kappa

#     return adv_loss.mean()


def adversarial_loss(predictions: Tensor, labels: Tensor, kappa: float = 5) -> Tensor:
    """Adversarial loss for sentiment analysis

    Args:
        predictions (Tensor) [batch_size, output_dim]: Predictions from the model
        labels (Tensor) [batch_size, output_dim]: Labels
        kappa (float): Margin parameter

    Returns:
        tensor: Adversarial loss
    """
    batch_size, output_dim = predictions.size()
    label_indices = labels.argmax(dim=1)

    # Extract the predicted probabilities for correct and incorrect classes
    correct_probabilities = predictions[range(batch_size), label_indices]
    incorrect_probabilities = predictions.clone()
    incorrect_probabilities[range(batch_size), label_indices] = float("-inf")

    # Find the maximum probability among incorrect classes
    max_incorrect_prob, _ = incorrect_probabilities.max(dim=1)

    # Compute the adversarial loss
    adv_loss = correct_probabilities - max_incorrect_prob + kappa
    adv_loss = torch.clamp(adv_loss, min=0)  # Apply the hinge loss

    return adv_loss.mean()


def perplexity(logits: Tensor) -> float:
    """Perplexity loss for sentiment analysis

    Args:
        logits (Tensor): Logits from the model

    Returns:
        float: Perplexity loss
    """
    return torch.exp(logits).mean()


# def cosine_similarity(
#     original_sentences: List[List[int]],
#     perturbed_sentences: List[List[int]],
#     model: torch.nn.Module,
# ) -> float:
#     """Cosine similarity loss for sentiment analysis

#     Args:
#         original_sentence (Tensor) [batch_size, seq_len]: Original sentence
#         perturbed_sentence (Tensor) [batch_size, seq_len]: Perturbed sentence
#         model (torch.nn.Module): Sentiment Analysis model

#     Returns:
#         float: Cosine similarity loss
#     """
#     embeds = model.embedding
#     # original_sentences = torch.tensor(original_sentences)
#     perturbed_sentences = torch.tensor(perturbed_sentences)

#     original_embeds = embeds(original_sentences)
#     perturbed_embeds = embeds(perturbed_sentences)

#     # GUARDAR SIMILARITY DE CADA PALABRA Y HACER MEDIA (SIMILARITY DE LA FRASE)
#     # HACER MEDIA DE CADA SIMILARITY DE LAS FRASES PARA SACAR UN VALOR

#     batch_size, seq_len = original_sentences.shape

#     cosine_similarity_loss = torch.zeros(batch_size)

#     for i, original_embed in enumerate(original_embeds):
#         cosine_similarity_loss_sentence = torch.zeros(seq_len)
#         for j, original_word in enumerate(original_embed):
#             cosine_similarity_loss_sentence[j] = torch.dot(
#                 original_word, perturbed_embeds[i][j]
#             ) / (torch.norm(original_word) * torch.norm(perturbed_embeds[i][j]))

#         cosine_similarity_loss[i] = sum(cosine_similarity_loss_sentence) / len(
#             cosine_similarity_loss_sentence
#         )

#     cosine_similarity_loss = cosine_similarity_loss.mean()

#     # We can mean() the cosine_similarity_loss if we want to scale the loss
#     return cosine_similarity_loss


def cosine_similarity(
    original_sentences: List[List[int]],
    perturbed_sentences: List[List[int]],
    model: torch.nn.Module,
) -> Tensor:
    """Cosine similarity loss for sentiment analysis

    Args:
        original_sentences (List[List[int]]): Original sentences
        perturbed_sentences (List[List[int]]): Perturbed sentences
        model (torch.nn.Module): Sentiment Analysis model

    Returns:
        tensor: Cosine similarity loss
    """
    embeds = model.embedding

    perturbed_sentences = torch.tensor(perturbed_sentences)

    original_embeds = embeds(original_sentences)
    perturbed_embeds = embeds(perturbed_sentences)

    original_norms = torch.norm(original_embeds, dim=-1)
    perturbed_norms = torch.norm(perturbed_embeds, dim=-1)

    similarities = torch.einsum("ijk,ijk->ij", original_embeds, perturbed_embeds)
    cosine_similarities = similarities / (original_norms * perturbed_norms)

    # Take the mean of cosine similarities across each sentence
    cosine_similarity_loss = cosine_similarities.mean()

    return cosine_similarity_loss
