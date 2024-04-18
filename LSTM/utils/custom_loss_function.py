import torch
from torch import Tensor

from typing import List


# loss = cosine_similarity + perplexity + euclidean_distance


def adversarial_loss(predictions: Tensor, labels: Tensor, kappa: float = 5) -> float:
    """Adversarial loss for sentiment analysis

    Args:
        predictions (Tensor) [batch_size, output_dim]: Predictions from the model
        labels (Tensor) [batch_size, output_dim]: Labels

    Returns:
        float: Adversarial loss
    """
    adv_loss = torch.zeros_like(predictions)

    for i, label in enumerate(labels):
        label_int = label.argmax()
        prediction = predictions[i].tolist()
        prediction.pop(label_int)
        incorrect_pred = prediction

        max_incorrect_pred = max(incorrect_pred)

        adv_loss[i] = predictions[i][label] - max_incorrect_pred + kappa

    return adv_loss.mean()


def perplexity(logits: Tensor) -> float:
    """Perplexity loss for sentiment analysis

    Args:
        logits (Tensor): Logits from the model

    Returns:
        float: Perplexity loss
    """
    return torch.exp(logits).mean()


def cosine_similarity(
    original_sentences: List[List[int]],
    perturbed_sentences: List[List[int]],
    model: torch.nn.Module,
) -> float:
    """Cosine similarity loss for sentiment analysis

    Args:
        original_sentence (Tensor) [batch_size, seq_len]: Original sentence
        perturbed_sentence (Tensor) [batch_size, seq_len]: Perturbed sentence
        model (torch.nn.Module): Sentiment Analysis model

    Returns:
        float: Cosine similarity loss
    """
    embeds = model.embedding
    original_sentences = torch.tensor(original_sentences)
    perturbed_sentences = torch.tensor(perturbed_sentences)

    original_embeds = embeds(original_sentences)
    perturbed_embeds = embeds(perturbed_sentences)

    cosine_similarity_loss = 0
    for i, original_embed in enumerate(original_embeds):
        for j, original_word in enumerate(original_embed):
            cosine_similarity_loss += torch.dot(
                original_word, perturbed_embeds[i][j]
            ) / (torch.norm(original_word) * torch.norm(perturbed_embeds[i][j]))

    # We can mean() the cosine_similarity_loss if we want to scale the loss
    return cosine_similarity_loss / original_embeds.shape[0]


def custom_loss_function(
    model: torch.nn.Module,
    original_sentence: Tensor,
    perturbed_sentence: Tensor,
    predictions: Tensor,
    labels: Tensor,
    kappa: float = 5,
) -> float:
    """Custom loss function for sentiment analysis

    Args:
        original_sentence (Tensor): Original sentence
        perturbed_sentence (Tensor): Perturbed sentence
        predictions (Tensor): Predictions from the model
        labels (Tensor): Labels
        kappa (float, optional): Adversarial loss parameter. Defaults to 5.

    Returns:
        float: Custom loss
    """
    return (
        adversarial_loss(predictions, labels, kappa)
        + cosine_similarity(original_sentence, perturbed_sentence, model)
        + perplexity(predictions)
    )
