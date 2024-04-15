from bert_score.utils import get_idf_dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import transformers
import math
import argparse
import math
import jiwer
import numpy as np
import os
import warnings
transformers.logging.set_verbosity(transformers.logging.ERROR)
import time 
import torch
import torch.nn.functional as F

from src.dataset import load_data
from src.utils import bool_flag, get_output_file, print_args, load_gpt2_from_dict

def wer(x, y):
    '''
    Calculate the Word Error Rate (WER) between two strings.

    Args:
        x (str): The first string.
        y (str): The second string.

    Returns:
        float: The WER between the two strings.
    '''
    # Remove extra spaces
    x = " ".join(x.split())
    y = " ".join(y.split())

    return jiwer.wer(x, y)

def bert_score(refs, cands, weights=None):
    '''
    Calculate the BERTScore between two strings.

    Args:
        refs (str): The reference string.
        cands (str): The candidate string.
        weights (list): The weights for the BERTScore.

    Returns:
        float: The BERTScore between the two strings.
    '''
    # normalize the weights to sum to 1
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)

    # apply weights
    if weights is not None:
        refs_norm = refs_norm * weights[:, None]

    # normalize the candidates
    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)

    # calculate the cosine similarity
    cosines = refs_norm @ cands_norm.transpose(-1, -2)

    # calculate the BERTScore as the maximum cosine similarity
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(dim=-1).sum(1)

    return R

def calculate_perplexity(logits, coefficients):
    '''
    Calculate the perplexity of a set of logits.

    Args:
        logits (torch.Tensor): The logits.
        coefficients (torch.Tensor): The coefficients.

    Returns:
        float: The perplexity of the logits.
    '''
    # Shift logits and coefficients to align with each other
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_coefficients = coefficients[:, 1:, :].contiguous()
    
    # Ensure the dimensions match
    shifted_logits = shifted_logits[:, :, :shifted_coefficients.size(2)]
    
    # Calculate the negative log likelihood of the shifted logits
    # multiplied by the coefficients, summed across the vocabulary dimension,
    # and then averaged across the batch dimension
    negative_log_likelihood = -(shifted_coefficients * F.log_softmax(shifted_logits, dim=-1)).sum(-1).mean()
    
    return negative_log_likelihood


