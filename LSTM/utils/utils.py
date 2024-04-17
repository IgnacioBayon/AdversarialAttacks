import torch
from torch.nn import Module

from typing import List, Tuple, Dict
import re
import string


def remove_punctuations(text: List[str]) -> List[str]:
    """To remove all the punctuations present in the text.Input the text column"""
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenizes a sentence into words

    Args:
        sentence (str): Input sentence

    Returns:
        List[str]: List of words (tokens)
    """
    sentence = remove_punctuations(sentence)
    sentence = re.sub(r"[^A-Za-z0-9(),.!?\'`\-\"]", " ", sentence)
    sentence = re.sub(r"\'s", " 's", sentence)
    sentence = re.sub(r"\'ve", " 've", sentence)
    sentence = re.sub(r"n\'t", " n't", sentence)
    sentence = re.sub(r"\'re", " 're", sentence)
    sentence = re.sub(r"\'d", " 'd", sentence)
    sentence = re.sub(r"\'ll", " 'll", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\-", " - ", sentence)
    sentence = re.sub(r"\"", ' " ', sentence)
    # We may have introduced double spaces, so collapse these down
    sentence = re.sub(r"\s{2,}", " ", sentence)
    sentence = sentence.lower()
    return list(filter(lambda x: len(x) > 0, sentence.split(" ")))


def tokenize(text: List[str]) -> List[List[str]]:
    """Tokenizes a list of sentences into words

    Args:
        text (List[str]): List of sentences

    Returns:
        List[List[str]]: List of list of words (tokens)
    """
    return [tokenize_sentence(sentence) for sentence in text]


def create_vocab(texts: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Creates word2idx and idx2word dictionaries from a list of sentences

    Args:
        text (List[List[str]]): List of list of words (tokens)

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: word2idx and idx2word dictionaries
    """
    word2idx = {}
    idx2word = {}
    idx = 1
    for text in texts:
        for word in text:
            if word not in word2idx:
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1

    return word2idx, idx2word


def pad_features(texts: List[List[int]], seq_length: int) -> List[List[int]]:
    """Pad text features to a fixed length

    Args:
        texts (List[List[int]]): List of word2idx sentences
        seq_length (int): Fixed length to pad the features

    Returns:
        List[List[int]]: Padded features
    """
    features: List[List[int]] = []
    for text in texts:
        if len(text) >= seq_length:
            features.append(text[:seq_length])
        else:
            features.append([0] * (seq_length - len(text)) + text)

    return features


def save_model(model: Module, path: str) -> None:
    """Saves the model to the given path

    Args:
        model (torch.nn.Module): The model to save
        path (str): Path to save the model
    """
    torch.save(model.state_dict(), path)


def load_model(model: Module, path: str, train_on_gpu: bool) -> Module:
    """Loads the model from the given path

    Args:
        model (torch.nn.Module): The model to load
        path (str): Path to load the model

    Returns:
        torch.nn.Module: The loaded model
    """
    if train_on_gpu:
        model.load_state_dict(torch.load(path, map_location=torch.device("cuda")))
        return model
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

    return model
