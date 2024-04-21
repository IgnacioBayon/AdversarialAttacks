import torch
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np

from pandas import DataFrame
from typing import List, Tuple, Dict

# own imports
from utils.utils import tokenize, pad_features


def load_sentiment_data(
    path_reviews: str, path_labels: str
) -> Tuple[List[List[str]], List[int]]:
    """Loads sentiment analysis data from .txt files on the given paths

    Args:
        path_reviews (str): Path to the reviews file
        path_labels (str): Path to the labels file

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the reviews and labels
    """

    with open(path_reviews, "r") as f:
        reviews = f.read()
    with open(path_labels, "r") as f:
        labels = f.read()

    reviews = reviews.lower()
    reviews = reviews.split("\n")

    reviews = tokenize(reviews)

    labels = labels.lower()
    labels = labels.split("\n")

    labels = [1 if label == "positive" else 0 for label in labels]

    return reviews, labels


def load_classification_data(path: str) -> Tuple[List[List[str]], List[List[int]]]:
    """Loads the news classification data from .csv file at the given path

    Args:
        path (str): Path to the training data

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the news headlines and one-hot encoded labels
    """

    df = pd.read_csv(path)

    titles = df["title"].to_list()
    descriptions = df["description"].to_list()
    labels = df["label"].to_list()

    titles = tokenize(titles)
    descriptions = tokenize(descriptions)

    headlines = [
        title + description for title, description in zip(titles, descriptions)
    ]

    one_hot_labels = [
        [1 if label == i + 1 else 0 for i in range(4)] for label in labels
    ]

    return headlines, one_hot_labels


def process_texts(
    texts: List[List[str]], seq_len: int, word2idx: Dict[str, int]
) -> List[List[int]]:
    """Processes the texts, turning words to integers and padding them to the same length

    Args:
        texts (List[List[str]]): List of list of words (tokens)
        seq_len (int): Length to pad the sequences

    Returns:
        List[List[int]]: List of list of padded integers (words)

    """
    text_ints = []
    for text in texts:
        sentence = []
        for word in text:
            try:
                sentence.append(word2idx[word])
            except:
                # If we do not recognize the word, we will just append a 0 (padding)
                sentence.append(0)
        text_ints.append(sentence)

    texts = pad_features(text_ints, seq_len)

    return texts


def prepare_data_for_training(
    features: List[List[int]],
    labels: List[int] | List[List[int]],
    batch_size: int,
    split: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    """Prepares the data for training, creating TensorDatasets and Dataloaders
       The data is split into training, validation and test sets.

    Args:
        texts (List[List[int]]): List of tokenized sentences
        labels (List[int] | List[List[int]]): List of labels (sentiment analysis) or one-hot encoded labels (classification)
        seq_len (int): Length to pad the sequences
        split (float): Split between training and validation sets

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders
    """

    split_idx = int(len(features) * split)
    train_x, val_x = features[:split_idx], features[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_loader, val_loader


def prepare_data_for_testing(
    features: List[List[int]], labels: List[int], batch_size: int
) -> DataLoader:
    """Prepares the data for testing, creating a DataLoader

    Args:
        features (List[List[int]]): List of tokenized sentences
        labels (List[int]): List of labels (sentiment analysis) or one-hot encoded labels (classification)

    Returns:
        DataLoader: DataLoader containing the testing data
    """
    test_x = np.array(features)
    test_y = np.array(labels)

    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return test_loader
