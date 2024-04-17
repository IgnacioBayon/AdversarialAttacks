from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def tokenize(text: str) -> List[str]:
    """
    Tokenizes the input text by replacing punctuation with tokens

    Args:
        text (str): The input text to tokenize.

    Returns:
        List[str]: A list of words that have been tokenized
    """
    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()

    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')

    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return words


def load_and_preprocess_data(data_path: str) -> List[List[str]]:
    """
    Load and preprocess the data from the given path.

    Args:
        data_path(str): The path to the data file.

    Returns:
        List[List[str]]: A list of tokenized words from the data file.
        List[int]: A list of sentiment labels for each review.
    """
    sentences = []
    labels = []
    with open(data_path, 'r', encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            review, sentiment = line[1:-11], line[-9:-1]
            tokens = tokenize(review)
            sentences.append(tokens)
            sentiment = 1 if sentiment == 'positive' else 0
            labels.append(sentiment)

    return sentences, labels


def create_lookup_tables(sentences: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    # Count the word frequency
    words = [word for sentence in sentences for word in sentence]
    word_counts: Counter = Counter(words)
    # Sorting the words from most to least frequent in text occurrence.
    sorted_vocab: List[int] = sorted(
        word_counts, key=word_counts.get, reverse=True)

    # Create int_to_vocab and vocab_to_int dictionaries.
    int_to_vocab: Dict[int, str] = {
        i+1: word for i, word in enumerate(sorted_vocab)}
    vocab_to_int: Dict[str, int] = {
        word: i+1 for i, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


class SentimentAnalysisDataset(Dataset):
    def __init__(self, sentences: List[List[int]], labels: List[int], word_to_int: Dict[str, int], int_to_word: Dict[int, str]):
        """
        Initialize the SentimentAnalysisDataset.

        Args:
            sentences (List[List[int]]): A list of tokenized sentences.
            labels (List[int]): A list of sentiment labels.
        """
        self.sentences = sentences
        self.labels = labels
        self.word_to_int = word_to_int
        self.int_to_word = int_to_word

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    def get_word_to_int(self, word: str) -> int:
        return self.word_to_int[word]

    def get_int_to_word(self, idx: int) -> str:
        return self.int_to_word[idx]


def generate_data_loader(data_path: str, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, int, Dict[str, int], Dict[int, str]]:
    """
    Generate the data loaders for the training and test sets.

    Args:
        data_path (str): The path to the data file.
        batch_size (int): The batch size to use for the data loaders.

    Returns:
        Tuple[DataLoader, DataLoader, int, Dict[str, int], Dict[int, str]]: A tuple containing the training DataLoader,
        the test DataLoader, the vocabulary size, the vocab_to_int dictionary, and the int_to_vocab dictionary.
    """
    # Load and preprocess the data
    sentences, labels = load_and_preprocess_data(data_path)

    # Create lookup tables
    vocab_to_int, int_to_vocab = create_lookup_tables(sentences)
    vocab_size = len(vocab_to_int) + 1

    sentences = [
        [vocab_to_int[word] for word in sentence] for sentence in sentences
    ]
    
    # We choose the sentence length as the 95th percentile of the sentence length
    sentence_length = sorted([len(sentence) for sentence in sentences])[int(0.95 * len(sentences))]

    # Pad the sentences to the sentence length
    for i, sentence in enumerate(sentences):
        if len(sentence) < sentence_length:
            sentences[i] = sentence + [0] * (sentence_length - len(sentence))
        else:
            sentences[i] = sentence[:sentence_length]

    sentences = torch.tensor(sentences)
    labels = torch.tensor(labels)

    dataset = SentimentAnalysisDataset(
        sentences, labels, vocab_to_int, int_to_vocab)

    # Split the data into training, validation, and test sets
    # 80% train, 10% validation, 10% test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    # Create the DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, vocab_size, vocab_to_int, int_to_vocab, sentence_length
