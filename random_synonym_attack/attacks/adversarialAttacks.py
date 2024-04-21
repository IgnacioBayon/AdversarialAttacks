from textblob import TextBlob
from nltk.corpus import wordnet

from typing import List
from random import random


CHANGE_RATE = 0.5


def synonym(word: str) -> str:
    """Given a word, returns a synonym

    Args:
        word (str): word to be changed

    Returns:
        str: synonym of the word
    """

    synonyms: List[str] = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())

    synonyms = set(synonyms)

    output: str = None
    for synonym in synonyms:
        if "-" not in synonym or "_" not in synonym:
            output = synonym
            return output

    # If no synonym was found, return the original word
    return word


def synonym_attack(texts: List[str]) -> List[str]:
    """Given a text, changes some words into their synonyms

    Args:
        texts (List[str]): list of sentences

    Returns:
        List[str]: altered list of sentences
    """
    changed_texts = []
    for text in texts:
        blob = TextBlob(text).tags
        words = text.split()

        for i, word_with_tag in enumerate(blob):
            if word_with_tag[1][0] in ["N", "V", "J"] and random() < CHANGE_RATE:
                try:
                    words[i] = synonym(words[i])
                except:
                    pass
        new_text = " ".join(words)

        changed_texts.append(new_text)

    return changed_texts


if __name__ == "__main__":
    texts = [
        "I really enjoyed this film, it is my favourite",
    ]
    print(synonym_attack(texts))


def generate_synonym_texts(texts: List[List[str]]) -> List[List[str]]:
    """Given a list of texts, changes some words into their synonyms

    Args:
        texts (List[List[str]]): list of sentences

    Returns:
        List[List[str]]: altered list of sentences
    """
    changed_texts = []
    for text in texts:
        changed_texts.append(synonym_attack(text))

    return changed_texts
