"""tokenize_chunk_count.py

This module provides functions to tokenize text into sentences and words, and
to count the most common noun and verb phrase chunks in a set of sentences.
"""

from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize


def word_sentence_tokenize(text):
    """
    Tokenizes the input text into sentences and words.

    Args:
        text (str): The text to be tokenized.

    Returns:
        list: A list of lists, where each inner list contains the words of a
        sentence.
    """
    s_tokenizer = PunktSentenceTokenizer(text)
    j = s_tokenizer.tokenize(text)
    return [word_tokenize(i) for i in j]


def count_chunks(chnks, label):
    """
    Counts the most common chunks in a set of sentences, filtering by a
    specific label.

    Args:
        chnks (list): The sentences to be analyzed.
        label (str): The label to filter chunks by.

    Returns:
        list: A list of tuples, where each tuple contains a chunk and its
          count. The list is sorted by count in descending order.
    """
    chunks = [tuple(subtree) for i in chnks
              for subtree in i.subtrees(filter=lambda t: t.label() == label)]
    chunk_counter = Counter(chunks)
    return chunk_counter.most_common(30)
