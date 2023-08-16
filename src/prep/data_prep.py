"""data_prep.py

This module contains functions for data preparation tasks like chunk counter.

The functions in this module use the NLTK library to perform tasks like
chunking sentences and finding the most common chunks. It is designed to
prepare data for natural language processing tasks.

Returns:
    Various: Depending on the function called, could return different types
    such as list or Counter.

Examples:
>>> data_prep = DataPrep()

>>> # Now call the methods
>>> data_prep.np_chunk_counter(my_chunked_sentences)
>>> data_prep.vp_chunk_counter(my_chunked_sentences)
>>> data_prep.word_sentence_tokenize(my_text)

>>> DataPrep.np_chunk_counter(my_chunked_sentences)
>>> DataPrep.vp_chunk_counter(my_chunked_sentences)
>>> DataPrep.word_sentence_tokenize(my_text)
"""

from collections import Counter
from typing import Any
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize


class DataPrep:
    """
    This class contains methods for data preparation tasks like chunk counter.

    The methods in this class use the NLTK library to perform tasks like
    chunking sentences and finding the most common chunks. It is designed to
    prepare data for natural language processing tasks.
    """
    def __init__(self, text):
        self.text = text

    def __setattr__(self, __name: str, __value: Any) -> None:
        self._name = __name
        self._value = __value

    def main(self):
        """
        main _summary_

        _extended_summary_

        Returns:
            _type_: _description_
        """
        data_prep = DataPrep(self.text)
        data_prep.np_chunk_counter(self.chunked_sentences)
        data_prep.vp_chunk_counter(self.chunked_sentences)
        data_prep.word_sentence_tokenize(self.text)


    @staticmethod
    def np_chunk_counter(chunked_sentences):
        """
        Counts the occurrences of noun phrase (NP) chunks in chunked sentences.

        This function extracts NP chunks from chunked sentences and counts their
        occurrences using a Counter. It returns the Counter object, which maps each
        NP chunk to its number of occurrences.

        Args:
            chunked_sentences (list): A list of chunked sentences from which to
            extract NP chunks.

        Returns:
            collections.Counter: A Counter object mapping each NP chunk to its
            number of occurrences.
        """
        chunks = []

        # for-loop through each chunked sentence to extract noun phrase chunks
        for chnkd_sntc in chunked_sentences:
            for subtree in chnkd_sntc.subtrees(filter=lambda t: t.label() == "NP"):
                chunks.append(tuple(subtree))

        # create a Counter object
        chunk_counter = Counter()

        # for-loop through the list of chunks
        for chunk in chunks:
            # increase counter of specific chunk by 1
            chunk_counter[chunk] += 1

        # return 30 most frequent chunks
        return chunk_counter.most_common(30)

    @staticmethod
    def vp_chunk_counter(chunked_sentences):
        """
        Counts the occurrences of verb phrase (VP) chunks in chunked sentences.

        This function extracts VP chunks from chunked sentences and counts their
        occurrences using a Counter. It returns the Counter object, which maps
        each VP chunk to its number of occurrences.

        Args:
            chunked_sentences (list): A list of chunked sentences from which to
            extract VP chunks.

        Returns:
            collections.Counter: A Counter object mapping each VP chunk to its
            number of occurrences.
        """
        chunks = []

        # for-loop through each chunked sentence to extract verb phrase chunks
        for chnkd_sntc in chunked_sentences:
            for subtree in chnkd_sntc.subtrees(filter=lambda t: t.label() == "VP"):
                chunks.append(tuple(subtree))

        # create a Counter object
        chunk_counter = Counter()

        # for-loop through the list of chunks
        for chunk in chunks:
            # increase counter of specific chunk by 1
            chunk_counter[chunk] += 1

        # return 30 most frequent chunks
        return chunk_counter.most_common(30)

    @staticmethod
    def word_sentence_tokenize(text):
        """
        Tokenizes the text into words and sentences.

        This function uses NLTK's PunktSentenceTokenizer to split the input text
        into sentences and then tokenizes each sentence into words. The result is a
        list of word-tokenized sentences.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list: A list of word-tokenized sentences.
        """
        sentence_tokenizer = PunktSentenceTokenizer(text)
        sentence_tokenized = sentence_tokenizer.tokenize(text)
        word_tokenized = list()

        # for-loop through each tokenized sentence in sentence_tokenized
        for tokenized_sentence in sentence_tokenized:
            # word tokenize each sentence and append to word_tokenized
            word_tokenized.append(word_tokenize(tokenized_sentence))

        return word_tokenized
