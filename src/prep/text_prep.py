"""text_preprocessing.py

This module contains the TextProcessor class for text preprocessing tasks like
cleaning and lemmatizing.

The TextProcessor class uses the NLTK library to perform text cleaning tasks
such as removing special characters, single characters, and applying
lemmatization to words. It is designed to prepare text data for natural
language processing tasks.

Returns:
    TextProcessor: An instance of the TextProcessor class.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


class TextProcessor:
    """
     A class used to preprocess text data.

    This class provides utility functions for cleaning and lemmatizing text
    data. It uses the NLTK library for tokenization, stopword removal, and
    lemmatization.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Cleans the input text by removing special characters and single
        characters.


        This method performs several cleaning tasks:
        - It removes all special characters from the text.
        - It removes single characters.
        - It removes single characters from the start of the text.
        - It substitutes multiple spaces with a single space.
        - Finally, it converts all text to lowercase.

        Args:
            text (str): The text string to be cleaned.

        Returns:
            str: The cleaned text string.
        """

        # Remove all the special characters
        processed_feature = re.sub(r"\W", " ", str(text))

        # remove all single characters
        processed_feature = re.sub(r"\s+[a-zA-Z]\s+", " ", processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r"\^[a-zA-Z]\s+", " ", processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r"\s+", " ", processed_feature, flags=re.I)

        # Removing prefixed 'b'
        processed_feature = re.sub(r"^b\s+", "", processed_feature)

        # Converting to Lowercase
        processed_feature = processed_feature.lower()

        # Lemmatization
        tokens = word_tokenize(processed_feature)
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stopwords.words("english")
        ]
        return " ".join(tokens)


    def count_words(self, text):
        """
        Counts the number of words in the input text.

        This method tokenizes the input text using NLTK's word_tokenize and
        returns the number of tokens.

        Args:
            text (str): The text string to be counted.

        Returns:
            int: The number of words in the text.
        """
        tokens = word_tokenize(text)
        return len(tokens)


    def contains_word(self, text, word):
        """
        Checks if the input text contains a specific word.

        This method tokenizes the input text and checks if the specified word
        is in the list of tokens.

        Args:
            text (str): The text string to be checked.
            word (str): The word to look for in the text.

        Returns:
            bool: True if the word is in the text, False otherwise.
        """
        tokens = word_tokenize(text.lower())
        return word.lower() in tokens


    def preprocess_csv(self, file_name):
        """
        Preprocesses a CSV file by cleaning the text in a specified column.

        This method reads a CSV file into a pandas DataFrame, applies the
        clean_text method to a column named 'text', and saves the resulting
        DataFrame to a new CSV file.

        Args:
            file_name (str): The path of the CSV file to be preprocessed.
        """
        # Load the dataset
        df = pd.read_csv(file_name)

        # Assuming the column containing the text data is named 'text'
        # Replace 'text' with the actual column name in your CSV file
        df["processed_text"] = df["text"].map(self.clean_text)

        # Save the preprocessed data to a new CSV file
        df.to_csv("preprocessed_" + file_name, index=False)

    @classmethod
    def to_lower(cls, text):
        """
        Converts the input text to lowercase.

        This method applies Python's lower() function to the input string.

        Args:
            text (str): The text string to be converted.

        Returns:
            str: The lowercase version of the input text.
        """
        return text.lower()

    # Remove punctuation
    def remove_punctuation(self, text):
        """
        Removes punctuation from the input text.

        This method uses a regular expression to remove all characters that are
        not alphanumeric or whitespace.

        Args:
            text (str): The text string to be processed.

        Returns:
            str: The input text with all punctuation removed.
        """
        return re.sub(r"[^\w\s]", "", text)

    # Remove stop words
    def remove_stop_words(self, text):
        """
        Removes English stop words from the input text.

        This method tokenizes the input text, removes any tokens that are
        English stop words according to NLTK, and then joins the tokens back
        together into a string.

        Args:
            text (str): The text string to be processed.

        Returns:
            str: The input text with all English stop words removed.
        """
        tokens = word_tokenize(text.lower())
        tokens = [i for i in tokens if i not in stopwords.words("english")]
        return " ".join(tokens)

    def replace_word(self, text, old_word, new_word):
        """
        Replaces occurrences of a specific word in the input text.

        This method tokenizes the input text, replaces any tokens that match
        old_word with new_word, and then joins the tokens back together into a
        string.

        Args:
            text (str): The text string to be processed.
            old_word (str): The word to be replaced.
            new_word (str): The word to replace old_word with.

        Returns:
            str: The input text with old_word replaced by new_word.
        """
        tokens = word_tokenize(text)
        tokens = [new_word if i == old_word else i for i in tokens]
        return " ".join(tokens)

    def count_word_occurrences(self, text, word):
        """
        Counts the number of occurrences of a specific word in the input text.

        This method tokenizes the input text and counts the number of tokens
        that match the specified word.

        Args:
            text (str): The text string to be processed.
            word (str): The word to count occurrences of.

        Returns:
            int: The number of occurrences of word in the text.
        """
        tokens = word_tokenize(text.lower())
        return tokens.count(word.lower())
