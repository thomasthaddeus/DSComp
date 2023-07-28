"""text_preprocessing.py

_summary_

_extended_summary_

Returns:
    _type_: _description_
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
     _summary_

    _extended_summary_
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        clean_text _summary_

        _extended_summary_

        Args:
            text (_type_): _description_

        Returns:
            _type_: _description_
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

    # Count number of words
    def count_words(self, text):
        tokens = word_tokenize(text)
        return len(tokens)

    # Check if text contains a particular word
    def contains_word(self, text, word):
        tokens = word_tokenize(text.lower())
        return word.lower() in tokens

    def preprocess_csv(self, file_name):
        # Load the dataset
        df = pd.read_csv(file_name)

        # Assuming the column containing the text data is named 'text'
        # Replace 'text' with the actual column name in your CSV file
        df["processed_text"] = df["text"].map(self.clean_text)

        # Save the preprocessed data to a new CSV file
        df.to_csv("preprocessed_" + file_name, index=False)

    # Convert text to lowercase
    def to_lower(self, text):
        return text.lower()

    # Remove punctuation
    def remove_punctuation(self, text):
        return re.sub(r"[^\w\s]", "", text)

    # Remove stop words
    def remove_stop_words(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in stopwords.words("english")]
        return " ".join(tokens)

    # Additional use case 6: Replace a word in the text
    def replace_word(self, text, old_word, new_word):
        tokens = word_tokenize(text)
        tokens = [new_word if word == old_word else word for word in tokens]
        return " ".join(tokens)

    # Additional use case 7: Count number of occurrences of a word
    def count_word_occurrences(self, text, word):
        tokens = word_tokenize(text.lower())
        return tokens.count(word.lower())
