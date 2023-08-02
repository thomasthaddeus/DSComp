"""test_text_preprocessing.py"""

import pytest
from transformers import BertTokenizerFast
from language_models import TweetDataset
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def test_text_preprocessing():
    # Initialize a TweetDataset instance
    dataset = TweetDataset(
        [
            "This is a sample tweet. It has punctuation, uppercase letters, and stopwords!"
        ],
        [0],
        tokenizer,
    )

    # Get the preprocessed text
    preprocessed_text = dataset.texts[0]

    # Check if the text has been converted to lowercase
    assert preprocessed_text == preprocessed_text.lower()

    # Check if punctuation has been removed
    assert "." not in preprocessed_text
    assert "!" not in preprocessed_text

    # Check if stopwords have been removed
    assert "is" not in preprocessed_text
    assert "a" not in preprocessed_text
    assert "it" not in preprocessed_text
    assert "has" not in preprocessed_text
    assert "and" not in preprocessed_text

    # Check if the text has been lemmatized
    assert (
        "tweets" not in preprocessed_text
    )  # Assuming the lemma of "tweets" is "tweet"


# Run the test
pytest.main(["-v", "test_text_preprocessing.py"])
