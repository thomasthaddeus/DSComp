"""
_summary_

Data loading and preprocessing: It loads your train and test data from CSV files, splits the training data into a train and validation set, and wraps them in a TweetDataset class instance, which tokenizes the texts and prepares them for input into the BERT model.

Model setup: It loads the pretrained BERT model from the transformers library, sets it up for binary classification, and initializes a Trainer instance with this model, the training arguments, and the train and validation datasets.

Training: It trains the model for the specified number of epochs, saving the model after each epoch and logging the progress. The best model according to the validation performance is then loaded at the end of training.

Prediction: It makes predictions on the test set using the trained model, selecting the class with the highest predicted probability for each instance.

Output: Finally, it outputs the test predictions to a CSV file, with each row containing an ID and the corresponding predicted label.

Returns:
    _type_: _description_

Data Fields:
    train: id,keyword,location,text,target
    test:  id,keyword,location,text
    submission: id,target

Example Data Fields:
    train: 9431,survivors,,Û÷Faceless body belonged to my sisterÛª: #Hiroshima #Nagasaki #nuke survivors recall horrors 70 years on ÛÓ RT News http://t.co/918EQmTkrL,1
    9432,survivors,Shanghai,Survivors of Shanghai Ghetto reunite after 70 years - http://t.co/1Ki8LgVAy4 #Shanghai #China #??,0
    9434,survivors,Upstairs.,People with netflix there's a really good documentary about Hiroshima narrated by John Hurt. 2 Parter that interviews Pilots + Survivors.,0
    9435,survivors,Anywhere Safe,@LawfulSurvivor T-Dog had been holed up in an apartment store with several other survivors Glenn Morales Andrea Jacqui and Merle.--,1
    9436,survivors,"Marietta, GA",Stemming from my #Cubs talk- the team rosters 2 cancer survivors in @ARizzo44 &amp; @JLester34...@Cubs fans: help another http://t.co/XGnjgLE9eQ,1
    9438,survivors,,Haunting memories drawn by survivors http://t.co/WJ7UjFs8Fd,1
    test: 69,ablaze,threeonefive. ,beware world ablaze sierra leone &amp; guap.
    submission: 0,0
                2,0
                3,0
                9,0
"""


import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import EvalPrediction
import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = self.preprocess_text(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Check if we have labels
        if self.labels is not None:
            label = self.labels[idx]
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.Tensor(label, dtype=torch.long),
            }
        else:  # Return only inputs for test data
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
            }

    def preprocess_text(self, texts):
        preprocessed_texts = []
        for text in texts:
            # Apply SpaCy pipeline on the text
            doc = nlp(text)
            # Lemmatize the text and join the words back into a single string
            lemma_text = " ".join([token.lemma_ for token in doc])
            preprocessed_texts.append(lemma_text)
        return preprocessed_texts