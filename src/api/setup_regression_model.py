# Import necessary libraries
import os
from datetime import timedelta

import pandas as pd
import random
import spacy
import torch
import wandb
from wandb import AlertLevel
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

!pip install wandb
!wandb login


# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load the data
tst_df = pd.read_csv("/kaggle/input/test.csv")
trn_df = pd.read_csv("/kaggle/input/train.csv")

# Load the user secrets
WANDB_API_KEY = os.environ["WANDB_API_KEY"]

## Data Preprocessing
# Identify categorical and numerical columns
categorical_features = ["keyword"]
numerical_features = ["some_numerical_column"]

# Create a transformer for numerical features
numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

# Create a transformer for categorical features
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combine the transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Apply the transformations to the training data
trn_df = preprocessor.fit_transform(trn_df)

# Apply the transformations to the test data
tst_df = preprocessor.transform(tst_df)

# Handle missing values in the data
trn_df = trn_df.fillna("None")
tst_df = tst_df.fillna("None")

trn_txt = trn_df["text"].tolist()
trn_lbl = trn_df[["content", "wording"]].values.tolist()
trn_txt, tst_txt, trn_lbl, tst_lbl = train_test_split(trn_txt, trn_lbl, test_size=0.2)


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()


class Tknz(Dataset):
    """
    Tokenizer class for preprocessing text data.

    This class handles the tokenization and preprocessing of text data
    for use with BERT models.

    Args:
        texts (list): List of text strings to be tokenized.
        labels (list): Corresponding labels for the text data.
        tokenizer (Tokenizer): Tokenizer object for encoding the text.
        max_len (int): Maximum length for the encoded text.
    """

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
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Check if we have labels
        if self.labels is not None:
            label = self.labels[idx]
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long),
            }
        else:  # Return only inputs for test data
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
            }

    def preprocess_text(self, texts):
        """
        preprocess_text _summary_

        _extended_summary_

        Args:
            texts (_type_): _description_

        Returns:
            _type_: _description_
        """
        preprocessed_texts = []
        for text in texts:
            # Apply SpaCy pipeline on the text
            doc = nlp(text)
            # Lemmatize the text and join the words back into a single string
            lemma_text = " ".join([token.lemma_ for token in doc])
            preprocessed_texts.append(lemma_text)
        return preprocessed_texts


# Compute Metrics for Regression
def compute_metrics(eval_pred: EvalPrediction):
    """
    compute_metrics _summary_

    _extended_summary_

    Args:
        eval_pred (EvalPrediction): _description_

    Returns:
        _type_: _description_
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions
    mse_content = mean_squared_error(labels[:, 0], preds[:, 0])
    mse_wording = mean_squared_error(labels[:, 1], preds[:, 1])
    return {"mse_content": mse_content, "mse_wording": mse_wording}


# Initialize the datasets
test_dataset = Tknz(tst_txt)
train_dataset = Tknz(trn_txt, trn_lbl, train_test_split())


## Model Creation and Training
# --------------------------------
# Load the model

# Define the configuration
# 2 labels for content and wording
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)

# Create the model with the defined configuration
model = BertForSequenceClassification(config)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="../results",  # output directory
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # function to compute metrics
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
    ],  # Stop if validation loss doesn't improve for 3 evaluations
)

threshold = 20
acc =
if acc < threshold:
    wandb.alert(
        title="Low accuracy",
        text=f"Accuracy {acc} is below the acceptable threshold {threshold}",
        level=AlertLevel.WARN,
        wait_duration=timedelta(minutes=5),
    )

# Train the model
trainer.train()


## Submission Section
# -------------------------------------

# Making predictions on test data
test_predictions = trainer.predict(test_dataset)

# Prepare a DataFrame with student_id and predictions
submission_df = pd.DataFrame(
    {
        "student_id": tst_df["student_id"],
        "content": test_predictions.predictions[:, 0],
        "wording": test_predictions.predictions[:, 1],
    }
)

# Save the predictions to a CSV file
submission_df.to_csv("submission.csv", index=False)
