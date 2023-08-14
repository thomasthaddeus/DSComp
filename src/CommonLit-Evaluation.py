# CommonLit-Evaluation

# Import necessary libraries
import os
import pandas as pd
import spacy
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
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

# Load the data
tst_df = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_test.csv")
trn_df = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv")

# Preprocessing
class Tknz(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = self.preprocess_text(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
​
    def __len__(self):
        return len(self.texts)
​
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
​
    def preprocess_text(self, texts):
        preprocessed_texts = []
        for text in texts:
            # Apply SpaCy pipeline on the text
            doc = nlp(text)
            # Lemmatize the text and join the words back into a single string
            lemma_text = " ".join([token.lemma_ for token in doc])
            preprocessed_texts.append(lemma_text)
        return preprocessed_texts

​# Compute Metrics for Regression
def compute_metrics(eval_pred: EvalPrediction):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions
    mse_content = mean_squared_error(labels[:, 0], preds[:, 0])
    mse_wording = mean_squared_error(labels[:, 1], preds[:, 1])
    return {"mse_content": mse_content, "mse_wording": mse_wording}

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

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
​
# Apply the transformations to the test data
tst_df = preprocessor.transform(tst_df)

# Handle missing values in the data
trn_df = trn_df.fillna("None")
tst_df = tst_df.fillna("None")
​
trn_txt = trn_df["text"].tolist()
trn_lbl = trn_df[["content", "wording"]].values.tolist()
trn_txt, tst_txt, trn_lbl, tst_lbl = train_test_split(trn_txt, trn_lbl, test_size=0.2)

# Initialize the datasets
test_dataset = Tknz(tst_txt)
train_dataset = Tknz(trn_txt, trn_lbl, train_test_split())

## Model Creation and Training

# Define the configuration
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

## Submission Section
​# Making predictions on test data
test_predictions = trainer.predict(test_dataset)
​
# Prepare a DataFrame with student_id and predictions
submission_df = pd.DataFrame(
    {
        "student_id": tst_df["student_id"],
        "content": test_predictions.predictions[:, 0],
        "wording": test_predictions.predictions[:, 1],
    }
)
​
# Save the predictions to a CSV file
submission_df.to_csv("submission.csv", index=False)
