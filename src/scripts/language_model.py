import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load the tokenizer
my_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

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

# rest of the code remains the same...



def compute_metrics(eval_pred: EvalPrediction):
    labels = eval_pred.label_ids
    preds = np.argmax(eval_pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Load the CSV data
train_df = pd.read_csv('../data/twitter/train.csv')
test_df = pd.read_csv('../data/twitter/test.csv')

# Extract 'text' as input and 'target' as labels from the train data
train_texts = train_df['text'].tolist()
train_labels = train_df['target'].tolist()

# For the test data, we only need the 'text' column
test_texts = test_df['text'].tolist()

# Split the train data into train and validation subsets
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

# Initialize the datasets
train_dataset = TweetDataset(train_texts, train_labels, my_tokenizer)
val_dataset = TweetDataset(val_texts, val_labels, my_tokenizer)
test_dataset = TweetDataset(test_texts, None, my_tokenizer)  # No labels for the test data

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # num_labels=2 for binary classification

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

training_args = TrainingArguments(
    output_dir='../models/results',
    num_train_epochs=3,  # reduce from 5 to 3
    per_device_train_batch_size=32,  # increase from 16 to 32
    gradient_accumulation_steps=2,  # new line for gradient accumulation
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=3e-5,  # increase from 2e-5 to 3e-5
    logging_dir='../models/logs',
    logging_steps=10,  # Log every 10 steps
    evaluation_strategy="epoch",  # Evaluation and Save happens at every epoch
    save_strategy="epoch",  # Save the model after every epoch
    load_best_model_at_end=True,
    fp16=True,  # new line for mixed precision training
)

# ### Split the data into training and validation sets

# Initialize the Trainer with the correct datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Use validation dataset for evaluation
    compute_metrics=compute_metrics,
)

trainer.train()

# ### Submission Output to CSV file

# Making predictions on test data
test_predictions = trainer.predict(test_dataset)

# We take the output class with the highest probability
test_preds = np.argmax(test_predictions.predictions, axis=1)

# Prepare a DataFrame with test IDs and predictions
submission_df = pd.DataFrame({
    'id': test_df['id'],  # Assuming that 'test_df' is the DataFrame with your test data
    'target': test_preds
})

# Save the DataFrame into a CSV file
submission_df.to_csv('submission.csv', index=False)
