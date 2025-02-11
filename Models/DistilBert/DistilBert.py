import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SarcasmTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred)

    return {"accuracy": accuracy, "f1_score": f1}

if __name__ == '__main__':
    # Load datasets
    path = 'Train_Dataset.csv'
    path_test = 'Test_Dataset.csv'

    df = pd.read_csv(path)
    test = pd.read_csv(path_test)
    df = df.dropna(subset=['tweet'])

    train_tweets = df['tweet'].values.tolist()
    train_labels = df['sarcastic'].values.tolist()
    test_tweets = test['tweet'].values.tolist()
    test_labels = test['sarcastic'].values.tolist()

    train_tweets, val_tweets, train_labels, val_labels = train_test_split(
        train_tweets, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    train_encodings = tokenizer(train_tweets, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_tweets, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_tweets, truncation=True, padding=True, max_length=512)

    train_dataset = SarcasmDataset(train_encodings, train_labels)
    val_dataset = SarcasmDataset(val_encodings, val_labels)
    test_dataset = SarcasmTestDataset(test_encodings)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./res',
        evaluation_strategy="steps",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs4',
        load_best_model_at_end=True,
    )

    # Model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Predict on test set
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    predicted_labels = np.argmax(logits, axis=1)

    # Calculate F1-score
    f1 = f1_score(test_labels, predicted_labels)
    print(f"Final Test F1-score: {f1}")

    test['sarcastic_result'] = predicted_labels
    test[['tweet', 'sarcastic_result']].to_csv('test_results2.csv', index=False)
    print("预测结果已保存到 test_results2.csv")
