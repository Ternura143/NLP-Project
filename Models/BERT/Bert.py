import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

warnings.filterwarnings('ignore')

# 数据集类定义
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
        return len(next(iter(self.encodings.values())))

# 评估指标函数
def compute_metrics(p):
    pred = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred)

    return {"accuracy": accuracy, "f1_score": f1}

if __name__ == '__main__':
    # 数据路径
    path = 'Train_Dataset.csv'
    path_test = 'Test_Dataset.csv'

    # 加载数据
    df = pd.read_csv(path)
    test = pd.read_csv(path_test)

    # 确保没有空值
    df = df.dropna(subset=['tweet'])


    # 提取训练数据
    train_tweets = df['tweet'].values.tolist()
    train_labels = df['sarcastic'].values.tolist()
    test_tweets = test['tweet'].values.tolist()

    # 划分训练集和验证集
    train_tweets, val_tweets, train_labels, val_labels = train_test_split(
        train_tweets, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 编码数据
    train_encodings = tokenizer(train_tweets, padding=True, truncation=True, max_length=512)
    val_encodings = tokenizer(val_tweets, padding=True, truncation=True, max_length=512)
    test_encodings = tokenizer(test_tweets, padding=True, truncation=True, max_length=512)

    # 数据集
    train_dataset = SarcasmDataset(train_encodings, train_labels)
    val_dataset = SarcasmDataset(val_encodings, val_labels)
    test_dataset = SarcasmTestDataset(test_encodings)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        seed=0,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=100,
    )

    # 初始化模型
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # 定义 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    trainer.train()

    # 模型评估
    trainer.evaluate()

    # 预测
    preds = trainer.predict(test_dataset=test_dataset)
    probs = torch.from_numpy(preds.predictions).softmax(1).numpy()
    predictions = np.argmax(probs, axis=1)

    # 保存结果
    test['sarcastic_result'] = predictions
    print(f"F1 Score: {f1_score(test['sarcastic'], test['sarcastic_result'])}")

    # 输出结果到文件
    test[['tweet', 'sarcastic_result']].to_csv('test_results.csv', index=False)

