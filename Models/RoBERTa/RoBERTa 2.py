import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, classification_report)
import time


# ========== 数据集封装类 ==========
class SarcasmDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class SarcasmTestDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings)


# ========== 无注意力的模型结构：只使用 RoBERTa/BERT 的 Embedding，不走 Transformer Block ==========
class NoAttentionBERT_Arch(nn.Module):
    """
    这里我们仅调用 `bert.embeddings` 来获取输入序列的 embedding 表示，
    之后做一个简单的平均池化，然后接全连接层完成分类。
    """
    def __init__(self, roberta_model):
        super(NoAttentionBERT_Arch, self).__init__()
        # 只保留 embeddings 层
        self.bert_embeddings = roberta_model.embeddings

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        # 下游分类器
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, mask):
        # 1) 仅取 embedding: shape (batch_size, seq_len, hidden_dim)
        x = self.bert_embeddings(input_ids)

        # 2) 简单平均池化 (不做attention)
        #    这里 mask 没有参与 attention，但可以用于加权平均或忽略padding位置
        #    目前示例直接做最简单的 mean pooling
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim)

        # 3) 送入下游分类器
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# ========== 评估指标函数 ==========
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred)
    return {"accuracy": accuracy, "f1_score": f1}


# ========== 手动训练/验证的函数定义 ==========
def train_one_epoch():
    model.train()
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and step != 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        sent_id, mask, labels = batch

        # 清空梯度
        model.zero_grad()

        # 前向
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)

        # 累计loss
        total_loss += loss.item()

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新参数
        optimizer.step()

        # 记录预测结果
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


def evaluate():
    print("\nEvaluating...")
    model.eval()
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(test_dataloader):
        if step % 50 == 0 and step != 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()

            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    avg_loss = total_loss / len(test_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


# ========== 主流程 ==========
if __name__ == '__main__':
    # 1. 读取数据
    df_train = pd.read_csv('Train_Dataset.csv')
    df_test = pd.read_csv('Test_Dataset.csv')

    X_train = df_train['tweet'].tolist()
    y_train = df_train['sarcastic'].tolist()
    X_test = df_test['tweet'].tolist()
    y_test = df_test['sarcastic'].tolist()

    # 2. 加载模型与分词器
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, num_labels=2)

    # 注意：AutoModel.from_pretrained(MODEL) 依旧是 RoBERTa，本身带有注意力层
    # 我们后面会只用它的 embeddings 层
    roberta_model = AutoModel.from_pretrained(MODEL)

    # 3. 数据编码
    # （简单处理，保留了原先的 batch_encode_plus 逻辑）
    tokens_train = tokenizer.batch_encode_plus(
        X_train,
        max_length=25,
        padding='max_length',
        truncation=True
    )
    tokens_test = tokenizer.batch_encode_plus(
        X_test,
        max_length=25,
        padding='max_length',
        truncation=True
    )

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(y_train)

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(y_test)

    # 4. DataLoader
    batch_size = 32

    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_seq, test_mask, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # 5. 冻结预训练权重（可选）
    #    如果您只想训练下游分类头，可以保持对 embeddings 也 freeze；或者只冻结Transformer层
    for param in roberta_model.parameters():
        param.requires_grad = False

    # 6. 实例化 NoAttentionBERT_Arch
    model = NoAttentionBERT_Arch(roberta_model)

    # 7. 定义优化器
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # 8. 类别权重 (可选)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df_train['sarcastic']),
        y=df_train['sarcastic']
    )
    weights = torch.tensor(class_weights, dtype=torch.float)
    cross_entropy = nn.NLLLoss(weight=weights)

    # 9. 训练循环
    epochs = 5
    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        train_loss, _ = train_one_epoch()
        valid_loss, _ = evaluate()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights_no_attention.pt')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    # 10. 加载最优权重并评估
    model.load_state_dict(torch.load('saved_weights_no_attention.pt'))

    with torch.no_grad():
        preds = model(test_seq, test_mask)  # (batch_size, 2)
        preds = preds.cpu().numpy()

    preds = np.argmax(preds, axis=1)
    print("Classification Report (No-Attention Model):")
    print(classification_report(test_y, preds))
    print("F1 Score:", f1_score(test_y, preds))
