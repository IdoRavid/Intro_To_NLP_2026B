###################################################
# Exercise 2 - Natural Language Processing 67658  #
###################################################

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'}

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 16
N_FEATURES = 2000


def get_data(categories=None, portion=1.):
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


class LogLinear(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def MLP_classification(portion=1., model=None):
    from sklearn.feature_extraction.text import TfidfVectorizer

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    vectorizer = TfidfVectorizer(max_features=N_FEATURES)
    X_train = torch.tensor(vectorizer.fit_transform(x_train).toarray().astype(np.float32))
    X_test  = torch.tensor(vectorizer.transform(x_test).toarray().astype(np.float32))
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_accs = [], []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        train_losses.append(epoch_loss / len(x_train))

        model.eval()
        with torch.no_grad():
            acc = (model(X_test).argmax(dim=1) == y_test_t).float().mean().item()
        val_accs.append(acc)
        print(f"[portion={portion}] Epoch {epoch+1}/{EPOCHS}  loss={train_losses[-1]:.4f}  val_acc={acc:.4f}")

    return train_losses, val_accs


# Q3
def transformer_classification(portion=1.):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import evaluate
    from tqdm import tqdm

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev='cpu'):
        model.train()
        total_loss = 0.
        for batch in tqdm(data_loader):
            input_ids      = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels         = batch['labels'].to(dev)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate_model(model, data_loader, dev='cpu', metric=None):
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids      = batch['input_ids'].to(dev)
                attention_mask = batch['attention_mask'].to(dev)
                labels         = batch['labels'].to(dev)
                preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=1)
                metric.add_batch(predictions=preds, references=labels)
        return metric.compute()

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels  = len(category_dict)
    epochs      = 3
    batch_size  = 16
    learning_rate = 5e-5

    model     = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    metric    = evaluate.load("accuracy")

    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    val_dataset   = Dataset(tokenizer(x_test,  truncation=True, padding=True), y_test)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses, val_accs = [], []
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, dev)
        result = evaluate_model(model, val_loader, dev, metric)
        train_losses.append(loss)
        val_accs.append(result['accuracy'])
        print(f"[portion={portion}] Epoch {epoch+1}/{epochs}  loss={loss:.4f}  val_acc={result['accuracy']:.4f}")

    return train_losses, val_accs


def plot_results(all_results, model_name):
    """all_results: dict {portion: (train_losses, val_accs)}"""
    epochs = range(1, len(next(iter(all_results.values()))[0]) + 1)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    slug = model_name.lower().replace(' ', '_')

    fig, ax = plt.subplots()
    for (portion, (losses, _)), color in zip(all_results.items(), colors):
        ax.plot(epochs, losses, label=f"portion={portion}", color=color)
    ax.set_title(f"{model_name} Train Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{slug}_loss.png")); plt.close()

    fig, ax = plt.subplots()
    for (portion, (_, accs)), color in zip(all_results.items(), colors):
        ax.plot(epochs, accs, label=f"portion={portion}", color=color)
    ax.set_title(f"{model_name} Validation Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{slug}_acc.png")); plt.close()


if __name__ == "__main__":
    portions = [0.1, 0.2, 0.5, 1.]

    # Q1 - Log-Linear (single layer)
    print("=== Q1: Log-Linear ===")
    results = {p: MLP_classification(portion=p, model=LogLinear(N_FEATURES, len(category_dict))) for p in portions}
    plot_results(results, "Log-Linear")

    # Q2 - MLP (one hidden layer)
    print("\n=== Q2: MLP ===")
    results = {p: MLP_classification(portion=p, model=MLP(N_FEATURES, 500, len(category_dict))) for p in portions}
    plot_results(results, "MLP")

    # Q3 - Transformer
    print("\n=== Q3: Transformer ===")
    transformer_results = {}
    for p in portions[:2]:
        print(f"Portion: {p}")
        transformer_results[p] = transformer_classification(portion=p)
    plot_results(transformer_results, "Transformer")
