# train_bert_small.py
"""
Курсовая работа: Fine-tuning prajjwal1/bert-small на датасете IMDB.

Требования:
1. Структурированный код
2. Бейзлайн-модель (TF-IDF + Logistic Regression)
3. Оценка на сложных данных (двойные отрицания)
4. Визуализация и сохранение всех артефактов в logs/
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import matplotlib.pyplot as plt
import re


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)


def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path)
    texts = df['bert_input'].tolist()
    labels = [0 if x == 'positive' else 1 for x in df['sentiment']]
    return texts, labels


def create_baseline_model(X_train, y_train, X_test, y_test):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred


def contains_double_negation(text):
    text_lower = text.lower()
    neg_words = ['not', 'no', 'never', 'nobody', 'nothing']
    return sum(word in text_lower for word in neg_words) >= 2


def evaluate_on_hard_cases(model, tokenizer, texts, labels, device, batch_size=32):
    """
    Оценка на сложных случаях с оптимизированным DataLoader.
    """
    hard_idx = [i for i, t in enumerate(texts) if contains_double_negation(t)]
    if not hard_idx:
        return None

    hard_texts = [texts[i] for i in hard_idx]
    hard_labels = [labels[i] for i in hard_idx]

    enc = tokenizer(
        hard_texts,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    ds = TensorDataset(enc['input_ids'], enc['attention_mask'], torch.tensor(hard_labels))
    # ✅ Ускоренная загрузка данных
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            labels_batch = batch[2]
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_batch.tolist())

    return accuracy_score(all_labels, all_preds)


def plot_confusion_matrix(y_true, y_pred, log_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (BERT-small)")
    plt.savefig(os.path.join(log_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history, log_dir):
    """
    ✅ Безопасное построение графиков: принудительное приведение к float.
    """
    train_loss = [float(x) for x in history['train_loss']]
    val_loss = [float(x) for x in history['val_loss']]
    train_acc = [float(x) for x in history['train_acc']]
    val_acc = [float(x) for x in history['val_acc']]

    epochs = range(1, len(train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, label='Train Loss', marker='o')
    axes[0].plot(epochs, val_loss, label='Val Loss', marker='o')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_acc, label='Train Accuracy', marker='o')
    axes[1].plot(epochs, val_acc, label='Val Accuracy', marker='o')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_history.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {os.path.join(log_dir, 'training_history.png')}")


def save_report(log_dir, baseline_acc, bert_acc, bert_hard_acc):
    path = os.path.join(log_dir, 'report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=== Coursework Evaluation Report ===\n\n")
        f.write("1. Baseline Model (TF-IDF + Logistic Regression):\n")
        f.write(f"   - Test Accuracy: {baseline_acc:.4f}\n\n")
        f.write("2. Fine-tuned BERT-small:\n")
        f.write(f"   - Test Accuracy: {bert_acc:.4f}\n")
        if bert_hard_acc is not None:
            f.write(f"   - Accuracy on Hard Cases (double negation): {bert_hard_acc:.4f}\n")
        else:
            f.write("   - Hard cases: not found in test set\n")
    print(f"Report saved to: {path}")


# ============================================================================
# КАСТОМНЫЙ CALLBACK
# ============================================================================

class MetricsHistoryCallback(Callback):
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        # Безопасное извлечение как float
        def safe_float(val):
            if torch.is_tensor(val):
                return val.detach().cpu().item()
            return float(val)

        self.history['train_loss'].append(safe_float(logs.get('train_loss_epoch', float('nan'))))
        self.history['val_loss'].append(safe_float(logs.get('val_loss', float('nan'))))
        self.history['train_acc'].append(safe_float(logs.get('train_acc_epoch', float('nan'))))
        self.history['val_acc'].append(safe_float(logs.get('val_acc', float('nan'))))


# ============================================================================
# МОДЕЛЬ
# ============================================================================

class BertSmallClassifier(pl.LightningModule):
    def __init__(self, model_name, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_emb)

    def _shared_step(self, batch, stage):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    torch.set_float32_matmul_precision('high')

    DATA_PATH = r'N:\PyCharmRepo\NN_course\IMDB_Dataset_BERT_ready.csv'
    MODEL_PATH = r'N:\PyCharmRepo\NN_course\prajjwal1_bert_small'
    LOG_DIR = r'N:\PyCharmRepo\NN_course\logs'

    setup_logging(LOG_DIR)

    texts, labels = load_and_prepare_data(DATA_PATH)
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    baseline_acc, _ = create_baseline_model(X_train, y_train, X_test, y_test)
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def encode_data(texts, max_length=256):
        return tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

    train_enc = encode_data(X_train)
    val_enc = encode_data(X_val)
    test_enc = encode_data(X_test)

    train_dataset = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], torch.tensor(y_train))
    val_dataset = TensorDataset(val_enc['input_ids'], val_enc['attention_mask'], torch.tensor(y_val))
    test_dataset = TensorDataset(test_enc['input_ids'], test_enc['attention_mask'], torch.tensor(y_test))

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )

    metrics_callback = MetricsHistoryCallback()

    model = BertSmallClassifier(model_name=MODEL_PATH)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=2, mode="min"),
            metrics_callback
        ],
        logger=TensorBoardLogger(save_dir=LOG_DIR, name="bert_small_training"),
        enable_checkpointing=False,
        precision="16-mixed"
    )
    trainer.fit(model, train_loader, val_loader)

    test_results = trainer.test(model, test_loader, verbose=False)
    bert_acc = test_results[0]['test_acc']
    print(f"BERT-small test accuracy: {bert_acc:.4f}")

    # Получение предсказаний для матрицы ошибок
    model.eval()
    all_preds, all_labels = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            labels_batch = batch[2]
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_batch.tolist())

    # ✅ Быстрая оценка на сложных случаях
    bert_hard_acc = evaluate_on_hard_cases(model, tokenizer, X_test, y_test, device)
    if bert_hard_acc is not None:
        print(f"BERT-small accuracy on hard cases: {bert_hard_acc:.4f}")

    # Сохранение артефактов
    plot_confusion_matrix(all_labels, all_preds, LOG_DIR)
    plot_training_history(metrics_callback.history, LOG_DIR)
    save_report(LOG_DIR, baseline_acc, bert_acc, bert_hard_acc)

    # Сохранение модели
    model_save_path = os.path.join(LOG_DIR, "bert_small_finetuned")
    model.bert.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Fine-tuned model saved to: {model_save_path}")

    print("\nFinal Results:")
    print(f"Baseline (TF-IDF + LR):        {baseline_acc:.4f}")
    print(f"BERT-small (fine-tuned):       {bert_acc:.4f}")
    if bert_hard_acc is not None:
        print(f"BERT-small (hard cases):       {bert_hard_acc:.4f}")


if __name__ == "__main__":
    main()