"""
Guardrail Binary Classification Training Script

Supports three backbones for yes/no guardrail decisions:
- CNN text classifier
- LSTM classifier
- `prajjwal1/bert-tiny` transformer

The input CSV must contain at least two columns:
    text,input_guardrail
where the guardrail column is either yes/no style strings or binary values.
"""

import argparse
import copy
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from data_preprocessing import TextPreprocessor


def encode_guardrail_label(value):
    """Map textual or numeric labels to {0, 1}."""
    if pd.isna(value):
        raise ValueError("Encountered empty value in input_guardrail column.")

    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"yes", "y", "true", "1"}:
            return 1
        if normalised in {"no", "n", "false", "0"}:
            return 0
        raise ValueError(f"Unrecognised guardrail label: {value}")

    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unrecognised guardrail label: {value}") from exc

    if numeric not in {0, 1}:
        raise ValueError(f"Guardrail label must be binary. Received: {value}")
    return numeric


def prepare_guardrail_data(
    file_path,
    max_length=128,
    val_split=0.2,
    random_state=42,
):
    """Load, clean, and split guardrail data."""
    df = pd.read_csv(file_path)
    if "text" not in df.columns or "input_guardrail" not in df.columns:
        raise ValueError("Expected columns 'text' and 'input_guardrail' in the CSV file.")

    text_processor = TextPreprocessor()
    df["text_cleaned"] = df["text"].apply(text_processor.clean_text)
    df["guardrail_label"] = df["input_guardrail"].apply(encode_guardrail_label)

    text_processor.build_vocab(df["text_cleaned"].tolist())

    sequences = [text_processor.text_to_sequence(text, max_length=max_length) for text in df["text_cleaned"]]
    raw_texts = df["text_cleaned"].tolist()
    labels = df["guardrail_label"].astype(np.float32).values

    (
        seq_train,
        seq_val,
        text_train,
        text_val,
        y_train,
        y_val,
    ) = train_test_split(
        sequences,
        raw_texts,
        labels,
        test_size=val_split,
        random_state=random_state,
        stratify=labels,
    )

    return {
        "seq_train": seq_train,
        "seq_val": seq_val,
        "text_train": text_train,
        "text_val": text_val,
        "y_train": y_train,
        "y_val": y_val,
        "text_processor": text_processor,
        "vocab_size": text_processor.vocab_size,
    }


class SequenceDataset(Dataset):
    """Dataset wrapper for CNN and LSTM models."""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class BertGuardrailDataset(Dataset):
    """Dataset wrapper for transformer models."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class CNNTextClassifier(nn.Module):
    """CNN-based binary text classifier."""

    def __init__(self, vocab_size, embedding_dim=128, num_filters=128, filter_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).transpose(1, 2)
        conv_outputs = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max(conv_out, dim=2)[0] for conv_out in conv_outputs]
        features = torch.cat(pooled, dim=1)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits.squeeze(-1)


class LSTMTextClassifier(nn.Module):
    """LSTM-based binary text classifier."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        bidirectional=True,
        dropout=0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim, 1)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        features = lstm_out[:, -1, :]
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits.squeeze(-1)


class BertTinyClassifier(nn.Module):
    """`prajjwal1/bert-tiny` fine-tuned for binary classification."""

    def __init__(self, model_name="prajjwal1/bert-tiny", dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits.squeeze(-1)


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for batch in data_loader:
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        if attention_mask is not None:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = model(input_ids=input_ids)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(len(data_loader), 1)


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in data_loader:
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if attention_mask is not None:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = model(input_ids=input_ids)

        loss = criterion(logits, labels)
        epoch_loss += loss.item()

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = epoch_loss / max(len(data_loader), 1)
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0) if all_labels else 0.0

    return (
        avg_loss,
        accuracy,
        f1,
        classification_report(all_labels, all_preds, target_names=["no", "yes"], zero_division=0),
    )


def train_and_validate(model, train_loader, val_loader, optimizer, device, epochs):
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_report = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_report": val_report,
            }
        )

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def main():
    parser = argparse.ArgumentParser(description="Train guardrail binary classifier.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file with text and input_guardrail.")
    parser.add_argument("--model_type", choices=["cnn", "lstm", "bert"], default="bert", help="Backbone to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override default learning rate.")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length for tokenisation.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Proportion of data for validation.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for CNN/LSTM.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for LSTM.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM if set.")
    parser.add_argument("--bert_model_name", type=str, default="prajjwal1/bert-tiny", help="HF model checkpoint.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimiser.")
    parser.add_argument("--output_dir", type=str, default="guardrail_artifacts", help="Directory to save artefacts.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected model: {args.model_type}")

    data = prepare_guardrail_data(
        args.data_path,
        max_length=args.max_length,
        val_split=args.val_split,
        random_state=args.random_state,
    )

    lr_default = 2e-5 if args.model_type == "bert" else 1e-3
    learning_rate = args.learning_rate if args.learning_rate is not None else lr_default

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.model_type in {"cnn", "lstm"}:
        train_dataset = SequenceDataset(data["seq_train"], data["y_train"])
        val_dataset = SequenceDataset(data["seq_val"], data["y_val"])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        if args.model_type == "cnn":
            model = CNNTextClassifier(
                vocab_size=data["vocab_size"],
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
            )
        else:
            model = LSTMTextClassifier(
                vocab_size=data["vocab_size"],
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                bidirectional=args.bidirectional,
                dropout=args.dropout,
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
        train_dataset = BertGuardrailDataset(data["text_train"], data["y_train"], tokenizer, args.max_length)
        val_dataset = BertGuardrailDataset(data["text_val"], data["y_val"], tokenizer, args.max_length)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = BertTinyClassifier(model_name=args.bert_model_name, dropout=args.dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

        tokenizer.save_pretrained(args.output_dir)
        print(f"Tokenizer saved to {args.output_dir}")

    model = model.to(device)
    history = train_and_validate(model, train_loader, val_loader, optimizer, device, args.epochs)

    final_report = history[-1]["val_report"] if history else "No training history available."
    print("\nValidation classification report (best checkpoint):")
    print(final_report)

    model_path = Path(args.output_dir) / f"guardrail_{args.model_type}_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

    if args.model_type in {"cnn", "lstm"}:
        preprocessor_path = Path(args.output_dir) / "guardrail_text_preprocessor.pkl"
        with open(preprocessor_path, "wb") as handle:
            pickle.dump(
                {
                    "text_processor": data["text_processor"],
                    "max_length": args.max_length,
                },
                handle,
            )
        print(f"Text preprocessor saved to {preprocessor_path}")


if __name__ == "__main__":
    main()
