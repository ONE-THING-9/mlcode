"""
Data preprocessing utilities for multi-label text classification
Handles text preprocessing and label encoding for the training pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re
import string

class TextPreprocessor:
    """Text preprocessing utilities"""

    def __init__(self):
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.max_vocab_size = 10000

    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_freq = {}

        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and take top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, freq in sorted_words[:self.max_vocab_size-2]]

        # Add special tokens
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}

        for idx, word in enumerate(vocab_words, 2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self.vocab_size = len(self.word_to_idx)

    def text_to_sequence(self, text, max_length=128):
        """Convert text to sequence of indices"""
        words = text.split()
        sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>

        # Pad or truncate
        if len(sequence) < max_length:
            sequence.extend([0] * (max_length - len(sequence)))  # 0 is <PAD>
        else:
            sequence = sequence[:max_length]

        return sequence

class LabelProcessor:
    """Process multi-label targets"""

    def __init__(self):
        self.routing_encoder = LabelEncoder()

    def encode_labels(self, df):
        """Encode all labels"""
        # Binary labels
        df['input_guardrail_encoded'] = (df['input_guardrail'] == 'yes').astype(int)
        df['bye_intent_encoded'] = (df['bye_intent'] == 'yes').astype(int)

        # Multi-class label (routing)
        df['routing_encoded'] = self.routing_encoder.fit_transform(df['routing'])

        return df

    def get_routing_classes(self):
        """Get routing class names"""
        return self.routing_encoder.classes_

class MultiLabelDataset(Dataset):
    """PyTorch Dataset for multi-label classification"""

    def __init__(self, texts, labels, tokenizer=None, max_length=128, use_bert=False):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.use_bert = use_bert
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.use_bert:
            # For BERT models
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        else:
            # For CNN-LSTM (text is already converted to sequence)
            return {
                'input_ids': torch.tensor(text, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.float)
            }

def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    Load and preprocess data for training

    Expected CSV format:
    text,input_guardrail,routing,bye_intent
    "Hello doctor",yes,Health,no
    "I want to quit",no,coaching,yes
    """

    # Load data
    df = pd.read_csv(file_path)

    # Initialize processors
    text_processor = TextPreprocessor()
    label_processor = LabelProcessor()

    # Clean texts
    df['text_cleaned'] = df['text'].apply(text_processor.clean_text)

    # Encode labels
    df = label_processor.encode_labels(df)

    # Build vocabulary for CNN-LSTM
    text_processor.build_vocab(df['text_cleaned'].tolist())

    # Convert texts to sequences
    sequences = [text_processor.text_to_sequence(text) for text in df['text_cleaned']]

    # Prepare labels (input_guardrail, routing, bye_intent)
    labels = np.column_stack([
        df['input_guardrail_encoded'].values,
        df['routing_encoded'].values,
        df['bye_intent_encoded'].values
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state, stratify=labels[:, 1]
    )

    # Also split raw texts for BERT
    texts_train, texts_test = train_test_split(
        df['text_cleaned'].tolist(), test_size=test_size, random_state=random_state, stratify=labels[:, 1]
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'texts_train': texts_train,
        'texts_test': texts_test,
        'text_processor': text_processor,
        'label_processor': label_processor,
        'vocab_size': text_processor.vocab_size,
        'num_routing_classes': len(label_processor.get_routing_classes())
    }

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=32, use_bert=False, tokenizer=None):
    """Create PyTorch DataLoaders"""

    train_dataset = MultiLabelDataset(X_train, y_train, tokenizer=tokenizer, use_bert=use_bert)
    test_dataset = MultiLabelDataset(X_test, y_test, tokenizer=tokenizer, use_bert=use_bert)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    print("Data preprocessing utilities loaded successfully!")
    print("Use load_and_preprocess_data('your_data.csv') to process your dataset")