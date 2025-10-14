"""
CNN-LSTM Model Training Script for Multi-label Text Classification

This script trains a CNN-LSTM model for multi-label classification with:
- input_guardrail: Binary classification (yes/no)
- routing: Multi-class classification (Health, general, coaching)
- bye_intent: Binary classification (yes/no)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pickle
import os
from data_preprocessing import load_and_preprocess_data, create_data_loaders

class CNNLSTMModel(nn.Module):
    """CNN-LSTM model for multi-label text classification"""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_filters=100,
                 filter_sizes=[3, 4, 5], num_routing_classes=3, dropout=0.5):
        super(CNNLSTMModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        # LSTM layer
        lstm_input_size = num_filters * len(filter_sizes)
        self.lstm = nn.LSTM(lstm_input_size, hidden_dim, batch_first=True, dropout=dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layers for each task
        self.input_guardrail_classifier = nn.Linear(hidden_dim, 1)  # Binary
        self.routing_classifier = nn.Linear(hidden_dim, num_routing_classes)  # Multi-class
        self.bye_intent_classifier = nn.Linear(hidden_dim, 1)  # Binary

    def forward(self, x):
        # Embedding: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # CNN: need (batch_size, embedding_dim, seq_len) for Conv1d
        embedded = embedded.transpose(1, 2)

        # Apply convolutions and ReLU + MaxPool
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # (batch_size, num_filters, new_seq_len)
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # Global max pooling
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)

        # Concatenate all conv outputs
        cnn_output = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))

        # Reshape for LSTM: (batch_size, 1, lstm_input_size)
        lstm_input = cnn_output.unsqueeze(1)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(lstm_input)

        # Use the last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Apply dropout
        last_hidden = self.dropout(last_hidden)

        # Classification heads
        input_guardrail_out = torch.sigmoid(self.input_guardrail_classifier(last_hidden))
        routing_out = self.routing_classifier(last_hidden)  # Will apply softmax in loss
        bye_intent_out = torch.sigmoid(self.bye_intent_classifier(last_hidden))

        return {
            'input_guardrail': input_guardrail_out,
            'routing': routing_out,
            'bye_intent': bye_intent_out
        }

class MultiTaskLoss(nn.Module):
    """Multi-task loss function"""

    def __init__(self, num_routing_classes):
        super(MultiTaskLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # targets: [input_guardrail, routing, bye_intent]
        input_guardrail_loss = self.bce_loss(
            predictions['input_guardrail'].squeeze(),
            targets[:, 0].float()
        )

        routing_loss = self.ce_loss(
            predictions['routing'],
            targets[:, 1].long()
        )

        bye_intent_loss = self.bce_loss(
            predictions['bye_intent'].squeeze(),
            targets[:, 2].float()
        )

        # Combine losses (you can adjust weights)
        total_loss = input_guardrail_loss + routing_loss + bye_intent_loss

        return {
            'total_loss': total_loss,
            'input_guardrail_loss': input_guardrail_loss,
            'routing_loss': routing_loss,
            'bye_intent_loss': bye_intent_loss
        }

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cpu'):
    """Train the CNN-LSTM model"""

    criterion = MultiTaskLoss(num_routing_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)
            loss_dict = criterion(outputs, labels)
            loss = loss_dict['total_loss']

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)
                loss_dict = criterion(outputs, labels)
                loss = loss_dict['total_loss']

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model and return metrics"""

    model.eval()
    all_predictions = {'input_guardrail': [], 'routing': [], 'bye_intent': []}
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)

            # Convert predictions to numpy
            input_guardrail_pred = (outputs['input_guardrail'].cpu().numpy() > 0.5).astype(int)
            routing_pred = torch.argmax(outputs['routing'], dim=1).cpu().numpy()
            bye_intent_pred = (outputs['bye_intent'].cpu().numpy() > 0.5).astype(int)

            all_predictions['input_guardrail'].extend(input_guardrail_pred.flatten())
            all_predictions['routing'].extend(routing_pred)
            all_predictions['bye_intent'].extend(bye_intent_pred.flatten())

            all_targets.extend(labels.cpu().numpy())

    all_targets = np.array(all_targets)

    # Calculate metrics for each task
    metrics = {}

    # Input guardrail metrics
    metrics['input_guardrail'] = {
        'accuracy': accuracy_score(all_targets[:, 0], all_predictions['input_guardrail']),
        'f1': f1_score(all_targets[:, 0], all_predictions['input_guardrail'])
    }

    # Routing metrics
    metrics['routing'] = {
        'accuracy': accuracy_score(all_targets[:, 1], all_predictions['routing']),
        'f1': f1_score(all_targets[:, 1], all_predictions['routing'], average='weighted')
    }

    # Bye intent metrics
    metrics['bye_intent'] = {
        'accuracy': accuracy_score(all_targets[:, 2], all_predictions['bye_intent']),
        'f1': f1_score(all_targets[:, 2], all_predictions['bye_intent'])
    }

    return metrics, all_predictions, all_targets

def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses"""

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN-LSTM Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_lstm_training_history.png')
    plt.show()

def main():
    """Main training function"""

    # Configuration
    DATA_PATH = 'your_data.csv'  # Update this path
    MODEL_SAVE_PATH = 'cnn_lstm_model.pth'
    PREPROCESSOR_SAVE_PATH = 'cnn_lstm_preprocessor.pkl'

    BATCH_SIZE = 32
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(DATA_PATH)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        batch_size=BATCH_SIZE,
        use_bert=False
    )

    # Initialize model
    model = CNNLSTMModel(
        vocab_size=data['vocab_size'],
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_routing_classes=data['num_routing_classes']
    ).to(device)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )

    # Plot training history
    plot_training_history(train_losses, val_losses)

    # Evaluate model
    print("Evaluating model...")
    metrics, predictions, targets = evaluate_model(model, test_loader, device)

    # Print results
    print("\nFinal Results:")
    for task, task_metrics in metrics.items():
        print(f"\n{task.upper()}:")
        print(f"  Accuracy: {task_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {task_metrics['f1']:.4f}")

    # Save model and preprocessors
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    with open(PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump({
            'text_processor': data['text_processor'],
            'label_processor': data['label_processor'],
            'vocab_size': data['vocab_size'],
            'num_routing_classes': data['num_routing_classes']
        }, f)

    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Preprocessors saved to: {PREPROCESSOR_SAVE_PATH}")

if __name__ == "__main__":
    main()