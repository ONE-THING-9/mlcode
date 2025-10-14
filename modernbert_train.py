"""
ModernBERT Model Training Script for Multi-label Text Classification

This script trains a ModernBERT model with three different training strategies:
1. Full model fine-tuning
2. LoRA (Low-Rank Adaptation) fine-tuning
3. Last layer only fine-tuning

Tasks:
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
import argparse
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from data_preprocessing import load_and_preprocess_data, create_data_loaders

class ModernBERTModel(nn.Module):
    """ModernBERT model for multi-label text classification"""

    def __init__(self, model_name='answerdotai/ModernBERT-base', num_routing_classes=3, dropout=0.1):
        super(ModernBERTModel, self).__init__()

        # Load ModernBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification heads
        self.input_guardrail_classifier = nn.Linear(self.hidden_size, 1)  # Binary
        self.routing_classifier = nn.Linear(self.hidden_size, num_routing_classes)  # Multi-class
        self.bye_intent_classifier = nn.Linear(self.hidden_size, 1)  # Binary

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification heads
        input_guardrail_out = torch.sigmoid(self.input_guardrail_classifier(pooled_output))
        routing_out = self.routing_classifier(pooled_output)  # Will apply softmax in loss
        bye_intent_out = torch.sigmoid(self.bye_intent_classifier(pooled_output))

        return {
            'input_guardrail': input_guardrail_out,
            'routing': routing_out,
            'bye_intent': bye_intent_out
        }

class ModernBERTLoRAModel(nn.Module):
    """ModernBERT with LoRA adaptation"""

    def __init__(self, model_name='answerdotai/ModernBERT-base', num_routing_classes=3,
                 dropout=0.1, lora_rank=8, lora_alpha=32):
        super(ModernBERTLoRAModel, self).__init__()

        # Load base model
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["query", "value", "key", "dense"]  # ModernBERT attention modules
        )

        # Apply LoRA to the model
        self.bert = get_peft_model(self.bert, lora_config)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification heads
        self.input_guardrail_classifier = nn.Linear(self.hidden_size, 1)
        self.routing_classifier = nn.Linear(self.hidden_size, num_routing_classes)
        self.bye_intent_classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification heads
        input_guardrail_out = torch.sigmoid(self.input_guardrail_classifier(pooled_output))
        routing_out = self.routing_classifier(pooled_output)
        bye_intent_out = torch.sigmoid(self.bye_intent_classifier(pooled_output))

        return {
            'input_guardrail': input_guardrail_out,
            'routing': routing_out,
            'bye_intent': bye_intent_out
        }

class ModernBERTLastLayerModel(nn.Module):
    """ModernBERT with only last layer fine-tuning"""

    def __init__(self, model_name='answerdotai/ModernBERT-base', num_routing_classes=3, dropout=0.1):
        super(ModernBERTLastLayerModel, self).__init__()

        # Load ModernBERT and freeze all parameters
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Freeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification heads (only these will be trained)
        self.input_guardrail_classifier = nn.Linear(self.hidden_size, 1)
        self.routing_classifier = nn.Linear(self.hidden_size, num_routing_classes)
        self.bye_intent_classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs (frozen)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :].detach()

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification heads
        input_guardrail_out = torch.sigmoid(self.input_guardrail_classifier(pooled_output))
        routing_out = self.routing_classifier(pooled_output)
        bye_intent_out = torch.sigmoid(self.bye_intent_classifier(pooled_output))

        return {
            'input_guardrail': input_guardrail_out,
            'routing': routing_out,
            'bye_intent': bye_intent_out
        }

class MultiTaskLoss(nn.Module):
    """Multi-task loss function"""

    def __init__(self):
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

        # Combine losses
        total_loss = input_guardrail_loss + routing_loss + bye_intent_loss

        return {
            'total_loss': total_loss,
            'input_guardrail_loss': input_guardrail_loss,
            'routing_loss': routing_loss,
            'bye_intent_loss': bye_intent_loss
        }

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=2e-5, device='cpu'):
    """Train the ModernBERT model"""

    criterion = MultiTaskLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss_dict = criterion(outputs, labels)
            loss = loss_dict['total_loss']

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f'  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
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
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)

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

def plot_training_history(train_losses, val_losses, strategy):
    """Plot training and validation losses"""

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'ModernBERT Training History ({strategy})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'modernbert_{strategy}_training_history.png')
    plt.show()

def main():
    """Main training function"""

    parser = argparse.ArgumentParser(description='Train ModernBERT model')
    parser.add_argument('--strategy', choices=['full', 'lora', 'last_layer'],
                       default='full', help='Training strategy')
    parser.add_argument('--data_path', type=str, default='your_data.csv',
                       help='Path to training data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')

    args = parser.parse_args()

    # Configuration
    MODEL_NAME = 'answerdotai/ModernBERT-base'
    MODEL_SAVE_PATH = f'modernbert_{args.strategy}_model.pth'
    TOKENIZER_SAVE_PATH = f'modernbert_{args.strategy}_tokenizer'
    PREPROCESSOR_SAVE_PATH = f'modernbert_{args.strategy}_preprocessor.pkl'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Training strategy: {args.strategy}')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(args.data_path)

    # Create data loaders for BERT
    train_loader, test_loader = create_data_loaders(
        data['texts_train'], data['texts_test'],
        data['y_train'], data['y_test'],
        batch_size=args.batch_size,
        use_bert=True,
        tokenizer=tokenizer
    )

    # Initialize model based on strategy
    if args.strategy == 'full':
        model = ModernBERTModel(
            model_name=MODEL_NAME,
            num_routing_classes=data['num_routing_classes']
        )
    elif args.strategy == 'lora':
        model = ModernBERTLoRAModel(
            model_name=MODEL_NAME,
            num_routing_classes=data['num_routing_classes']
        )
    else:  # last_layer
        model = ModernBERTLastLayerModel(
            model_name=MODEL_NAME,
            num_routing_classes=data['num_routing_classes']
        )

    model = model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )

    # Plot training history
    plot_training_history(train_losses, val_losses, args.strategy)

    # Evaluate model
    print("Evaluating model...")
    metrics, predictions, targets = evaluate_model(model, test_loader, device)

    # Print results
    print(f"\nFinal Results ({args.strategy}):")
    for task, task_metrics in metrics.items():
        print(f"\n{task.upper()}:")
        print(f"  Accuracy: {task_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {task_metrics['f1']:.4f}")

    # Save model and tokenizer
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)

    # Save preprocessing info
    with open(PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump({
            'label_processor': data['label_processor'],
            'num_routing_classes': data['num_routing_classes'],
            'model_name': MODEL_NAME,
            'strategy': args.strategy
        }, f)

    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Tokenizer saved to: {TOKENIZER_SAVE_PATH}")
    print(f"Preprocessors saved to: {PREPROCESSOR_SAVE_PATH}")

if __name__ == "__main__":
    main()