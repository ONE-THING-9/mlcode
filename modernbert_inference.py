"""
ModernBERT Model Inference Script

This script loads a trained ModernBERT model and performs inference on new text data.
Supports all three training strategies: full, LoRA, and last_layer.
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import argparse
from transformers import AutoTokenizer
from modernbert_train import ModernBERTModel, ModernBERTLoRAModel, ModernBERTLastLayerModel

class ModernBERTInference:
    """ModernBERT model inference class"""

    def __init__(self, model_path, tokenizer_path, preprocessor_path, device='cpu'):
        """
        Initialize the inference class

        Args:
            model_path: Path to the saved model state dict
            tokenizer_path: Path to the saved tokenizer
            preprocessor_path: Path to the saved preprocessors
            device: Device to run inference on
        """
        self.device = device

        # Load preprocessors
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)

        self.label_processor = preprocessor_data['label_processor']
        self.num_routing_classes = preprocessor_data['num_routing_classes']
        self.model_name = preprocessor_data['model_name']
        self.strategy = preprocessor_data['strategy']

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Initialize model based on strategy
        if self.strategy == 'full':
            self.model = ModernBERTModel(
                model_name=self.model_name,
                num_routing_classes=self.num_routing_classes
            )
        elif self.strategy == 'lora':
            self.model = ModernBERTLoRAModel(
                model_name=self.model_name,
                num_routing_classes=self.num_routing_classes
            )
        else:  # last_layer
            self.model = ModernBERTLastLayerModel(
                model_name=self.model_name,
                num_routing_classes=self.num_routing_classes
            )

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # Get routing class names
        self.routing_classes = self.label_processor.get_routing_classes()

        print("ModernBERT model loaded successfully!")
        print(f"Strategy: {self.strategy}")
        print(f"Model: {self.model_name}")
        print(f"Routing classes: {list(self.routing_classes)}")

    def preprocess_text(self, text, max_length=128):
        """Preprocess a single text for inference"""
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    def predict_single(self, text):
        """
        Make prediction for a single text

        Args:
            text: Input text string

        Returns:
            Dictionary with predictions for all tasks
        """
        # Preprocess text
        inputs = self.preprocess_text(text)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])

        # Process outputs
        input_guardrail_prob = outputs['input_guardrail'].cpu().numpy()[0][0]
        input_guardrail_pred = "yes" if input_guardrail_prob > 0.5 else "no"

        routing_probs = torch.softmax(outputs['routing'], dim=1).cpu().numpy()[0]
        routing_pred_idx = np.argmax(routing_probs)
        routing_pred = self.routing_classes[routing_pred_idx]

        bye_intent_prob = outputs['bye_intent'].cpu().numpy()[0][0]
        bye_intent_pred = "yes" if bye_intent_prob > 0.5 else "no"

        return {
            'input_guardrail': {
                'prediction': input_guardrail_pred,
                'probability': float(input_guardrail_prob)
            },
            'routing': {
                'prediction': routing_pred,
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.routing_classes, routing_probs)
                }
            },
            'bye_intent': {
                'prediction': bye_intent_pred,
                'probability': float(bye_intent_prob)
            }
        }

    def predict_batch(self, texts, batch_size=16):
        """
        Make predictions for multiple texts efficiently

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = []

            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )

            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            # Make predictions
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)

            # Process outputs for each item in batch
            for j in range(len(batch_texts)):
                input_guardrail_prob = outputs['input_guardrail'][j].cpu().numpy()[0]
                input_guardrail_pred = "yes" if input_guardrail_prob > 0.5 else "no"

                routing_probs = torch.softmax(outputs['routing'], dim=1)[j].cpu().numpy()
                routing_pred_idx = np.argmax(routing_probs)
                routing_pred = self.routing_classes[routing_pred_idx]

                bye_intent_prob = outputs['bye_intent'][j].cpu().numpy()[0]
                bye_intent_pred = "yes" if bye_intent_prob > 0.5 else "no"

                batch_predictions.append({
                    'input_guardrail': {
                        'prediction': input_guardrail_pred,
                        'probability': float(input_guardrail_prob)
                    },
                    'routing': {
                        'prediction': routing_pred,
                        'probabilities': {
                            class_name: float(prob)
                            for class_name, prob in zip(self.routing_classes, routing_probs)
                        }
                    },
                    'bye_intent': {
                        'prediction': bye_intent_pred,
                        'probability': float(bye_intent_prob)
                    }
                })

            predictions.extend(batch_predictions)

        return predictions

    def explain_prediction(self, text):
        """
        Provide a detailed explanation of the prediction

        Args:
            text: Input text string

        Returns:
            Formatted explanation string
        """
        prediction = self.predict_single(text)

        explanation = f"\nPrediction for: '{text}'\n"
        explanation += "=" * 50 + "\n"
        explanation += f"Model: ModernBERT ({self.strategy} strategy)\n\n"

        explanation += f"Input Guardrail: {prediction['input_guardrail']['prediction']} "
        explanation += f"(confidence: {prediction['input_guardrail']['probability']:.3f})\n"

        explanation += f"Routing: {prediction['routing']['prediction']}\n"
        explanation += "  Routing probabilities:\n"
        for class_name, prob in prediction['routing']['probabilities'].items():
            explanation += f"    {class_name}: {prob:.3f}\n"

        explanation += f"Bye Intent: {prediction['bye_intent']['prediction']} "
        explanation += f"(confidence: {prediction['bye_intent']['probability']:.3f})\n"

        return explanation

def main():
    """Example usage of the inference class"""

    parser = argparse.ArgumentParser(description='ModernBERT Inference')
    parser.add_argument('--strategy', choices=['full', 'lora', 'last_layer'],
                       default='full', help='Training strategy used')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')

    args = parser.parse_args()

    # Configuration
    MODEL_PATH = f'modernbert_{args.strategy}_model.pth'
    TOKENIZER_PATH = f'modernbert_{args.strategy}_tokenizer'
    PREPROCESSOR_PATH = f'modernbert_{args.strategy}_preprocessor.pkl'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Initialize inference class
        inference = ModernBERTInference(MODEL_PATH, TOKENIZER_PATH, PREPROCESSOR_PATH, device)

        if args.interactive:
            # Interactive mode
            print(f"\nInteractive ModernBERT Inference Mode ({args.strategy})")
            print("Type 'quit' to exit")
            print("=" * 50)

            while True:
                text = input("\nEnter text for prediction: ").strip()

                if text.lower() == 'quit':
                    break

                if text:
                    print(inference.explain_prediction(text))
                else:
                    print("Please enter some text.")

        else:
            # Example texts for testing
            test_texts = [
                "Hello doctor, I have a headache and feel dizzy",
                "I want to quit smoking and need help",
                "Thank you so much, goodbye!",
                "Can you help me with my workout plan and nutrition?",
                "What's the weather like today?",
                "I'm having chest pain, should I be worried?",
                "How can I improve my running technique?",
                "Thanks for everything, see you later"
            ]

            print("\nMaking predictions for example texts:")
            print("=" * 60)

            for text in test_texts:
                print(inference.explain_prediction(text))

            # Batch prediction example
            print("\nBatch prediction results:")
            print("=" * 30)
            batch_predictions = inference.predict_batch(test_texts)

            for i, (text, pred) in enumerate(zip(test_texts, batch_predictions)):
                print(f"{i+1}. '{text}'")
                print(f"   Guardrail: {pred['input_guardrail']['prediction']}")
                print(f"   Routing: {pred['routing']['prediction']}")
                print(f"   Bye Intent: {pred['bye_intent']['prediction']}")
                print()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure you have trained the {args.strategy} model first using:")
        print(f"python modernbert_train.py --strategy {args.strategy}")
    except Exception as e:
        print(f"An error occurred: {e}")

def compare_strategies():
    """Compare predictions from different training strategies"""

    test_texts = [
        "Hello doctor, I have a headache",
        "I want to quit smoking",
        "Thank you, goodbye",
        "Can you help me with my workout plan?"
    ]

    strategies = ['full', 'lora', 'last_layer']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Comparing ModernBERT strategies:")
    print("=" * 50)

    for text in test_texts:
        print(f"\nText: '{text}'")
        print("-" * 30)

        for strategy in strategies:
            try:
                model_path = f'modernbert_{strategy}_model.pth'
                tokenizer_path = f'modernbert_{strategy}_tokenizer'
                preprocessor_path = f'modernbert_{strategy}_preprocessor.pkl'

                inference = ModernBERTInference(model_path, tokenizer_path, preprocessor_path, device)
                prediction = inference.predict_single(text)

                print(f"{strategy.upper():>10}: Guardrail={prediction['input_guardrail']['prediction']}, "
                      f"Routing={prediction['routing']['prediction']}, "
                      f"Bye={prediction['bye_intent']['prediction']}")

            except FileNotFoundError:
                print(f"{strategy.upper():>10}: Model not found")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_strategies()
    else:
        main()