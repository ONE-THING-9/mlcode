"""
CNN-LSTM Model Inference Script

This script loads a trained CNN-LSTM model and performs inference on new text data.
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
from data_preprocessing import TextPreprocessor
from cnn_lstm_train import CNNLSTMModel

class CNNLSTMInference:
    """CNN-LSTM model inference class"""

    def __init__(self, model_path, preprocessor_path, device='cpu'):
        """
        Initialize the inference class

        Args:
            model_path: Path to the saved model state dict
            preprocessor_path: Path to the saved preprocessors
            device: Device to run inference on
        """
        self.device = device

        # Load preprocessors
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)

        self.text_processor = preprocessor_data['text_processor']
        self.label_processor = preprocessor_data['label_processor']
        self.vocab_size = preprocessor_data['vocab_size']
        self.num_routing_classes = preprocessor_data['num_routing_classes']

        # Initialize and load model
        self.model = CNNLSTMModel(
            vocab_size=self.vocab_size,
            num_routing_classes=self.num_routing_classes
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # Get routing class names
        self.routing_classes = self.label_processor.get_routing_classes()

        print("Model loaded successfully!")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Routing classes: {list(self.routing_classes)}")

    def preprocess_text(self, text):
        """Preprocess a single text for inference"""
        # Clean text
        cleaned_text = self.text_processor.clean_text(text)

        # Convert to sequence
        sequence = self.text_processor.text_to_sequence(cleaned_text)

        # Convert to tensor
        input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)

        return input_tensor

    def predict_single(self, text):
        """
        Make prediction for a single text

        Args:
            text: Input text string

        Returns:
            Dictionary with predictions for all tasks
        """
        # Preprocess text
        input_tensor = self.preprocess_text(text)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)

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

    def predict_batch(self, texts):
        """
        Make predictions for multiple texts

        Args:
            texts: List of text strings

        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for text in texts:
            pred = self.predict_single(text)
            predictions.append(pred)
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

    # Configuration
    MODEL_PATH = 'cnn_lstm_model.pth'
    PREPROCESSOR_PATH = 'cnn_lstm_preprocessor.pkl'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Initialize inference class
        inference = CNNLSTMInference(MODEL_PATH, PREPROCESSOR_PATH, device)

        # Example texts for testing
        test_texts = [
            "Hello doctor, I have a headache",
            "I want to quit smoking",
            "Thank you, goodbye",
            "Can you help me with my workout plan?",
            "What's the weather like today?"
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
        print("Please make sure you have trained the model first using cnn_lstm_train.py")
    except Exception as e:
        print(f"An error occurred: {e}")

def interactive_mode():
    """Interactive mode for testing the model"""

    MODEL_PATH = 'cnn_lstm_model.pth'
    PREPROCESSOR_PATH = 'cnn_lstm_preprocessor.pkl'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        inference = CNNLSTMInference(MODEL_PATH, PREPROCESSOR_PATH, device)

        print("\nInteractive CNN-LSTM Inference Mode")
        print("Type 'quit' to exit")
        print("=" * 40)

        while True:
            text = input("\nEnter text for prediction: ").strip()

            if text.lower() == 'quit':
                break

            if text:
                print(inference.explain_prediction(text))
            else:
                print("Please enter some text.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained the model first using cnn_lstm_train.py")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        main()