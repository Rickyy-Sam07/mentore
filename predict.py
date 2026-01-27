"""
Inference Script for BERT Chapter Classification
Predicts chapter from input text
"""

import torch
from transformers import BertTokenizer
import pickle
import os
import sys

from config import *
from train import BertChapterClassifier


class ChapterPredictor:
    """Class for making predictions on new text"""
    
    def __init__(self, model_path, label_encoder_path):
        """Initialize the predictor"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.num_classes = len(self.label_encoder.classes_)
        print(f"Number of classes: {self.num_classes}")
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model
        self.model = BertChapterClassifier(num_classes=self.num_classes)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print("Ready for predictions!\n")
    
    def predict(self, text, return_probabilities=False):
        """
        Predict chapter for given text
        
        Args:
            text: Input text string
            return_probabilities: If True, return top 5 predictions with probabilities
        
        Returns:
            Predicted chapter name (and probabilities if requested)
        """
        
        # Tokenize input
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            _, prediction = torch.max(outputs, dim=1)
        
        predicted_label = prediction.item()
        predicted_chapter = self.label_encoder.inverse_transform([predicted_label])[0]
        confidence = probabilities[0][predicted_label].item()
        
        if return_probabilities:
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities[0], k=min(5, self.num_classes))
            top_chapters = self.label_encoder.inverse_transform(top_indices.cpu().numpy())
            
            results = {
                'predicted_chapter': predicted_chapter,
                'confidence': confidence,
                'top_predictions': [
                    {'chapter': chapter, 'probability': prob.item()}
                    for chapter, prob in zip(top_chapters, top_probs)
                ]
            }
            return results
        
        return predicted_chapter, confidence
    
    def predict_batch(self, texts):
        """Predict chapters for multiple texts"""
        
        predictions = []
        for text in texts:
            chapter, confidence = self.predict(text)
            predictions.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'predicted_chapter': chapter,
                'confidence': confidence
            })
        
        return predictions


def main():
    """Main function for interactive predictions"""
    
    # Check if model exists
    model_path = os.path.join(MODEL_DIR, 'best_model.pth')
    label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running: python train.py")
        return
    
    if not os.path.exists(label_encoder_path):
        print(f"Error: Label encoder not found at {label_encoder_path}")
        print("Please train the model first by running: python train.py")
        return
    
    # Initialize predictor
    print("="*70)
    print("BERT CHAPTER CLASSIFICATION - INFERENCE")
    print("="*70)
    print()
    
    predictor = ChapterPredictor(model_path, label_encoder_path)
    
    # Example predictions
    print("="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    example_texts = [
        "O Sage, I would hear of such a man from you, who art able to describe him to me.",
        "The wise and eloquent Valmiki with his disciple, Bharadvaja, having listened to the words of Narada, was filled with wonder and worshipped Rama in his heart.",
        "Rama, Lakshmana and Sita dwelt happily in the forest like devas or gandharvas."
    ]
    
    for i, text in enumerate(example_texts, 1):
        print(f"\nExample {i}:")
        print(f"Text: {text}")
        
        result = predictor.predict(text, return_probabilities=True)
        
        print(f"\nPredicted Chapter: {result['predicted_chapter']}")
        print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        
        print(f"\n  Top 3 Predictions:")
        for j, pred in enumerate(result['top_predictions'][:3], 1):
            print(f"    {j}. {pred['chapter']}")
            print(f"       Probability: {pred['probability']:.4f} ({pred['probability']*100:.2f}%)")
        
        print("-"*70)
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter text to classify (or 'quit' to exit):")
    print("-"*70)
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            result = predictor.predict(user_input, return_probabilities=True)
            
            print(f"\nPredicted Chapter: {result['predicted_chapter']}")
            print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            
            print(f"\n  Top 5 Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"    {i}. {pred['chapter']}")
                print(f"       Probability: {pred['probability']:.4f} ({pred['probability']*100:.2f}%)")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == '__main__':
    main()
