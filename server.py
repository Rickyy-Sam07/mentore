"""
FastAPI Server for DistilBERT Model
For Computer 1 (Remote Server)
Run this on the computer with DistilBERT model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer
import pickle
import os
from typing import List, Dict
import uvicorn

from config_distilbert import *
from train_distilbert import DistilBertChapterClassifier

# Initialize FastAPI app
app = FastAPI(
    title="DistilBERT Chapter Classification API",
    description="API for predicting chapter from text using DistilBERT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
tokenizer = None
label_encoder = None
device = None


class PredictionRequest(BaseModel):
    text: str
    top_k: int = 5


class PredictionResponse(BaseModel):
    predicted_chapter: str
    confidence: float
    top_predictions: List[Dict[str, float]]


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer, label_encoder, device
    
    print("Loading DistilBERT model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load label encoder
    label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model
    model_path = os.path.join(MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = DistilBertChapterClassifier(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DistilBERT Chapter Classification API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "info": "/info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": MODEL_NAME,
        "num_classes": len(label_encoder.classes_),
        "max_length": MAX_LENGTH,
        "device": str(device),
        "total_params": sum(p.numel() for p in model.parameters())
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict chapter from text"""
    
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        encoding = tokenizer.encode_plus(
            request.text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            _, prediction = torch.max(outputs, dim=1)
        
        predicted_label = prediction.item()
        predicted_chapter = label_encoder.inverse_transform([predicted_label])[0]
        confidence = probabilities[0][predicted_label].item()
        
        # Get top k predictions
        top_k = min(request.top_k, len(label_encoder.classes_))
        top_probs, top_indices = torch.topk(probabilities[0], k=top_k)
        top_chapters = label_encoder.inverse_transform(top_indices.cpu().numpy())
        
        top_predictions = [
            {"chapter": chapter, "probability": prob.item()}
            for chapter, prob in zip(top_chapters, top_probs)
        ]
        
        return PredictionResponse(
            predicted_chapter=predicted_chapter,
            confidence=confidence,
            top_predictions=top_predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    print("="*70)
    print("STARTING DISTILBERT API SERVER")
    print("="*70)
    print(f"\nServer will run on http://{SERVER_HOST}:{SERVER_PORT}")
    print("\nAfter starting, use ngrok to create public URL:")
    print(f"  ngrok http {SERVER_PORT}")
    print("\nEndpoints:")
    print(f"  - Health: http://localhost:{SERVER_PORT}/health")
    print(f"  - Info: http://localhost:{SERVER_PORT}/info")
    print(f"  - Predict: POST http://localhost:{SERVER_PORT}/predict")
    print("="*70)
    
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
