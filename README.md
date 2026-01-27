# BERT Chapter Classification

Classification of Ramayana chapters using BERT model (PyTorch)

## Dataset
- **File**: `01_Bala-kanda-output.txt`
- **Total samples**: 1,832
- **Number of classes**: 77 chapters
- **Task**: Given a text passage, predict which chapter it belongs to

## Hardware Optimization
Optimized for **RTX 3050 (6GB VRAM)**:
- Batch size: 8
- Max sequence length: 128
- Mixed precision training (FP16)
- Gradient accumulation: 2 steps
- Effective batch size: 16

## Installation

### 1. Install PyTorch with CUDA support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install other dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model
```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Split data into train (80%), validation (10%), test (10%)
- Train BERT model for chapter classification
- Save the best model to `saved_models/best_model.pth`
- Generate training history plot
- Evaluate on test set and save classification report

**Training time**: Approximately 20-30 minutes on RTX 3050

### Step 2: Make Predictions
```bash
python predict.py
```

This will:
- Load the trained model
- Run example predictions
- Enter interactive mode for custom predictions

### Example Usage
```python
from predict import ChapterPredictor

# Initialize predictor
predictor = ChapterPredictor(
    model_path='saved_models/best_model.pth',
    label_encoder_path='saved_models/label_encoder.pkl'
)

# Make prediction
text = "O Sage, I would hear of such a man from you, who art able to describe him to me."
chapter, confidence = predictor.predict(text)
print(f"Predicted Chapter: {chapter}")
print(f"Confidence: {confidence:.2%}")

# Get top 5 predictions with probabilities
result = predictor.predict(text, return_probabilities=True)
for pred in result['top_predictions']:
    print(f"{pred['chapter']}: {pred['probability']:.2%}")
```

## Project Structure
```
mentore/
├── 01_Bala-kanda-output.txt    # Dataset
├── config.py                    # Configuration settings
├── dataset_preparation.py       # Data loading and preprocessing
├── train.py                     # Training script
├── predict.py                   # Inference script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── saved_models/                # Trained models (created after training)
│   ├── best_model.pth
│   └── label_encoder.pkl
└── outputs/                     # Training outputs (created after training)
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── training_history.png
    ├── classification_report.txt
    └── training_summary.txt
```

## Model Architecture
- **Base Model**: `bert-base-uncased`
- **Total Parameters**: ~110M
- **Input**: Text passages (max 128 tokens)
- **Output**: Chapter classification (77 classes)
- **Dropout**: 0.3

## Training Configuration
- **Learning Rate**: 2e-5
- **Epochs**: 10
- **Warmup Steps**: 100
- **Optimizer**: AdamW
- **Scheduler**: Linear with warmup
- **Loss Function**: CrossEntropyLoss

## Performance Optimization Tips

### For Better Memory Usage:
- Reduce `MAX_LENGTH` in `config.py` (e.g., 64 or 96)
- Reduce `BATCH_SIZE` (e.g., 4)
- Increase `GRADIENT_ACCUMULATION_STEPS`

### For Better Performance:
- Increase `NUM_EPOCHS` (e.g., 15-20)
- Try `bert-base-cased` for better handling of proper nouns
- Experiment with different learning rates

### For Faster Training:
- Reduce `MAX_LENGTH`
- Reduce number of epochs
- Use `DistilBERT` instead of BERT

## Troubleshooting

### CUDA Out of Memory
If you get OOM errors:
1. Reduce `BATCH_SIZE` to 4 or 2
2. Reduce `MAX_LENGTH` to 64
3. Set `FP16 = True` (if not already)

### Low Accuracy
1. Increase training epochs
2. Try different learning rates (1e-5 or 3e-5)
3. Increase `MAX_LENGTH` if text is being truncated
4. Check class imbalance in dataset

## Results
After training, check:
- `outputs/training_history.png` - Training and validation curves
- `outputs/classification_report.txt` - Detailed metrics per class
- `outputs/training_summary.txt` - Overall training summary

## License
This project is for educational purposes.

## Author
Created for Ramayana text classification using BERT
