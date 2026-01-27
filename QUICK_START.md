# 🎯 BERT Chapter Classification - Project Summary

## 📋 What This Does

This project classifies text passages from the Ramayana into their respective chapters using BERT.

**Example:**
- **Input**: "O Sage, I would hear of such a man from you, who art able to describe him to me."
- **Output**: "01_Shri Narada relates to Valmiki the story of Rama" (with 95% confidence)

## 📊 Dataset Information

- **File**: `01_Bala-kanda-output.txt`
- **Format**: Tab-separated (Entity, Kanda, Chapter, Text)
- **Total Samples**: 1,832 passages
- **Number of Classes**: 77 different chapters
- **Data Split**: 
  - Training: 80% (1,466 samples)
  - Validation: 10% (183 samples)
  - Testing: 10% (183 samples)

## 🚀 Quick Start

### Option 1: Run Setup Script (Easiest)
```bash
setup.bat
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install matplotlib seaborn

# Test environment
python test_setup.py

# Train the model
python train.py

# Make predictions
python predict.py
```

## 📁 Files Created

### Core Files
- **config.py** - All configuration settings (batch size, learning rate, etc.)
- **dataset_preparation.py** - Data loading and preprocessing
- **train.py** - Training script with all optimization
- **predict.py** - Inference script for predictions
- **requirements.txt** - Python dependencies

### Documentation
- **README.md** - Detailed documentation
- **INSTALLATION.md** - GPU setup guide
- **QUICK_START.md** - This file

### Utility Files
- **test_setup.py** - Environment verification
- **setup.bat** - Automated setup script

## 🔧 Hardware Optimization

Your RTX 3050 with 6GB VRAM is perfectly suited for this task!

**Optimizations Applied:**
- ✓ Batch size: 8 (fits in 6GB VRAM)
- ✓ Max sequence length: 128 tokens
- ✓ Mixed precision training (FP16) - 2x faster, 50% less memory
- ✓ Gradient accumulation - effective batch size of 16
- ✓ Efficient memory management

**Expected Performance:**
- Training time: ~20-30 minutes (GPU) or ~3-5 hours (CPU)
- Memory usage: ~4-5 GB VRAM
- Expected accuracy: 70-85% (depends on class distribution)

## 🎓 Model Architecture

```
Input Text (max 128 tokens)
         ↓
BERT Tokenizer (bert-base-uncased)
         ↓
BERT Encoder (~110M parameters)
         ↓
Dropout Layer (0.3)
         ↓
Linear Classifier (768 → 77 classes)
         ↓
Softmax
         ↓
Chapter Prediction + Confidence
```

## 📈 What Happens During Training

1. **Data Preparation** (1-2 minutes)
   - Loads dataset
   - Splits into train/val/test
   - Tokenizes all text
   - Creates batches

2. **Training** (20-30 minutes on GPU)
   - 10 epochs
   - Progress bar for each epoch
   - Automatic model saving (best validation accuracy)
   - Memory-efficient training with FP16

3. **Evaluation** (1-2 minutes)
   - Tests on held-out test set
   - Generates classification report
   - Creates accuracy plots
   - Saves all results

## 📊 Output Files (After Training)

### saved_models/
- `best_model.pth` - Trained BERT model
- `label_encoder.pkl` - Chapter name mappings

### outputs/
- `train.csv`, `val.csv`, `test.csv` - Data splits
- `training_history.png` - Accuracy/loss curves
- `classification_report.txt` - Detailed metrics per class
- `training_summary.txt` - Overall results

## 🔮 Making Predictions

### Interactive Mode
```bash
python predict.py
```

### Programmatic Use
```python
from predict import ChapterPredictor

predictor = ChapterPredictor(
    'saved_models/best_model.pth',
    'saved_models/label_encoder.pkl'
)

text = "Your input text here..."
chapter, confidence = predictor.predict(text)
print(f"{chapter} ({confidence:.1%})")
```

## ⚙️ Customization

### To change batch size (if you get memory errors):
Edit `config.py`:
```python
BATCH_SIZE = 4  # Reduce from 8
```

### To train longer:
Edit `config.py`:
```python
NUM_EPOCHS = 20  # Increase from 10
```

### To use longer sequences:
Edit `config.py`:
```python
MAX_LENGTH = 256  # Increase from 128
```

## ⚠️ Important Notes

### GPU vs CPU
- **Current setup**: CPU version of PyTorch installed
- **For GPU**: Follow [INSTALLATION.md](INSTALLATION.md) to install CUDA-enabled PyTorch
- **Performance**: GPU is ~10x faster than CPU for training

### First Run
The first time you run training:
- BERT model will be downloaded (~440 MB)
- This happens automatically
- Subsequent runs use cached model

### Memory Issues
If you get "CUDA Out of Memory":
1. Reduce `BATCH_SIZE` to 4 or 2
2. Reduce `MAX_LENGTH` to 64
3. Close other GPU applications

## 📧 Example Use Cases

### 1. Academic Research
Classify passages from religious texts automatically

### 2. Digital Library
Organize and categorize large text collections

### 3. Text Analysis
Understand distribution of themes across chapters

### 4. Content Recommendation
Suggest similar chapters based on content

## 🎯 Expected Results

### Best Case (balanced classes):
- Training Accuracy: ~95%
- Validation Accuracy: ~85%
- Test Accuracy: ~80-85%

### Realistic Case (imbalanced classes):
- Training Accuracy: ~90%
- Validation Accuracy: ~75-80%
- Test Accuracy: ~70-75%

Some chapters have more samples than others, which affects accuracy.

## 🔍 Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "CUDA not available"
See [INSTALLATION.md](INSTALLATION.md) for GPU setup

### "File not found: 01_Bala-kanda-output.txt"
Make sure you're running scripts from the `mentore` directory

### Low accuracy
- Increase NUM_EPOCHS
- Increase MAX_LENGTH
- Check for class imbalance in dataset

## 📚 Learn More

- **BERT Paper**: https://arxiv.org/abs/1810.04805
- **Hugging Face Docs**: https://huggingface.co/docs/transformers
- **PyTorch Tutorials**: https://pytorch.org/tutorials

## 🎉 Next Steps

1. ✅ Run `setup.bat` or install dependencies manually
2. ✅ Run `python test_setup.py` to verify
3. ✅ Run `python train.py` to train model
4. ✅ Run `python predict.py` to test predictions
5. ✅ Experiment with different texts!

---

**Ready to get started?** Run `setup.bat` now!
