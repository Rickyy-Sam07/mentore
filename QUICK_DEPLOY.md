# Quick Start - Cloud Deployment

## For Deployment (Recommended: Hugging Face Hub)

### 1. Upload Models (One-time setup)
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Edit upload script with your username
# Change line: REPO_ID = "your-username/ramayana-chapter-classifier"

# Upload models
python upload_to_huggingface.py
```

### 2. Configure App
Edit `app.py` line 24:
```python
HF_REPO_ID = "your-username/ramayana-chapter-classifier"
```

### 3. Deploy to Streamlit Cloud
```bash
# Add huggingface support
pip install -r requirements_web.txt

# Push to GitHub (models excluded by .gitignore)
git add .
git commit -m "Deploy with Hugging Face models"
git push origin main

# Then deploy at: https://share.streamlit.io
```

## For Local Testing

### Test with Model Download
```bash
# Temporarily rename model directories to test download
mv saved_models saved_models_backup
mv saved_models_roberta saved_models_roberta_backup
mv saved_models_distilbert saved_models_distilbert_backup

# Run app (should download from Hugging Face)
python -m streamlit run app.py

# Restore backups after testing
mv saved_models_backup saved_models
mv saved_models_roberta_backup saved_models_roberta
mv saved_models_distilbert_backup saved_models_distilbert
```

## File Structure After Deployment

```
your-repo/
├── .gitignore                    # Excludes model files
├── app.py                        # Downloads models from HF
├── upload_to_huggingface.py      # Upload script
├── requirements.txt              # Training dependencies
├── requirements_web.txt          # Streamlit dependencies  
├── train.py                      # BERT training
├── train_roberta.py              # RoBERTa training
├── train_distilbert.py           # DistilBERT training
├── config*.py                    # Configurations
├── dataset_preparation.py        # Data utilities
└── predict*.py                   # CLI inference scripts
```

Models live on Hugging Face, not in GitHub!

## See Full Guide

Read [DEPLOYMENT_HF.md](DEPLOYMENT_HF.md) for complete instructions.
