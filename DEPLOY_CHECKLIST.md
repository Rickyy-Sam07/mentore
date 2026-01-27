# DEPLOYMENT CHECKLIST

## Before You Deploy

- [x] Models trained (BERT, RoBERTa, DistilBERT)
- [ ] Hugging Face account created
- [ ] Hugging Face token obtained
- [ ] Models uploaded to Hugging Face Hub
- [ ] app.py configured with your HF_REPO_ID
- [ ] Code pushed to GitHub
- [ ] Deployed to Streamlit Cloud

## Commands to Run (In Order)

### 1. Install Hugging Face Hub
```bash
pip install huggingface_hub
```

### 2. Login to Hugging Face
```bash
huggingface-cli login
```
Enter your token from: https://huggingface.co/settings/tokens

### 3. Edit Upload Script
Open `upload_to_huggingface.py` and change line 15:
```python
REPO_ID = "your-username/ramayana-chapter-classifier"  # Your actual username!
```

### 4. Upload Models to Hugging Face
```bash
python upload_to_huggingface.py
```
This will take a few minutes to upload ~2.5GB.

### 5. Update App Configuration
Open `app.py` and change line 24:
```python
HF_REPO_ID = "your-username/ramayana-chapter-classifier"  # Same as above!
```

### 6. Push to GitHub
```bash
git add .
git commit -m "Deploy with Hugging Face model hosting"
git push origin main
```
This should succeed now (models are excluded by .gitignore)!

### 7. Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click "New app"
3. Connect your GitHub repository
4. Set main file: `app.py`
5. Click "Deploy"

## What Happens When Someone Uses Your App

1. User visits your Streamlit Cloud URL
2. App checks if models exist locally
3. If not, downloads from Hugging Face (2-3 minutes first time)
4. Caches models for future use
5. App runs predictions instantly!

## Costs

- Hugging Face: FREE (public models)
- Streamlit Cloud: FREE (community tier)
- GitHub: FREE (public repo)
- **Total: $0/month**

## Verification

After deployment, verify:
- [ ] App loads at Streamlit Cloud URL
- [ ] Models download successfully (check logs)
- [ ] All 3 models (BERT/RoBERTa/DistilBERT) work
- [ ] Predictions are accurate
- [ ] Model comparison tab works

## Troubleshooting

**Upload fails?**
- Check your Hugging Face token is valid
- Verify you're logged in: `huggingface-cli whoami`

**Git push fails?**
- Verify .gitignore includes model directories
- Check: `git status` should not show .pth or .pkl files

**App can't download models?**
- Ensure HF_REPO_ID in app.py matches your repository
- Make sure Hugging Face repo is PUBLIC
- Check requirements_web.txt includes huggingface_hub

**Need help?**
See DEPLOYMENT_HF.md for detailed guide!
