# Deployment Guide - Hugging Face Hub Strategy

This guide explains how to deploy your Ramayana Chapter Classifier to Streamlit Cloud using Hugging Face Hub for model hosting.

## Why Hugging Face Hub?

- **Free & Professional**: Industry-standard platform for ML models
- **Fast CDN**: Quick downloads from anywhere in the world
- **Version Control**: Track model updates independently from code
- **No Size Limits**: GitHub has 2GB file limits, Hugging Face handles large models easily

## Step-by-Step Deployment

### 1. Prepare Hugging Face Account

1. Create account at [https://huggingface.co/join](https://huggingface.co/join)
2. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click "New token" → Create a **WRITE** token
4. Copy the token (you'll need it in the next step)

### 2. Install Hugging Face CLI

```bash
pip install huggingface_hub
```

### 3. Login to Hugging Face

```bash
huggingface-cli login
```

Paste your token when prompted.

### 4. Upload Models to Hugging Face

1. Edit `upload_to_huggingface.py`:
   ```python
   REPO_ID = "your-username/ramayana-chapter-classifier"
   ```
   Replace `your-username` with your actual Hugging Face username.

2. Run the upload script:
   ```bash
   python upload_to_huggingface.py
   ```

3. Wait for upload to complete (may take several minutes for 2.5GB)

4. Verify at: `https://huggingface.co/your-username/ramayana-chapter-classifier`

### 5. Update Streamlit App Configuration

1. Edit `app.py` line 24:
   ```python
   HF_REPO_ID = "your-username/ramayana-chapter-classifier"
   ```

2. Update `requirements_web.txt` to include:
   ```
   huggingface_hub>=0.20.0
   ```

### 6. Push Code to GitHub

```bash
# Stage all files (models are excluded by .gitignore)
git add .

# Commit changes
git commit -m "Add Hugging Face model download support"

# Push to GitHub
git push origin main
```

This time the push will succeed because model files are excluded!

### 7. Deploy to Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `your-username/mentore`
5. Set main file path: `app.py`
6. Click "Deploy"

### 8. First Run Behavior

When someone visits your app for the first time:
1. Streamlit loads the code from GitHub
2. App detects missing model files
3. Downloads models from Hugging Face Hub (one-time, ~2-3 minutes)
4. Caches models for future use
5. App is ready!

Subsequent visits are instant - models are cached.

## Troubleshooting

### "Cannot find repository" error
- Make sure your Hugging Face repository is **public**
- Verify the REPO_ID in app.py matches your Hugging Face repo exactly

### Models not downloading
- Check Streamlit Cloud logs for error messages
- Verify models were uploaded successfully to Hugging Face
- Ensure huggingface_hub is in requirements_web.txt

### App runs locally but not on Streamlit Cloud
- Check that all dependencies are in requirements_web.txt
- Verify Python version compatibility (use Python 3.9-3.11)

## Model Updates

To update models after training:
1. Train new model locally
2. Run `python upload_to_huggingface.py` again
3. Restart Streamlit Cloud app (it will download fresh models)

No need to redeploy the Streamlit app unless you change the code!

## Cost

- **Hugging Face**: FREE for public models (unlimited downloads)
- **Streamlit Cloud**: FREE tier (1 app, community support)
- **GitHub**: FREE for public repositories
- **Total**: $0/month 🎉

## Alternative: Private Models

If you need private models:
1. Make Hugging Face repository private
2. In Streamlit Cloud → Settings → Secrets, add:
   ```toml
   HF_TOKEN = "your-huggingface-token"
   ```
3. Update app.py to use the token:
   ```python
   token = st.secrets.get("HF_TOKEN", None)
   hf_hub_download(..., token=token)
   ```

## Performance

- **First load**: 2-3 minutes (one-time model download)
- **Subsequent loads**: <5 seconds (cached models)
- **Inference**: Same speed as local (models run on Streamlit servers)

## Support

- Hugging Face Docs: https://huggingface.co/docs/hub
- Streamlit Docs: https://docs.streamlit.io/streamlit-cloud
- GitHub Issues: Create issue in your repository
