# Setup Guide for Multi-Computer Deployment

## Overview
- **Computer 1**: Runs DistilBERT model with FastAPI server (exposed via ngrok)
- **Computer 2**: Runs Streamlit web app with BERT, RoBERTa (local) + DistilBERT (remote)

---

## Computer 1 Setup (DistilBERT Server)

### Step 1: Train DistilBERT Model
```bash
# Copy dataset to Computer 1
# Place 01_Bala-kanda-output.txt in the same directory

# Train the model
python train_distilbert.py
```

### Step 2: Install Dependencies
```bash
pip install fastapi uvicorn pyngrok
```

### Step 3: Start FastAPI Server
```bash
python server.py
```

Server will start on `http://0.0.0.0:8000`

### Step 4: Create Ngrok Tunnel

#### Option A: Using pyngrok (Easiest)
```python
# Add to server.py or run separately
from pyngrok import ngrok

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print(f"Ngrok URL: {public_url}")
```

#### Option B: Using ngrok CLI
```bash
# Download ngrok from https://ngrok.com/download
# Sign up for free account at https://ngrok.com

# Authenticate (one-time)
ngrok config add-authtoken YOUR_AUTH_TOKEN

# Start tunnel
ngrok http 8000
```

### Step 5: Copy Ngrok URL
You'll see output like:
```
Forwarding  https://abc123.ngrok.io -> http://localhost:8000
```

**Copy the HTTPS URL** (e.g., `https://abc123.ngrok.io`)

### Step 6: Test the API
```bash
# Health check
curl https://abc123.ngrok.io/health

# Test prediction
curl -X POST https://abc123.ngrok.io/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "O Sage, I would hear of such a man from you..."}'
```

---

## Computer 2 Setup (Streamlit App)

### Step 1: Install Dependencies
```bash
pip install -r requirements_web.txt
```

### Step 2: Test API Client
```bash
# Edit api_client.py and update NGROK_URL
python api_client.py
```

### Step 3: Run Streamlit App
```bash
streamlit run app.py
```

App will open at `http://localhost:8501`

### Step 4: Configure in Web Interface
1. Open Streamlit app in browser
2. In sidebar, select "DistilBERT (Remote)"
3. Enter the ngrok URL from Computer 1
4. Click "Test Connection"
5. If successful, you can now use the remote model!

---

## Usage

### Single Prediction
1. Select model (BERT/RoBERTa/DistilBERT)
2. Enter text
3. Click "Predict Chapter"
4. View results with confidence scores

### Compare Models
1. Go to "Compare Models" tab
2. Enter text
3. Click "Compare All Models"
4. See side-by-side comparison

### Batch Prediction
1. Go to "Batch Prediction" tab
2. Upload CSV file
3. Select text column
4. Run batch prediction

---

## Troubleshooting

### Computer 1 Issues

**Problem**: Model not found
```bash
# Solution: Train the model first
python train_distilbert.py
```

**Problem**: Port already in use
```python
# Solution: Change port in config_distilbert.py
SERVER_PORT = 8001  # Use different port
```

**Problem**: Ngrok tunnel closed
```bash
# Solution: Free ngrok tunnels close after 2 hours
# Restart ngrok and update URL in Streamlit app
```

### Computer 2 Issues

**Problem**: Connection timeout
```
Solution:
1. Check if Computer 1 server is running
2. Verify ngrok URL is correct
3. Check firewall settings
```

**Problem**: Model not loading
```bash
# Solution: Train models first
python train.py  # BERT
python train_roberta.py  # RoBERTa
```

---

## API Endpoints

### GET /
Root endpoint with API information

### GET /health
Health check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### GET /info
Model information
```json
{
  "model_name": "distilbert-base-uncased",
  "num_classes": 77,
  "max_length": 128,
  "device": "cuda"
}
```

### POST /predict
Make prediction
```json
Request:
{
  "text": "Your text here",
  "top_k": 5
}

Response:
{
  "predicted_chapter": "Chapter name",
  "confidence": 0.85,
  "top_predictions": [
    {"chapter": "Ch1", "probability": 0.85},
    {"chapter": "Ch2", "probability": 0.10}
  ]
}
```

---

## Ngrok Free Tier Limits

- **Session limit**: 2 hours (tunnel closes, need to restart)
- **Concurrent tunnels**: 1
- **Requests**: 40 requests/minute

**Tip**: For production, upgrade to ngrok paid plan or deploy to cloud

---

## Alternative Deployment (Cloud)

### Deploy Server to Cloud (Recommended for Production)

**Option 1: Railway.app**
```bash
# Free tier available
# Auto-deploy from GitHub
```

**Option 2: Render.com**
```bash
# Free tier available
# Deploy FastAPI app
```

**Option 3: AWS EC2**
```bash
# More control
# Requires setup
```

---

## Security Notes

1. **API Authentication**: Current setup has no auth (OK for testing)
2. **HTTPS**: Ngrok provides HTTPS automatically
3. **Rate Limiting**: Add rate limiting for production
4. **CORS**: Currently allows all origins (restrict in production)

---

## Performance Tips

### Computer 1 (Server)
- Use GPU for faster inference
- Enable FP16 if supported
- Use batch prediction for multiple requests

### Computer 2 (Streamlit)
- Cache model loading with @st.cache_resource
- Use async requests for remote calls
- Implement request retries

---

## Next Steps

1. Test all three models
2. Compare accuracy and speed
3. Add authentication if needed
4. Deploy to cloud for 24/7 availability
5. Add monitoring and logging

---

## Quick Commands Reference

**Computer 1:**
```bash
# Start server
python server.py

# Start ngrok (separate terminal)
ngrok http 8000
```

**Computer 2:**
```bash
# Start Streamlit
streamlit run app.py

# Test API client
python api_client.py
```

---

## Support

If you encounter issues:
1. Check server logs on Computer 1
2. Check ngrok dashboard: https://dashboard.ngrok.com
3. Test API with curl or Postman
4. Verify model files exist in saved_models_distilbert/
