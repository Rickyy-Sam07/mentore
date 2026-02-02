"""
Streamlit Web Interface for Chapter Classification
Supports BERT, RoBERTa, and DistilBERT (downloads from Hugging Face Hub)
"""

import streamlit as st
import torch
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer
import pickle
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from huggingface_hub import hf_hub_download
import base64

from models import BertChapterClassifier, RobertaChapterClassifier, DistilBertChapterClassifier
from config import MODEL_DIR as BERT_MODEL_DIR, MAX_LENGTH as BERT_MAX_LENGTH
from config_roberta import MODEL_DIR as ROBERTA_MODEL_DIR, MAX_LENGTH as ROBERTA_MAX_LENGTH
from config_distilbert import MODEL_DIR as DISTILBERT_MODEL_DIR, MAX_LENGTH as DISTILBERT_MAX_LENGTH

# Hugging Face configuration
HF_REPO_ID = "Sam-veda/ramayana-chapter-classifier"

# Page config
st.set_page_config(
    page_title="Pouranic Topic Classification",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and encode background image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get base64 string of background image
bg_image_path = "br_img.png"
if os.path.exists(bg_image_path):
    bg_image_base64 = get_base64_image(bg_image_path)
    bg_style = f"background-image: url('data:image/png;base64,{bg_image_base64}');"
else:
    bg_style = "background-color: #f0f2f6;"

# Custom CSS with background image and glassmorphism
st.markdown(f"""
<style>
    .stApp {{
        {bg_style}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Glassmorphism effect for all containers */
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: none;
        -webkit-backdrop-filter: none;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }}
    
    .metric-card {{
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: none;
        -webkit-backdrop-filter: none;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }}
    
    /* Input text styling with no blur and full transparency */
    .stTextInput > div > div, .stTextArea > div > div {{
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
        pointer-events: auto !important;
    }}
    
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea,
    [data-baseweb="textarea"],
    [data-baseweb="input"] {{
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 0.3rem !important;
        color: #000000 !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        color-scheme: light !important;
        padding: 0.3rem 0.4rem !important;
        margin: 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        font-size: 0.9rem !important;
        min-height: 2rem !important;
        pointer-events: auto !important;
        cursor: text !important;
    }}
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {{
        color: #2a2a2a !important;
        font-weight: 500 !important;
    }}
    
    /* Force all textarea elements with blur */
    textarea, textarea[class*="st"] {{
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        color: #000000 !important;
        color-scheme: light !important;
        padding: 0.3rem 0.4rem !important;
        margin: 0 !important;
        font-size: 0.9rem !important;
        min-height: 2rem !important;
        max-height: 10rem !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        box-shadow: none !important;
        pointer-events: auto !important;
        cursor: text !important;
    }}
    
    /* Reduce label spacing */
    .stTextInput label, .stTextArea label {{
        margin-bottom: 0.3rem !important;
        padding: 0 !important;
    }}
    
    /* Compact text input/textarea containers */
    .stTextInput, .stTextArea {{
        margin-bottom: 0.5rem !important;
        padding: 0 !important;
        max-width: 40% !important;
    }}
    
    .stTextInput > label, .stTextArea > label {{
        padding-bottom: 0.2rem !important;
    }}
    
    /* Selectbox styling */
    .stSelectbox > div > div {{
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: none;
        -webkit-backdrop-filter: none;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 0.5rem;
    }}
    
    .stSelectbox label, .stTextInput label, .stTextArea label {{
        color: #1a1a1a !important;
        font-weight: 600;
    }}
    
    /* Sidebar blur transparent effect */
    [data-testid="stSidebar"] {{
        background: rgba(128, 128, 128, 0.15) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(128, 128, 128, 0.3);
    }}
    
    [data-testid="stSidebar"] * {{
        color: #1a1a1a !important;
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: #1a1a1a !important;
    }}
    
    /* Main content area */
    .block-container {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: none;
        -webkit-backdrop-filter: none;
        border-radius: 1rem;
        padding: 2rem;
    }}
    
    /* Cards and expanders */
    .stExpander, [data-testid="stExpander"] {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: none;
        -webkit-backdrop-filter: none;
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 0.5rem;
    }}
</style>
""", unsafe_allow_html=True)


class LocalModelPredictor:
    """Predictor for local/downloaded BERT/RoBERTa/DistilBERT models"""
    
    def __init__(self, model_type):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def download_model_files(self, model_dir, hf_subfolder):
        """Download model files from Hugging Face Hub if not present locally"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        files_to_download = ['best_model.pth', 'label_encoder.pkl']
        
        for filename in files_to_download:
            local_path = os.path.join(model_dir, filename)
            
            # Skip if file already exists locally
            if os.path.exists(local_path):
                continue
            
            try:
                # Download from Hugging Face
                with st.spinner(f'Downloading {self.model_type} {filename}...'):
                    downloaded_path = hf_hub_download(
                        repo_id=HF_REPO_ID,
                        filename=filename,
                        subfolder=hf_subfolder,
                        cache_dir=None  # Use default cache
                    )
                    
                    # Copy to local directory
                    import shutil
                    shutil.copy(downloaded_path, local_path)
                    
            except Exception as e:
                st.error(f"Error downloading {filename}: {e}")
                st.info("💡 Make sure models are uploaded to Hugging Face. Run: python upload_to_huggingface.py")
                raise
    
    def load_model(self):
        """Load model, tokenizer, and label encoder"""
        
        if self.model_type == "BERT":
            model_dir = BERT_MODEL_DIR
            max_length = BERT_MAX_LENGTH
            tokenizer_name = 'bert-base-uncased'
            model_class = BertChapterClassifier
            hf_subfolder = "bert"
        elif self.model_type == "RoBERTa":
            model_dir = ROBERTA_MODEL_DIR
            max_length = ROBERTA_MAX_LENGTH
            tokenizer_name = 'roberta-base'
            model_class = RobertaChapterClassifier
            hf_subfolder = "roberta"
        else:  # DistilBERT
            model_dir = DISTILBERT_MODEL_DIR
            max_length = DISTILBERT_MAX_LENGTH
            tokenizer_name = 'distilbert-base-uncased'
            model_class = DistilBertChapterClassifier
            hf_subfolder = "distilbert"
        
        self.max_length = max_length
        
        # Download model files if needed (only runs on first load)
        self.download_model_files(model_dir, hf_subfolder)
        
        # Load label encoder
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.num_classes = len(self.label_encoder.classes_)
        
        # Load tokenizer
        if self.model_type == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        elif self.model_type == "RoBERTa":
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        
        # Load model
        self.model = model_class(num_classes=self.num_classes)
        checkpoint = torch.load(
            os.path.join(model_dir, 'best_model.pth'),
            map_location=self.device,
            weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text, top_k=5):
        """Make prediction"""
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            _, prediction = torch.max(outputs, dim=1)
        
        predicted_label = prediction.item()
        predicted_chapter = self.label_encoder.inverse_transform([predicted_label])[0]
        confidence = probabilities[0][predicted_label].item()
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities[0], k=min(top_k, self.num_classes))
        top_chapters = self.label_encoder.inverse_transform(top_indices.cpu().numpy())
        
        top_predictions = [
            {"chapter": chapter, "probability": prob.item()}
            for chapter, prob in zip(top_chapters, top_probs)
        ]
        
        return {
            "predicted_chapter": predicted_chapter,
            "confidence": confidence,
            "top_predictions": top_predictions
        }


@st.cache_resource
def load_local_model(model_type):
    """Load local model with caching"""
    return LocalModelPredictor(model_type)


def main():
    # Header
    st.markdown('<div class="main-header">Pouranic Topic Classification</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_option = st.sidebar.radio(
        "Choose Model:",
        ["BERT", "RoBERTa", "DistilBERT"],
        help="Select which model to use for prediction"
    )
    
    # Model info
    if model_option:
        st.sidebar.info(f"Selected: {model_option}")
        if torch.cuda.is_available():
            st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.sidebar.warning("Using CPU")
    
    # Top-k predictions
    top_k = st.sidebar.slider("Top-K Predictions:", 1, 10, 5)
    
    # Main content
    tab1, tab2 = st.tabs(["Single Prediction", "Compare Models"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.subheader("Enter Text for Classification")
        
        # Text input
        text_input = st.text_area(
            "Input Text:",
            height=200,
            placeholder="Enter text from Ramayana...",
            help="Enter a passage to classify which chapter it belongs to"
        )
        
        # Predict button
        if st.button("Predict Chapter", type="primary"):
            if not text_input:
                st.warning("Please enter some text!")
            else:
                with st.spinner(f"Making prediction with {model_option}..."):
                    try:
                        predictor = load_local_model(model_option)
                        result = predictor.predict(text_input, top_k=top_k)
                        display_results(result, model_option)
                    except FileNotFoundError as e:
                        st.error(f"Model not found! Please train {model_option} model first.")
                        st.info(f"Run: python train_{'distilbert' if model_option == 'DistilBERT' else model_option.lower()}.py")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Tab 2: Compare Models
    with tab2:
        st.subheader("Compare All Models")
        
        compare_text = st.text_area(
            "Input Text for Comparison:",
            height=150,
            placeholder="Enter text to compare all models..."
        )
        
        if st.button("Compare All Models", type="primary"):
            if not compare_text:
                st.warning("Please enter some text!")
            else:
                results = {}
                
                # Run models sequentially with memory cleanup
                models_to_compare = ["BERT", "RoBERTa", "DistilBERT"]
                
                for model_name in models_to_compare:
                    with st.spinner(f"Running {model_name} prediction..."):
                        try:
                            predictor = load_local_model(model_name)
                            results[model_name] = predictor.predict(compare_text, top_k=3)
                            st.success(f"✓ {model_name} complete")
                            
                            # Force garbage collection
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            results[model_name] = {"error": f"Model error: {str(e)[:50]}"}
                            st.warning(f"⚠ {model_name} not available")
                
                # Display comparison
                display_comparison(results)


def display_results(result, model_name):
    """Display prediction results"""
    
    st.success("Prediction Complete!")
    
    # Main prediction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Predicted Chapter")
        st.info(result['predicted_chapter'])
    
    with col2:
        st.markdown("### Confidence")
        st.metric("Score", f"{result['confidence']:.1%}")
    
    # Confidence bar
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['confidence'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, width='stretch')
    
    # Top predictions
    st.markdown("### Top Predictions")
    
    top_preds = result.get('top_predictions', [])
    
    for i, pred in enumerate(top_preds, 1):
        prob = pred['probability']
        chapter = pred['chapter']
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{i}.** {chapter}")
        with col2:
            st.progress(prob)
            st.caption(f"{prob:.1%}")


def display_comparison(results):
    """Display model comparison"""
    
    st.markdown("### Model Comparison Results")
    
    # Create comparison table
    comparison_data = []
    
    for model_name, result in results.items():
        if 'error' not in result:
            comparison_data.append({
                "Model": model_name,
                "Predicted Chapter": result['predicted_chapter'],
                "Confidence": f"{result['confidence']:.2%}"
            })
        else:
            comparison_data.append({
                "Model": model_name,
                "Predicted Chapter": "Error",
                "Confidence": result['error']
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, width='stretch')
        
        # Confidence comparison chart
        valid_results = [r for r in results.values() if 'error' not in r]
        
        if valid_results:
            fig = px.bar(
                x=[k for k, v in results.items() if 'error' not in v],
                y=[v['confidence'] for v in valid_results],
                labels={'x': 'Model', 'y': 'Confidence'},
                title="Confidence Comparison"
            )
            st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    main()
