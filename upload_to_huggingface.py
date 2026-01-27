"""
Upload trained models to Hugging Face Hub for deployment.

Before running this script:
1. Install huggingface_hub: pip install huggingface_hub
2. Login to Hugging Face: huggingface-cli login
   (Get your token from https://huggingface.co/settings/tokens)
3. Update REPO_ID below with your username
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = "Sam-veda/ramayana-chapter-classifier"
MODELS_TO_UPLOAD = {
    "bert": "saved_models",
    "roberta": "saved_models_roberta",
    "distilbert": "saved_models_distilbert"
}

def upload_models():
    """Upload all trained models to Hugging Face Hub."""
    
    # Initialize Hugging Face API
    api = HfApi()
    
    print(f"Creating repository: {REPO_ID}")
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
        print("✓ Repository created/verified")
    except Exception as e:
        print(f"✗ Error creating repository: {e}")
        print("\n⚠ Possible solutions:")
        print("  1. Go to https://huggingface.co/settings/tokens")
        print("  2. Create a NEW token with 'write' permission")
        print("  3. Run: python hf_login.py (with the new token)")
        print("\n  OR manually create the repo:")
        print(f"  - Visit: https://huggingface.co/new")
        print(f"  - Name: ramayana-chapter-classifier")
        print(f"  - Type: Model")
        print(f"  - Then run this script again")
        return
    
    # Upload each model
    for model_name, model_dir in MODELS_TO_UPLOAD.items():
        if not os.path.exists(model_dir):
            print(f"⚠ Skipping {model_name}: {model_dir} not found")
            continue
        
        print(f"\n📤 Uploading {model_name} model from {model_dir}...")
        
        try:
            # Upload the entire model directory
            api.upload_folder(
                folder_path=model_dir,
                repo_id=REPO_ID,
                path_in_repo=model_name,
                commit_message=f"Upload {model_name} model",
                token=True  # Use stored token
            )
            print(f"✓ {model_name} uploaded successfully")
            
        except Exception as e:
            print(f"✗ Error uploading {model_name}: {e}")
    
    print(f"\n✅ Upload complete! View your models at:")
    print(f"   https://huggingface.co/{REPO_ID}")
    print(f"\n📝 Next steps:")
    print(f"   1. Update app.py with REPO_ID = '{REPO_ID}'")
    print(f"   2. Push code to GitHub (models are now in .gitignore)")
    print(f"   3. Deploy to Streamlit Cloud")

if __name__ == "__main__":
    # Check if REPO_ID has been updated
    if "YOUR_USERNAME" in REPO_ID:
        print("⚠ ERROR: Please update REPO_ID in this script with your Hugging Face username!")
        print("   Example: REPO_ID = 'sambhranta/ramayana-chapter-classifier'")
        exit(1)
    
    upload_models()
