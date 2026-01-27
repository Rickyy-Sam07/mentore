"""
Check your Hugging Face username
"""
from huggingface_hub import whoami

try:
    user_info = whoami()
    username = user_info['name']
    print(f"Your Hugging Face username is: {username}")
    print(f"\nUpdate REPO_ID to: {username}/ramayana-chapter-classifier")
except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure you're logged in: python hf_login.py")
