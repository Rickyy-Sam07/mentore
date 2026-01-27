"""
Quick script to login to Hugging Face Hub
"""
from huggingface_hub import login

print("Hugging Face Login")
print("=" * 50)
print("\nGet your token from: https://huggingface.co/settings/tokens")
print("Create a WRITE token if you don't have one.\n")

token = input("Enter your Hugging Face token: ").strip()

if token:
    try:
        login(token=token, add_to_git_credential=True)
        print("\n✅ Successfully logged in to Hugging Face!")
        print("\nYou can now run: python upload_to_huggingface.py")
    except Exception as e:
        print(f"\n❌ Login failed: {e}")
else:
    print("\n❌ No token provided")
