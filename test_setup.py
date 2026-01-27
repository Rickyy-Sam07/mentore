"""
Quick Test Script to verify setup before training
"""

import sys
import torch

def check_environment():
    """Check if environment is set up correctly"""
    
    print("="*70)
    print("ENVIRONMENT CHECK")
    print("="*70)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    
    # Check PyTorch
    try:
        print(f"✓ PyTorch version: {torch.__version__}")
    except:
        print("✗ PyTorch not installed!")
        return False
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: Yes")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠ CUDA not available - will use CPU (slower)")
    
    # Check transformers
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except:
        print("✗ Transformers not installed!")
        return False
    
    # Check other packages
    packages = ['pandas', 'numpy', 'sklearn', 'tqdm', 'matplotlib', 'seaborn']
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {pkg}: {version}")
        except ImportError:
            print(f"✗ {pkg} not installed!")
            return False
    
    # Check dataset
    import os
    if os.path.exists('01_Bala-kanda-output.txt'):
        print(f"\n✓ Dataset found: 01_Bala-kanda-output.txt")
        file_size = os.path.getsize('01_Bala-kanda-output.txt') / 1024
        print(f"  Size: {file_size:.2f} KB")
    else:
        print("\n✗ Dataset not found: 01_Bala-kanda-output.txt")
        return False
    
    print("\n" + "="*70)
    print("✓ ALL CHECKS PASSED - READY TO TRAIN!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python train.py")
    print("2. After training, run: python predict.py")
    
    return True


if __name__ == '__main__':
    check_environment()
