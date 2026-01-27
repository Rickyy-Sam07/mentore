# Installation Guide for BERT Chapter Classification

## Step 1: Install PyTorch with CUDA Support (FOR GPU)

Your system has RTX 3050, so you need PyTorch with CUDA support for GPU acceleration.

### Uninstall CPU version (if installed):
```bash
pip uninstall torch torchvision torchaudio
```

### Install GPU version for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### OR for CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 2: Install Missing Dependencies

```bash
pip install matplotlib seaborn
```

## Step 3: Verify Installation

```bash
python test_setup.py
```

You should see:
- ✓ CUDA available: Yes
- ✓ GPU: NVIDIA GeForce RTX 3050

## Step 4: Start Training

Once everything is installed:

```bash
python train.py
```

## Important Notes

### If you want to use CPU (slower but works without GPU):
You can still train the model, but it will be much slower (hours instead of minutes).
Just proceed with the current CPU setup.

### Check your CUDA version:
```bash
nvidia-smi
```

Look for "CUDA Version" in the output and install the matching PyTorch version.

### If you get CUDA errors:
Make sure your NVIDIA drivers are up to date:
- Download from: https://www.nvidia.com/Download/index.aspx
- Select RTX 3050 and Windows

## Training Times Comparison

- **GPU (RTX 3050)**: ~20-30 minutes
- **CPU (i7)**: ~3-5 hours

## Recommended Installation Order

1. Update NVIDIA drivers
2. Install PyTorch with CUDA
3. Install other dependencies
4. Run test_setup.py
5. Run train.py
