"""
Configuration file for RoBERTa Chapter Classification
Optimized for RTX 3050 (6 GB VRAM)
"""

# Data Configuration
DATA_FILE = '01_Bala-kanda-output.txt'
MODEL_DIR = 'saved_models_roberta'
OUTPUT_DIR = 'outputs_roberta'

# Model Configuration
MODEL_NAME = 'roberta-base'  # Using RoBERTa base model
MAX_LENGTH = 128  # Reduced from 512 to save memory
BATCH_SIZE = 8  # Optimized for 6GB VRAM
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 100

# Data Split
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_SEED = 42

# Training Configuration
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16
FP16 = True  # Mixed precision training to save memory
