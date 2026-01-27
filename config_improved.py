"""
Improved Configuration for BERT Chapter Classification
Based on initial training results - targeting better accuracy
"""

# Data Configuration
DATA_FILE = '01_Bala-kanda-output.txt'
MODEL_DIR = 'saved_models_improved'
OUTPUT_DIR = 'outputs_improved'

# Model Configuration
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 256  # INCREASED from 128 - capture more context
BATCH_SIZE = 4  # REDUCED to fit larger sequences in memory
LEARNING_RATE = 3e-5  # INCREASED slightly for faster learning
NUM_EPOCHS = 20  # DOUBLED - more training time
WARMUP_STEPS = 200  # INCREASED warmup

# Data Split
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_SEED = 42

# Training Configuration
GRADIENT_ACCUMULATION_STEPS = 4  # INCREASED - effective batch size still 16
FP16 = True
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for 5 epochs

# Class Weights for Imbalanced Data
USE_CLASS_WEIGHTS = True  # NEW - handle class imbalance
