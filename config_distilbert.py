"""
Configuration for DistilBERT Chapter Classification
For Computer 1 (Remote Server)
"""

# Data Configuration
DATA_FILE = '01_Bala-kanda-output.txt'
MODEL_DIR = 'saved_models_distilbert'
OUTPUT_DIR = 'outputs_distilbert'

# Model Configuration
MODEL_NAME = 'distilbert-base-uncased'  # Faster, smaller than BERT
MAX_LENGTH = 128
BATCH_SIZE = 16  # DistilBERT is faster, can use larger batch
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 100

# Data Split
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_SEED = 42

# Training Configuration
GRADIENT_ACCUMULATION_STEPS = 1
FP16 = True

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
