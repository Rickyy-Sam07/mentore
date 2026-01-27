"""
Dataset Preparation for BERT Chapter Classification
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os
from config import *

class ChapterDataset(Dataset):
    """Custom Dataset for Chapter Classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE, sep='\t', header=None, 
                     names=['entity', 'kanda', 'chapter', 'text'])
    
    # Remove any rows with missing values
    df = df.dropna()
    
    print(f"Total samples: {len(df)}")
    print(f"Unique chapters: {df['chapter'].nunique()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['chapter'])
    
    # Save label encoder for later use
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nLabel encoding saved to {MODEL_DIR}/label_encoder.pkl")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    return df, label_encoder


def split_data(df):
    """Split data into train, validation, and test sets"""
    
    print("\nSplitting data...")
    
    # First split: train + val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=df['label']
    )
    
    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE / (TRAIN_SIZE + VAL_SIZE),
        random_state=RANDOM_SEED,
        stratify=train_val_df['label']
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df


def create_data_loaders(train_df, val_df, test_df, tokenizer):
    """Create PyTorch DataLoaders"""
    
    print("\nCreating DataLoaders...")
    
    train_dataset = ChapterDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    val_dataset = ChapterDataset(
        texts=val_df['text'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    test_dataset = ChapterDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main function to prepare data"""
    
    # Load and preprocess data
    df, label_encoder = load_and_preprocess_data()
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Save splits for reference
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    print(f"\nData splits saved to {OUTPUT_DIR}/")
    
    # Initialize tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, tokenizer
    )
    
    print("\n✓ Data preparation complete!")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, label_encoder


if __name__ == '__main__':
    main()
