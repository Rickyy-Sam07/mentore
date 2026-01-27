"""
RoBERTa Model for Chapter Classification
Optimized for RTX 3050 (6GB VRAM)
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from config_roberta import *


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


class RobertaChapterClassifier(nn.Module):
    """RoBERTa-based Chapter Classifier"""
    
    def __init__(self, num_classes, dropout=0.3):
        super(RobertaChapterClassifier, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token in RoBERTa)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


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


def train_epoch(model, data_loader, optimizer, device, scheduler, scaler):
    """Train for one epoch"""
    
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=FP16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        # Calculate accuracy
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)
        losses.append(loss.item() * GRADIENT_ACCUMULATION_STEPS)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': np.mean(losses[-50:]),
            'acc': (correct_predictions.double() / total_predictions).item()
        })
    
    return correct_predictions.double() / total_predictions, np.mean(losses)


def eval_model(model, data_loader, device):
    """Evaluate the model"""
    
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Evaluating')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with autocast(enabled=FP16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            _, preds = torch.max(outputs, dim=1)
            
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': np.mean(losses),
                'acc': (correct_predictions.double() / total_predictions).item()
            })
    
    return (correct_predictions.double() / total_predictions, 
            np.mean(losses), 
            np.array(all_preds), 
            np.array(all_labels))


def plot_training_history(train_accs, val_accs, train_losses, val_losses):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(train_accs, label='Train Accuracy', marker='o')
    ax1.plot(val_accs, label='Validation Accuracy', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('RoBERTa Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(train_losses, label='Train Loss', marker='o')
    ax2.plot(val_losses, label='Validation Loss', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('RoBERTa Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300)
    print(f"\n✓ Training history saved to {OUTPUT_DIR}/training_history.png")
    plt.close()


def main():
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare data
    print("\n" + "="*50)
    print("PREPARING DATA")
    print("="*50)
    
    df, label_encoder = load_and_preprocess_data()
    num_classes = len(label_encoder.classes_)
    
    train_df, val_df, test_df = split_data(df)
    
    # Save splits for reference
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    # Initialize tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, tokenizer
    )
    
    # Initialize model
    print("\n" + "="*50)
    print("INITIALIZING ROBERTA MODEL")
    print("="*50)
    
    model = RobertaChapterClassifier(num_classes=num_classes)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    
    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=FP16)
    
    # Training loop
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    best_val_acc = 0
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
        print('-' * 50)
        
        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, device, scheduler, scaler
        )
        
        val_acc, val_loss, _, _ = eval_model(model, val_loader, device)
        
        train_accs.append(train_acc.item())
        val_accs.append(val_acc.item())
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(MODEL_DIR, 'best_model.pth'))
            print(f'✓ Best model saved! (Val Acc: {val_acc:.4f})')
    
    # Plot training history
    plot_training_history(train_accs, val_accs, train_losses, val_losses)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("EVALUATING ON TEST SET")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc, test_loss, test_preds, test_labels = eval_model(model, test_loader, device)
    
    print(f'\nTest Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    
    # Get chapter names
    chapter_names = label_encoder.classes_
    
    # For readability, use shortened names in report
    short_names = [f"Ch_{i}" for i in range(len(chapter_names))]
    
    print(classification_report(test_labels, test_preds, target_names=short_names, zero_division=0))
    
    # Save detailed report
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(classification_report(test_labels, test_preds, target_names=chapter_names, zero_division=0))
    
    print(f"\n✓ Detailed report saved to {OUTPUT_DIR}/classification_report.txt")
    
    # Save training summary
    with open(os.path.join(OUTPUT_DIR, 'training_summary.txt'), 'w') as f:
        f.write(f"RoBERTa Chapter Classification - Training Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Max length: {MAX_LENGTH}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Number of epochs: {NUM_EPOCHS}\n")
        f.write(f"FP16 training: {FP16}\n\n")
        f.write(f"Results:\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
    
    print(f"✓ Training summary saved to {OUTPUT_DIR}/training_summary.txt")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"\nModel saved to: {MODEL_DIR}/best_model.pth")
    print(f"Outputs saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
