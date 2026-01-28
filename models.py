"""
Model class definitions for inference only (no training code)
"""

import torch.nn as nn
from transformers import BertModel, RobertaModel, DistilBertModel


class BertChapterClassifier(nn.Module):
    """BERT-based Chapter Classifier"""
    
    def __init__(self, num_classes, dropout=0.3):
        super(BertChapterClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class RobertaChapterClassifier(nn.Module):
    """RoBERTa-based Chapter Classifier"""
    
    def __init__(self, num_classes, dropout=0.3):
        super(RobertaChapterClassifier, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class DistilBertChapterClassifier(nn.Module):
    """DistilBERT-based Chapter Classifier"""
    
    def __init__(self, num_classes, dropout=0.3):
        super(DistilBertChapterClassifier, self).__init__()
        
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
