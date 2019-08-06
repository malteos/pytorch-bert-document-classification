import torch
from pytorch_pretrained_bert import BertModel
from torch import nn
from torch.nn import functional as F


class BertMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim=768, dropout=0.1):
        super().__init__()

        self.config = {
            'bert_model_path': bert_model_path,
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
        }

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, labels_count)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens, masks):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)

        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)

        return proba


class ExtraBertMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim=768, mlp_dim=100, extras_dim=6, dropout=0.1):
        super().__init__()

        self.config = {
            'bert_model_path': bert_model_path,
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'mlp_dim': mlp_dim,
            'extras_dim': extras_dim,
            'dropout': dropout,
        }

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + extras_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            # nn.ReLU(),
            # nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),            
            nn.Linear(mlp_dim, labels_count)
        )
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, tokens, masks, extras):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        
        concat_output = torch.cat((dropout_output, extras), dim=1)
        mlp_output = self.mlp(concat_output)
        # proba = self.sigmoid(mlp_output)
        proba = self.softmax(mlp_output)

        return proba


class LinearMultiClassifier(nn.Module):
    def __init__(self, labels_count, extras_dim=6, dropout=0.1):
        super().__init__()

        self.config = {
            'labels_count': labels_count,
            'extras_dim': extras_dim,
        }
        self.linear = nn.Linear(extras_dim, labels_count)
        self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, extras):
        lin_output = self.linear(extras)
        # proba = self.sigmoid(mlp_output)
        proba = self.softmax(lin_output)

        return proba


class ExtraMultiClassifier(nn.Module):
    def __init__(self, labels_count, mlp_dim=100, extras_dim=6, dropout=0.1):
        super().__init__()

        self.config = {
            'labels_count': labels_count,
            'mlp_dim': mlp_dim,
            'extras_dim': extras_dim,
            'dropout': dropout,
        }

        self.mlp = nn.Sequential(
            nn.Linear(extras_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, labels_count)
        )
        self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, extras):

        mlp_output = self.mlp(extras)
        # proba = self.sigmoid(mlp_output)
        proba = self.softmax(mlp_output)
        
        return proba

