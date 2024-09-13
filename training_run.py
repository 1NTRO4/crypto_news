#!/usr/bin/env python3
# imports
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification
from transformers import  DistilBertTokenizer, RobertaTokenizer, BertTokenizer
 
from model_training import train_and_log_model, get_lstm_dataset, evaluate_model, train_model, plot_comparison

# prepare dataset
print("Loading the preprocessed dataset from the folder titled data...")
df= pd.read_csv("data/entity+sentiment_dataset.csv")
data = df[['clean_headline', 'sentiments']].copy() 
data['sentiments'] = data['sentiments'].apply(lambda x: json.loads(x.replace("'", "\"")))

print(" Splitting the data into test and training dataset")
# Split dataset into train and test  dataset
train_headlines, test_headlines, train_sentiments, test_sentiments = train_test_split(
    data['clean_headline'], data['sentiments'], test_size=0.2, random_state=42
)


# model finetuning

# Dictionary of model names and their corresponding classes
models = {
    'distilbert': ('distilbert-base-uncased', DistilBertForSequenceClassification),
    'bert': ('bert-base-uncased', BertForSequenceClassification),
    'roberta': ('roberta-base', RobertaForSequenceClassification),
    'finbert': ('yiyanghkust/finbert-tone', BertForSequenceClassification)
}

# Initialize the DistilBERT tokenizer 
tokenizer_dict = {
    "distilbert-base-uncased":DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
    "bert-base-uncased":BertTokenizer.from_pretrained('bert-base-uncased'),
    "yiyanghkust/finbert-tone":BertTokenizer.from_pretrained('yiyanghkust/finbert-tone'),
    "roberta-base":RobertaTokenizer.from_pretrained('roberta-base'),
}

results = {}
data = []

# Iterate over the models dictionary and train each model
for model_name, (model_path, model_class) in models.items():
    print(f"Finetuning {model_name} model...")
    results[model_name] = train_and_log_model(model_path, model_class, train_headlines, test_headlines, train_sentiments, test_sentiments, tokenizer_dict)
    _, eval_metrics, __= results[model_name]
    print(f"Completed fine-tuning of {model_name}. Evaluation metrics: accuracy = {eval_metrics['eval_accuracy']}, f1 score = {eval_metrics['eval_f1']}, precision = {eval_metrics['eval_precision']}, recall = {eval_metrics['eval_recall']} ")
   
    data.append([
        model_name,
        eval_metrics['eval_accuracy'],
        eval_metrics['eval_f1'],
        eval_metrics['eval_precision'],
        eval_metrics['eval_recall']
    ])

# Create a DataFrame to hold results 
df_eval = pd.DataFrame(data, columns=['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])


print("Training the LSTM model...")
lstm_train_dataset, lstm_test_dataset, vocab = get_lstm_dataset(train_headlines= train_headlines, test_headlines= test_headlines, train_sentiments= train_sentiments, test_sentiments= test_sentiments)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_out, _ = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((lstm_out[:, -1, :self.lstm.hidden_size], lstm_out[:, 0, self.lstm.hidden_size:]), dim=1))
        else:
            hidden = self.dropout(lstm_out[:, -1, :])
        output = self.fc(hidden)
        return output

#- Hyperparameters
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 3
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

# Instantiate the model
model = LSTMModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Train LSTM model
train_model(model, lstm_train_dataset, lstm_test_dataset, criterion, optimizer, epochs=1, batch_size=16)

# Evaluate the model and unpack the results
avg_loss, accuracy, precision, recall, f1 = evaluate_model(model, lstm_test_dataset, criterion)

# Add LSTM result to df_eval
new_data = {
    'Model': 'LSTM',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

# Append the new row to the existing DataFrame
new_row = pd.DataFrame([new_data])
df_eval = pd.concat([df_eval, new_row], ignore_index=True)
file_path = "results_and_plots/" + "eval_metrics_from_finetuning_training.csv"
df_eval.to_csv(file_path, index=False)
print(f"Evaluation metrics added to dataframe as saved to 'results_and_plots/'as {file_path}")

## acc score comparison
plot_comparison(df_eval, 'Accuracy', 'Model Accuracy Comparison', 'Accuracy', 'acc_score_comparison.png')
## f1 score comparison
plot_comparison(df_eval, 'F1 Score', 'Model F1-Score Comparison', 'F1 Score', 'f1_comparison.png')

print(" Finetuning  models completed")

