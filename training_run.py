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
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import DistilBertForSequenceClassification, DistilBertForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification
from transformers import  DistilBertTokenizer, RobertaTokenizer, BertTokenizer
from transformers import Trainer, TrainingArguments

#########################################################################################################
#  Dataset classes Preparation
#########################################################################################################

class SentimentDataset(Dataset):
    def __init__(self, headlines, sentiments, tokenizer):
        self.headlines = headlines  # List of text data (headlines)
        self.sentiments = sentiments  # List of sentiment data corresponding to the headlines
        self.tokenizer = tokenizer  # Tokenizer function from a library like HuggingFace Transformers

    def __len__(self):
        return len(self.headlines)  # Return the number of items in the dataset

    def __getitem__(self, idx):
        # Tokenize the headline at index `idx` using the provided tokenizer
        tokens = self.tokenizer(self.headlines[idx], padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        
        # Extract input IDs and attention mask from the tokenized output
        input_ids = tokens['input_ids'].squeeze()  # Remove extra dimension
        attention_mask = tokens['attention_mask'].squeeze()  # Remove extra dimension
        
        # Determine the sentiment label
        label = self._get_sentiment_label(self.sentiments[idx])
        
        # Return a dictionary with input IDs, attention mask, and label
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label, dtype=torch.long)}

    def _get_sentiment_label(self, sentiments):
        # Determine sentiment label: 0 for Negative, 1 for Neutral, 2 for Positive
        if any(sent['sentiment'] == 'positive' for sent in sentiments):
            return 2  # Positive
        elif any(sent['sentiment'] == 'negative' for sent in sentiments):
            return 0  # Negative
        else:
            return 1  # Neutral

class TextDataset(Dataset):
    def __init__(self, headlines, sentiments, vocab, max_length=128):
        self.headlines = headlines  # List of text data (headlines)
        self.sentiments = sentiments  # List of sentiment data corresponding to the headlines
        self.vocab = vocab  # Vocabulary mapping of words to indices
        self.max_length = max_length  # Maximum length of the tokenized sequence

    def __len__(self):
        return len(self.headlines)  # Return the number of items in the dataset

    def __getitem__(self, idx):
        # Convert the headline at index `idx` to a sequence of indices based on the vocabulary
        sequence = self.text_to_sequence(self.headlines[idx])
    
        # Pad or truncate the sequence to the maximum length
        sequence = sequence[:self.max_length] + [0] * (self.max_length - len(sequence))
        
        # Determine the sentiment label
        label = self._get_sentiment_label(self.sentiments[idx])
        
        # Return a dictionary with input IDs (sequence) and label
        return {'input_ids': torch.tensor(sequence, dtype=torch.long), 'labels': torch.tensor(label, dtype=torch.long)}

    def text_to_sequence(self, text):
        # Convert text to sequence of indices based on the vocabulary
        return [self.vocab.get(word, 0) for word in text.split()]

    def _get_sentiment_label(self, sentiments):
        # Determine sentiment label: 0 for Negative, 1 for Neutral, 2 for Positive
        if any(sent['sentiment'] == 'positive' for sent in sentiments):
            return 2  # Positive
        elif any(sent['sentiment'] == 'negative' for sent in sentiments):
            return 0  # Negative
        else:
            return 1  # Neutral
        
def get_lstm_dataset(train_headlines, train_sentiments, test_headlines, test_sentiments):
   
    # Build vocabulary
    vocab = set()
    for headline in data['clean_headline']:
        vocab.update(headline.split())
    vocab = {word: idx for idx, word in enumerate(vocab)}

    # Prepare data
    train_dataset = TextDataset(train_headlines.tolist(), train_sentiments.tolist(), vocab)
    test_dataset = TextDataset(test_headlines.tolist(), test_sentiments.tolist(), vocab)
    return train_dataset, test_dataset, vocab
        

#########################################################################################################
#  Functions for Model Finetuning/Training and Evaluation
#########################################################################################################

def compute_metrics(p): ###Models comparison
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='macro')
    acc = accuracy_score(labels, pred)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# a function to train and log the model
def train_and_log_model(model_name, model_class, train_headlines, val_headlines, train_sentiments, val_sentiments, tokenizer_dict, num_labels=3):
    
    # load suitable tokenizer
    tokenizer = tokenizer_dict[model_name]
    
    
    # creating the dataset 
    train_dataset = SentimentDataset(train_headlines.tolist(), train_sentiments.tolist(), tokenizer)
    val_dataset = SentimentDataset(val_headlines.tolist(), val_sentiments.tolist(), tokenizer)

    # Load the model
    model = model_class.from_pretrained(model_name, num_labels=num_labels)

    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results_and_plots',
        num_train_epochs=7,  
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        report_to="none",  
        learning_rate=1e-5  
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    train_result = trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()

    # Save the model
    save_dir = 'streamlit_app/saved_models'  # Define the save directory here
    model_save_path = os.path.join(save_dir, model_name)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    return train_result, eval_result, trainer

# Evaluation function
def evaluate_lstm_model(model, val_dataset, criterion):
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    model.eval()  # Ensure model is in evaluation mode
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            predictions = model(input_ids)
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            pred_classes = predictions.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_classes.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

# Training loop
def train_lstm_model(model, train_dataset, val_dataset, criterion, optimizer, epochs=5, batch_size=16):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()  # Ensure model is in training mode
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            predictions = model(input_ids)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_lstm_model(model, val_loader, criterion)
        print(f"Epoch: {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")
    
    save_dir = 'streamlit_app/saved_models'  # Define the save directory here
    model_save_path = os.path.join(save_dir, 'lstm_model.pth')
    torch.save(model.state_dict(), model_save_path)

#########################################################################################################
#  Load Data and Fine-tune Models
#########################################################################################################

print("Loading the preprocessed dataset from the folder titled data...")
df= pd.read_csv("data/entity+sentiment_dataset.csv")
data = df[['clean_headline', 'sentiments']].copy() 
data['sentiments'] = data['sentiments'].apply(lambda x: json.loads(x.replace("'", "\"")))

print(" Splitting the data into test and training dataset")
# Split dataset into train and test  dataset
train_headlines, test_headlines, train_sentiments, test_sentiments = train_test_split(
    data['clean_headline'], data['sentiments'], test_size=0.2, random_state=42
)

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

#########################################################################################################
#  LSTM Architecture
#########################################################################################################

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
train_lstm_model(model, lstm_train_dataset, lstm_test_dataset, criterion, optimizer, epochs=1, batch_size=16)

# Evaluate the model and unpack the results
avg_loss, accuracy, precision, recall, f1 = evaluate_lstm_model(model, lstm_test_dataset, criterion)

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

#########################################################################################################
#  Evaluation Results
#########################################################################################################

## plot the results from dataframe and save the output figure 
def plot_comparison(df, metric, title, ylabel, filename):
    filename = "results_and_plots/"+ filename
    
    plt.figure(figsize=(8, 4))  
    plt.bar(df['Model'], df[metric])  # Create the bar plot
    plt.title(title)  # Set the title of the plot
    plt.xlabel('Model')  # Set the label for the x-axis
    plt.ylabel(ylabel)  # Set the label for the y-axis
    plt.savefig(filename)
    print(f"Plot of {title} saved to 'results_and_plots/' ")  
    plt.close() 

print(f"Evaluation metrics added to dataframe as saved to 'results_and_plots/'as {file_path}")

## acc score comparison
plot_comparison(df_eval, 'Accuracy', 'Model Accuracy Comparison', 'Accuracy', 'acc_score_comparison.png')
## f1 score comparison
plot_comparison(df_eval, 'F1 Score', 'Model F1-Score Comparison', 'F1 Score', 'f1_comparison.png')

print(" Finetuning  models completed")