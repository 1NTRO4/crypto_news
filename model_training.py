from text_processing_utils import load_data, preprocess_dataframe, join_tokens, load_json, ner_on_dataframe 
from datasets_preparations import SentimentDataset, TextDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer, RobertaTokenizer, BertTokenizer, Trainer, TrainingArguments, DistilBertForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification
import json
import nltk
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os


df = load_data('data/scraped_news_headline.csv') #loading scraped crypto currency news headlines
df_preprocessed = preprocess_dataframe(df.copy(), 'headline_news','clean_tokens') #preprocessing 
df_preprocessed['clean_headline'] = df_preprocessed['clean_tokens'].apply(join_tokens) 
df_preprocessed_copy = df_preprocessed

# Using a manually created dictionary for crpto currency entity recognition
entity_dict =load_json('AnnotatedDictionary/annotataion_dict.json')


## apply the above function on the clean_headline column to get recognized entities
df_entities = ner_on_dataframe(df_preprocessed_copy, 'clean_headline', entity_dict)
df_entities.to_csv('data/recognized_entity_dataset.csv', index= False) ## save the dataframe as a csv file to create a checkpoint


# ####   manually labeled(Positive,Negative or Neutral) on the recognized entites

df = load_data('data/entity+sentiment_dataset.csv') # Load datasets

# Extract necessary columns and convert sentiments to list of dictionaries
data = df[['clean_headline', 'sentiments']].copy() 
data['sentiments'] = data['sentiments'].apply(lambda x: json.loads(x.replace("'", "\""))) #changing every single quote to double quote as JSON doesnt recognise single quote

# Initialize the DistilBERT tokenizer 
tokenizer_dict = {
    "distilbert-base-uncased":DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
    "bert-base-uncased":BertTokenizer.from_pretrained('bert-base-uncased'),
    "yiyanghkust/finbert-tone":BertTokenizer.from_pretrained('yiyanghkust/finbert-tone'),
    "roberta-base":RobertaTokenizer.from_pretrained('roberta-base'),
}

# Dictionary of model names and their corresponding classes
models = {
    'distilbert': ('distilbert-base-uncased', DistilBertForSequenceClassification),
    'bert': ('bert-base-uncased', BertForSequenceClassification),
    'roberta': ('roberta-base', RobertaForSequenceClassification),
    'finbert': ('yiyanghkust/finbert-tone', BertForSequenceClassification)
}

# Prepare data
train_headlines, val_headlines, train_sentiments, val_sentiments = train_test_split(data['clean_headline'], data['sentiments'], test_size=0.2, random_state=42)

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
        learning_rate=1e-5  # Set learning rate to 1e-5
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


# Dictionary to hold validation metrics
results = {}


# Iterate over the models dictionary and train each model
for model_name, (model_path, model_class) in models.items():
    results[model_name] = train_and_log_model(model_path, model_class, train_headlines, val_headlines, train_sentiments, val_sentiments, tokenizer_dict)

print(results)

data = []
for model_name, (train_output, eval_metrics, trainer) in results.items():
    data.append([
        model_name,
        eval_metrics['eval_accuracy'],
        eval_metrics['eval_f1'],
        eval_metrics['eval_precision'],
        eval_metrics['eval_recall']
    ])

# Create a DataFrame to hold results 
df_eval = pd.DataFrame(data, columns=['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])

#LSTM 
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

# Define the LSTM model
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


# Training loop
def train_model(model, train_dataset, val_dataset, criterion, optimizer, epochs=5, batch_size=16):
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

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion)
        print(f"Epoch: {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")
    
    save_dir = 'streamlit_app/saved_models'  # Define the save directory here
    model_save_path = os.path.join(save_dir, 'lstm_model.pth')
    torch.save(model.state_dict(), model_save_path)

# Evaluation function
def evaluate_model(model, val_dataset, criterion):
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

# Define the validation DataLoader
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# Train the model uaing train_model function created above 
train_model(model, train_dataset, val_dataset, criterion, optimizer, epochs=1, batch_size=16)

# Evaluate the model and unpack the results
avg_loss, accuracy, precision, recall, f1 = evaluate_model(model, val_loader, criterion)

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

## acc score comparison
plot_comparison(df_eval, 'Accuracy', 'Model Accuracy Comparison', 'Accuracy', 'acc_score_comparison.png')
## f1 score comparison
plot_comparison(df_eval, 'F1 Score', 'Model F1-Score Comparison', 'F1 Score', 'f1_comparison.png')

print('done')
