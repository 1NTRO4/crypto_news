
# dataset_preparation.py

import torch
from torch.utils.data import Dataset

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
