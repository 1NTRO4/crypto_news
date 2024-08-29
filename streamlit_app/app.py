import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import json
import string
import re
import nltk
from nltk.corpus import stopwords


# Load the trained model and tokenizer
model_path = 'distilbert_class_head'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load the entity dictionary
with open('AnnotatedDictionary/annotataion_dict.json', 'r') as f:
    entity_dict = json.load(f)


# Download necessary NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove symbols (except alphanumeric)
    text = re.sub(r'[^\w]', ' ', text)
    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    # Remove single characters
    text = re.sub(r"\b\w\b", "", text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)

# Function to recognize entities using the local dictionary
def recognize_entities(headline, entity_dict):
    entities = []
    words = headline.split()
    for word in words:
        word_lower = word.lower()
        if word_lower in entity_dict:
            entities.append((word, entity_dict[word_lower]))
    return entities

# Function to make predictions on new headlines
def predict_sentiment_and_entities(model, tokenizer, entity_dict, headline):
    model.eval()  # Set the model to evaluation mode
    results = []

    entities = recognize_entities(headline, entity_dict)
    if not entities:
        return []

    # Process the entire headline with all entities
    headline_results = {
        'headline': headline,
        'entities': []
    }
    for entity, label in entities:
        context = headline  # Using the entire headline as context

        inputs = tokenizer(context, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            sentiment = 'positive' if predicted_class == 2 else 'negative' if predicted_class == 0 else 'neutral'

        headline_results['entities'].append({
            'entity': entity,
            'label': label,
            'sentiment': sentiment
        })

    results.append(headline_results)
    return results

# Streamlit interface
st.title("Entity-based Sentiment Analysis for News Headlines")
st.write("Enter a news headline to predict its sentiment.")

headline = st.text_input("News Headline")

if st.button("Predict Sentiment"):
    if headline:
        cleaned_headline = preprocess_text(headline)
        results = predict_sentiment_and_entities(model, tokenizer, entity_dict, cleaned_headline)
        if results:
            for result in results:
                st.write(f"Headline: **{headline}**")
                for entity_info in result['entities']:
                    st.write(f"Entity: **{entity_info['entity']}**, Label: **{entity_info['label']}**, Sentiment: **{entity_info['sentiment']}**")
        else:
            st.write("No recognized entities in the headline.")
    else:
        st.write("Please enter a news headline.")

import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import json
import string
import re
import nltk
from nltk.corpus import stopwords


# Load the trained model and tokenizer
model_path = 'saved_models/distilbert-base-uncased'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load the entity dictionary
with open('AnnotatedDictionary/annotataion_dict.json', 'r') as f:
    entity_dict = json.load(f)


# Download necessary NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove symbols (except alphanumeric)
    text = re.sub(r'[^\w]', ' ', text)
    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    # Remove single characters
    text = re.sub(r"\b\w\b", "", text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)

# Function to recognize entities using the local dictionary
def recognize_entities(headline, entity_dict):
    entities = []
    words = headline.split()
    for word in words:
        word_lower = word.lower()
        if word_lower in entity_dict:
            entities.append((word, entity_dict[word_lower]))
    return entities

# Function to make predictions on new headlines
def predict_sentiment_and_entities(model, tokenizer, entity_dict, headline):
    model.eval()  # Set the model to evaluation mode
    results = []

    entities = recognize_entities(headline, entity_dict)
    if not entities:
        return []

    # Process the entire headline with all entities
    headline_results = {
        'headline': headline,
        'entities': []
    }
    for entity, label in entities:
        context = headline  # Using the entire headline as context

        inputs = tokenizer(context, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            sentiment = 'positive' if predicted_class == 2 else 'negative' if predicted_class == 0 else 'neutral'

        headline_results['entities'].append({
            'entity': entity,
            'label': label,
            'sentiment': sentiment
        })

    results.append(headline_results)
    return results

# Streamlit interface
st.title("Entity-based Sentiment Analysis for News Headlines")
st.write("Enter a news headline to predict its sentiment.")

headline = st.text_input("News Headline")

if st.button("Predict Sentiment"):
    if headline:
        cleaned_headline = preprocess_text(headline)
        results = predict_sentiment_and_entities(model, tokenizer, entity_dict, cleaned_headline)
        if results:
            for result in results:
                st.write(f"Headline: **{headline}**")
                for entity_info in result['entities']:
                    st.write(f"Entity: **{entity_info['entity']}**, Label: **{entity_info['label']}**, Sentiment: **{entity_info['sentiment']}**")
        else:
            st.write("No recognized entities in the headline.")
    else:
        st.write("Please enter a news headline.")
