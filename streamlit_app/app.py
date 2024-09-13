import streamlit as st  # Streamlit for building the web app interface
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer 
import torch 
import json  
import string  
import re  
import nltk  
from nltk.corpus import stopwords  

# Load the trained model and tokenizer
model_path = 'streamlit_app\saved_models\distilbert_class_head'  # Specify the path to your model directory
model = DistilBertForSequenceClassification.from_pretrained(model_path)  # Load the trained DistilBERT model
tokenizer = DistilBertTokenizer.from_pretrained(model_path)  # Load the tokenizer for the model

# Load the entity dictionary
with open('AnnotatedDictionary/annotataion_dict.json', 'r') as f:
    entity_dict = json.load(f)  # Load JSON file containing entities and their labels

# Download necessary NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    
    text = text.lower() 
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub(r'[^\w]', ' ', text)  
    text = ''.join([i for i in text if not i.isdigit()])  
    text = re.sub(r"\b\w\b", "", text)  
    text = ' '.join(text.split())
    text = re.sub(r'\W', ' ', text) 
    tokens = nltk.word_tokenize(text)  
    stop_words = set(stopwords.words('english'))  
    tokens = [word for word in tokens if word not in stop_words]  

    return ' '.join(tokens)  # Return the cleaned and tokenized text

def recognize_entities(headline, entity_dict):
    
    entities = []  # Initialize an empty list for entities
    words = headline.split()  # Split the headline into words
    for word in words:
        word_lower = word.lower()  # Convert each word to lowercase
        if word_lower in entity_dict:  # Check if the word exists in the entity dictionary
            entities.append((word, entity_dict[word_lower]))  # Append the entity and its label
    return entities  # Return the list of recognized entities

def predict_sentiment_and_entities(model, tokenizer, entity_dict, headline):
   
    model.eval()  # Set model to evaluation mode
    results = []  # Initialize a list for storing results

    entities = recognize_entities(headline, entity_dict)  # Recognize entities in the headline
    if not entities:  # If no entities found, return an empty list
        return []

    # Dictionary to hold the results for the current headline
    headline_results = {
        'headline': headline,  # Original headline
        'entities': []  # List to hold entities and their sentiment
    }

    for entity, label in entities:
        context = headline  # Use the entire headline as context for sentiment prediction

        # Tokenize the context using the model's tokenizer
        inputs = tokenizer(context, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']  # Extract input IDs
        attention_mask = inputs['attention_mask']  # Extract attention mask

        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(input_ids, attention_mask=attention_mask)  # Get model outputs
            logits = outputs.logits  # Extract logits from model output
            predicted_class = torch.argmax(logits, dim=1).item()  # Get the predicted class index

            # Map the predicted class index to a sentiment label
            sentiment = 'positive' if predicted_class == 2 else 'negative' if predicted_class == 0 else 'neutral'

        # Append the entity, its label, and predicted sentiment to the headline results
        headline_results['entities'].append({
            'entity': entity,
            'label': label,
            'sentiment': sentiment
        })

    results.append(headline_results)  # Append the headline results to the overall results list
    return results  # Return the results

# Streamlit interface for user interaction

st.title("Entity-based Sentiment Analysis for News Headlines")  # Set the title of the web app
st.write("Enter a news headline to predict its sentiment.")  # Display instructions to the user


headline = st.text_input("News Headline")

# Button to trigger sentiment prediction
if st.button("Predict Sentiment"):
    if headline:  # Check if a headline has been entered
        cleaned_headline = preprocess_text(headline)  # Preprocess the input headline
        results = predict_sentiment_and_entities(model, tokenizer, entity_dict, cleaned_headline)  # Get prediction results
        if results:  # Check if there are any results to display
            for result in results:  # Iterate through each result
                st.write(f"Headline: **{headline}**")  # Display the original headline
                for entity_info in result['entities']:  # Display each entity with its label and sentiment
                    st.write(f"Entity: **{entity_info['entity']}**, Label: **{entity_info['label']}**, Sentiment: **{entity_info['sentiment']}**")
        else:
            st.write("No recognized entities in the headline.")  # Message if no entities are recognized
    else:
        st.write("Please enter a news headline.")  # Message if no input was provided


