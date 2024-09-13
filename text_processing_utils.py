import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import string
import json

# Function to check if the NLTK data is already downloaded
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        # nltk.data.find('tokenizers/punkt_tab') # Uncomment if 'punkt_tab' is a custom resource.
        print("NLTK data already available.")
    except LookupError:
        print("Downloading necessary NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        # nltk.download('punkt_tab') # Uncomment if 'punkt_tab' is required
        print("NLTK data downloaded.")

# initializes NLTK functions
download_nltk_data()



def load_data(filename):
    # Load data from a CSV file into a pandas DataFrame
    return pd.read_csv(filename)


# Clean and tokenize the data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation from the text
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove symbols (except alphanumeric)
    text = re.sub(r'[^\w]', ' ', text)
    # Remove numbers from the text
    text = ''.join([i for i in text if not i.isdigit()])
    # Remove single characters (e.g., 'k')
    text = re.sub(r"\b\w\b", "", text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    # Remove stopwords from the list of tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# Apply text preprocessing to a specific column in a DataFrame
def preprocess_dataframe(df, text_column, new_column):
    # Create a new column with tokenized and cleaned text
    df[new_column] = df[text_column].apply(preprocess_text)
    return df


# Join tokenized data back into a single string
def join_tokens(tokens):
    # Convert a list of tokens back into a single string
    return ' '.join(tokens)


def load_json(filename):
    # Load JSON data from a file
    with open(filename, 'r') as f:
        entity_dict = json.load(f)
    return entity_dict


def ner_on_text(text, entity_dict):
    # Perform Named Entity Recognition (NER) on a single text
    entities = []
    words = text.split()  # Split text into words

    for word in words:
        word_lower = word.lower()  # Convert word to lowercase for case-insensitive matching
        if word_lower in entity_dict:
            start = text.lower().find(word_lower)  # Find the starting index of the entity
            end = start + len(word)  # Calculate the ending index of the entity
            entities.append({
                "text": text[start:end],  # Extract the entity text
                "label": entity_dict[word_lower],  # Get the label from the entity dictionary
                "start_char": start,  # Starting character index
                "end_char": end,  # Ending character index
            })

    return entities


# Apply NER to a specific column in a DataFrame
def ner_on_dataframe(df, text_column, entity_dict):
    # Create a new column with recognized entities
    df["entities"] = df[text_column].apply(lambda text: ner_on_text(text, entity_dict))
    return df


