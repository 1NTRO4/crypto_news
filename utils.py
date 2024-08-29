import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import string
import json
from textblob import TextBlob

## Download NLTK data --run only once to download packages
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')


def load_data(filename):
    return pd.read_csv(filename)

# ### Clean and tokenize the data
def preprocess_text(text):
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove symbols (except alphanumeric)
    text = re.sub(r'[^\w]', ' ', text)
    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    ## remove single characters eg 'k'
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
    return tokens

# this function indicates the column from the dataframe that will be preprocessed using the above function
def preprocess_dataframe(df, text_column, new_column):
    df[new_column] = df[text_column].apply(preprocess_text)
    return df

## join the tokenized data as strings
def join_tokens(tokens):  
# tokens contain a list of words
    return ' '.join(tokens)


def load_json(filename):
    with open(filename, 'r') as f:
        entity_dict = json.load(f)
    return entity_dict

def ner_on_text(text, entity_dict):
  
  entities = []
  words = text.split()  # Split text into words

  for word in words:
    word_lower = word.lower()  # Convert word to lowercase for case-insensitive matching
    if word_lower in entity_dict:
      start = text.lower().find(word_lower) # finds the entity starting index 
      end = start + len(word)
      entities.append({
        "text": text[start:end],
        "label": entity_dict[word_lower],
        "start_char": start,
        "end_char": end,
      })

  return entities


#  this function uses the function above to perform entity recognition on our dataframe
def ner_on_dataframe(df, text_column, entity_dict):
  
  #creates a new column in the dataframe to hold the recognised entities.
  df["entities"] = df[text_column].apply(lambda text: ner_on_text(text, entity_dict))
  return df

def classify_sentiment(polarity): 
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def ner_with_sentiment(text, entity_dict):
    
    entities_with_sentiment = []
    words = text.split()  # Split text into words

    for word in words:
        word_lower = word.lower()  # Convert word to lowercase for case-insensitive matching
        if word_lower in entity_dict:
            start = text.lower().find(word_lower)
            end = start + len(word)


            # Extract context around the entity for sentiment analysis
            context = text[max(0, start - 50):min(len(text), end + 50)]
            sentiment_polarity = TextBlob(context).sentiment.polarity
            sentiment = classify_sentiment(sentiment_polarity)

            entities_with_sentiment.append({
                "start_char": start,
                "end_char": end,
                "entity": text[start:end],
                "label": entity_dict[word_lower],
                "sentiment": sentiment,
            })

    return entities_with_sentiment

def ner_with_sentiment_on_dataframe(df, text_column, entity_dict):
    
    df["sentiments"] = df[text_column].apply(lambda text: ner_with_sentiment(text, entity_dict))
    return df

# print('complete')