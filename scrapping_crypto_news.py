import requests
import re
import json
import string
import csv
import time

import pandas as pd

import nltk
from nltk.corpus import stopwords

from bs4 import BeautifulSoup
from textblob import TextBlob

#########################################################################################################
# Build NLTK Corpus 
#########################################################################################################

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



#########################################################################################################
# Handling Preprocessing 
#########################################################################################################

def load_data(filename):
    # Load data from a CSV file into a pandas DataFrame
    return pd.read_csv(filename)

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

def join_tokens(tokens):
    # Convert a list of tokens back into a single string
    return ' '.join(tokens)

# Apply NER to a specific column in a DataFrame
def ner_on_dataframe(df, text_column, entity_dict):
    # Create a new column with recognized entities
    df["entities"] = df[text_column].apply(lambda text: ner_on_text(text, entity_dict))
    return df

#########################################################################################################
# Scraping Crypto-News headlines
#########################################################################################################

# Function to scrape Crypto Potato headlines with pagination
def scrape_cryptopotato(base_url, pages):
    all_headlines = []

    for page in range(1, pages + 1):
        url = f"{base_url}/page/{page}/"
        request = requests.get(url)
        soup = BeautifulSoup(request.text, 'html.parser')

        headlines = soup.find_all('h3', class_='entry-title')
        for headline in headlines:
            news = headline.get_text().strip()
            if news:
                all_headlines.append(news)

        print(f"Scraped page {page}/{pages}")

    return all_headlines

# Function to scrape CoinJournal headlines
def scrape_coinjournal(address):
    request = requests.get(address)
    soup = BeautifulSoup(request.text, 'html.parser')
    anchor = soup.find_all('h2')

    headline_news = [headline.get_text().strip() for headline in anchor if headline.get_text().strip()]
    return headline_news

# Function to scrape Crypto Times headlines
def scrape_cryptotimes(address):
    request = requests.get(address)
    soup = BeautifulSoup(request.text, 'html.parser')
    anchor = soup.find_all('a', {"class": 'p-flink'})# the class fr

    headline_news = [headline['title'] for headline in anchor if 'title' in headline.attrs]
    return headline_news

# Function to scrape NewsBTC headlines
def scrape_newsbtc(address):
    request = requests.get(address)
    soup = BeautifulSoup(request.text, 'html.parser')
    anchor = soup.find_all("h4", {"class": 'block-article__title'})

    headline_news = [headline.get_text().strip() for headline in anchor if headline.get_text().strip()]
    return headline_news

def get_headline(url):
    match = re.search(r'\/([^\/]+?)(?:\.html|$)', url)
    if match:
        headline = match.group(1).replace('-', ' ')
        headline = re.sub(r'[^\w\s]', '', headline)  
        return headline.lower()
    return None

# Function to process the file and return a list of headlines

def scrape_cnbc_site():
    base_url = "https://www.cnbc.com/cryptoworld/"

    # Open a CSV file to write the URLs
    file = open('data/cnbc_headlines.csv', "w", newline='')
    writer = csv.writer(file)

    # Write the header to the CSV file
    writer.writerow(["URL"])

    page_number = 1
    max_pages = 10  # Set this to the maximum number of pages you want to scrape

    while True:
        page_to_scrape = requests.get(base_url + f"?page={page_number}")
        soup = BeautifulSoup(page_to_scrape.text, "html.parser")

        # Find all <a> tags with the class 'RiverCard-mediaContainer'
        links = soup.find_all("a", class_="RiverCard-mediaContainer")

        if not links or page_number > max_pages:
            break

        # Extract and write the 'href' attribute from each <a> tag to the CSV file
        for link in links:
            url = link.get('href')
            if url:
                writer.writerow([url])
                print(url)

        page_number += 1
        time.sleep(1)  # Sleep to prevent overwhelming the server (politeness policy)

    # Close the CSV file
    file.close()

def scrape_cnbc(filename):
    headlines = []
    # Opening and reading the file
    with open(filename, 'r') as myfile:
        cnbc_data = myfile.readlines()

    # Looping through the URLs and extracting headlines
    for url in cnbc_data:
        url = url.strip()
        if url: 
            headline = get_headline(url)
            if headline:
                headlines.append(headline)
    
    return headlines


# Main code to scrape all sites and combine the headlines into one CSV file
if __name__ == "__main__":
    # Scrape Crypto Potato
    crypto_potato_headlines = scrape_cryptopotato('https://cryptopotato.com', 210)
    
    # Scrape CoinJournal
    coinjournal_headlines = scrape_coinjournal('https://coinjournal.net/')
    
    # Scrape Crypto Times
    cryptotimes_headlines = scrape_cryptotimes('https://www.cryptotimes.io/')
    
    # Scrape NewsBTC
    newsbtc_headlines = scrape_newsbtc('https://www.newsbtc.com/')

    cnbc_headlines = scrape_cnbc('data/cnbc_headlines.csv')

    # Combine all headlines into one DataFrame
    all_headlines = {
        'headline_news': crypto_potato_headlines + coinjournal_headlines + cryptotimes_headlines + newsbtc_headlines + cnbc_headlines
    }
    
    headlines_df = pd.DataFrame(all_headlines)

    # Save to a CSV file
    headlines_df.to_csv('data/scraped_news_headline.csv', index=False)

    print("All headlines have been scraped and saved to 'data/scraped_news_headline.csv'.")


#########################################################################################################
# Build Annotation Dictionary
#########################################################################################################

def get_all_exchanges():
    base_url = 'https://coinranking.com/exchanges?page='    
    num_pages = 4  # Number of pages to scrape
    all_exchanges = []

    for page in range(1, num_pages + 1):
        url = f'{base_url}{page}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table containing the exchanges
        table = soup.find('table')

        if table:
            # Iterate over the rows in the table
            for row in table.find_all('tr'):
                # Get the name of the exchange company
                name_cell = row.find('a', {'class': 'profile__link'})
                if name_cell:
                    name = name_cell.get_text(strip=True)
                    all_exchanges.append(name.lower())
    
    return all_exchanges


def get_top_cryptocurrencies():
    base_url = 'https://coinranking.com/?page='
    names = []
    abbreviations = []

    # Iterate through the first 4 pages
    for page in range(1, 5):
        url = f'{base_url}{page}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all rows in the table
        rows = soup.find_all('tr')
        
        for row in rows:
            # Find the name and abbreviation
            name_tag = row.find('a', {'class': 'profile__link'})
            abbrev_tag = row.find('span', {'class': 'profile__subtitle-name'})
            
            if name_tag and abbrev_tag:
                name = name_tag.get_text(strip=True)
                abbreviation = abbrev_tag.get_text(strip=True)
                names.append(name)
                abbreviations.append(abbreviation)
                
    return names, abbreviations


annotations = {
    "bitcoin": "CRYPTOCURRENCY",
    "marathon digital": "COMPANY",
    "kaspa": "CRYPTOCURRENCY",
    "cryptocurrencies": "CRYPTOCURRENCY",
    "bitbot": "CRYPTOCURRENCY",
    "btc": "CRYPTOCURRENCY",
    "us": "COUNTRY",
    "government": "ENTITY",
    "coinbase": "COMPANY",
    "germany": "COUNTRY",
    "bka": "ENTITY",
    "crypto": "CRYPTOCURRENCY",
    "fear and greed index": "INDEX",

}

exchanges = get_all_exchanges()

# Example of additional cryptocurrencies and companies to add
top_cryptocurrencies, top_cryptocurrencies_symb = get_top_cryptocurrencies()


# Add top cryptocurrencies to annotations
for crypto in top_cryptocurrencies:
    annotations[crypto] = "CRYPTOCURRENCY"

for i in top_cryptocurrencies_symb:
    annotations[i] = "CRYPTOCURRENCY"

# Add top companies to annotations
for company in exchanges:
    annotations[company] = "COMPANY"

with open("AnnotatedDictionary/annotataion_dict.json", "w") as outfile:
    json.dump(annotations, outfile)

print("The annotated dictionary has been saved as :'annotataion_dict.json'")

#########################################################################################################
#  Data Preprocessing 
#########################################################################################################

#clean and preprocess news headlines, recognize entities in the text using a manually created dictionary, and save the processed data.
def clean_preprocess_data(df_path, dictionary_path):
    df = load_data(df_path) #loading scraped crypto currency news headlines
    df_preprocessed = preprocess_dataframe(df.copy(), 'headline_news','clean_tokens') #preprocessing 
    df_preprocessed['clean_headline'] = df_preprocessed['clean_tokens'].apply(join_tokens) 
    df_preprocessed_copy = df_preprocessed

    # Using a manually created dictionary for crpto currency entity recognition
    entity_dict =load_json(dictionary_path)
    
    ## apply the above function on the clean_headline column to get recognized entities
    df_entities = ner_on_dataframe(df_preprocessed_copy, 'clean_headline', entity_dict)
    df_entities.to_csv('data/recognized_entity_dataset.csv', index= False) ## save the dataframe as a csv file to create a checkpoint
    return df_entities

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

df_processed = clean_preprocess_data(df_path='data/scraped_news_headline.csv', dictionary_path='AnnotatedDictionary/annotataion_dict.json')

## calling the funtion on clean_headline column
df_with_sentiments = ner_with_sentiment_on_dataframe(df_processed, "clean_headline",annotations)

df_with_sentiments.to_csv('entity+sentiment_dataset.csv', index = False) ## save the dataframe as a csv file