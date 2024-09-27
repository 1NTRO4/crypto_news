import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
from text_processing_utils import load_data, preprocess_dataframe, join_tokens, load_json, ner_on_dataframe 

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
    anchor = soup.find_all('a', {"class": 'p-flink'})

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


###############################################
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


df_processed = clean_preprocess_data(df_path='data/scraped_news_headline.csv', dictionary_path='AnnotatedDictionary/annotataion_dict.json')