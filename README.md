
This project performs sentiment analysis on the crypocurrency news headline Dataset using advanced deep learning techniques.

### About
In this Project I intend confirming our hypothesis that distilBERT an optimised version of BERT is best suited for sentiment analysis of cryptocurrency based News headline

I constructed a cryptocurrency based dictionary that is made of the top 200 crypto currencies and their abbreviations according to thier market capitalization. 192 crypto exchanges also made the list. This dictionary was then used to identify crypto currency entities from news headlines datasets and their sentiments determined using Textblob. Various BERT models were then trained and performance evaluated using this dataset. The best performing model would then be deployed for sentiments analysis in finance, particulary to crypto currency news headlines.

### Contributors
Terdoo M. Dugeri

### Data

The dataset comprises of 2374 news headlines related to cryptocurrencies: using the beautiful soup library, 2060 news were scraped from cryptopotato.com, 60 news were scraped from Coinjournal.com, 29 news were scraped from cryptotimes.io, 73 news were scraped from newsbtc.com, 152 news were scraped from cnbc.com. Based on the task at hand a flowchart for the methodology is as shown below.
The dataset contains 50,000 reviews from IMDB. There are 25,000 training reviews and 25,000 testing reviews. The sentiment labels are balanced between positive and negative.

The data is loaded and preprocessed by

Removing stopwords
Converting to lowercase
Removing punctuation
Tokenizing
remove special characters, symbols.

## Entity Recognition and Sentiment Analysis
NER was performed using the custom dictionary approach, and the resulting entities were assigned sentiments. The final dataset contains 1235 headlines, with a total of 1773 entities identified and classified as follows: 413 Positive, 423 Negative, and 937 Neutral.
The custom dictionary was populated with data scraped from a cryptocurrency website- coinranking.com, including the top 200 cryptocurrencies by market capitalization and their abbreviations, labeled as 'CryptoCurrency.' Exchange companies were labeled as 'Company.' The resulting key-value pairs were saved in a JSON file and used for entity annotation. The annotated entities were then assigned sentiments (positive, negative, or neutral) using the TextBlob library.


The following models are implemented and evaluated:

BERT
RoBERTa
DistilBERT
FinBERT
LSTM

### Training
Models are trained on 80% of the dataset.

They are trained for 7 epochs with a batch size of 16.
### Directory

1- Annotated Dictionary- in this folder file containing the source used in scraping the web for top cryptocurrencies, exchanges and abbreviations and the annotated dictionary.

a- Source code- top_crypto_scraping_annotation.ipynb

b- Entity Dictionary  annotataion_dict.json

2- data- in this folder the source code for scrapping the news headlines and the extracted headlines.

source Code is -  main scrapping.ipynb

Streamlit_app(Interface) _  contains the trained model ands its configuration






### Evaluation
All models are evaluated on the remaining 20% test set using accuracy and F1 score.
Classification reports are printed showing precision, recall and F1 for each sentiment class.

### Results
The DistilBERT showed better performance compared to the other models accuracy of 85% and F1 of 0.75 

There is still room for improvement by using more increasing the size of the training data, and increasing the list entities used in the dataset, more  diverse source of news data and regularization techniques.

## Interface
The final model was deployed used a streamlit interface.

### Using the Interface

1- download the trained model from the google link and paste in the "distilbert_head_case" folder located in the streamlit_app folder
     https://drive.google.com/file/d/1Rb9Yc4ZUPWs-ewsUU6jcaCz7ruP8br0k/view?usp=sharing

2- Dependencies Setup
You'll need to install the following dependencies for your Streamlit app:
Streamlit: The framework for running the app.
Transformers: The DistilBertForSequenceClassification and DistilBertTokenizer are part of the Hugging Face Transformers library.
Torch: PyTorch is the backend used by the Transformers library.
NLTK: For text preprocessing tasks such as tokenization and stopwords removal.
JSON: For loading the entity dictionary (this is a standard library in Python and does not require installation).


### Running the App

in your terminal

streamlit run app.py
