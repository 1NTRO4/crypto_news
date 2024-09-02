
This project performs sentiment analysis on the crypocurrency news headline Dataset using advanced deep learning techniques.

### About
In this Project I intend confirming my hypothesis that distilBERT an optimised version of BERT is best suited for sentiment analysis in Finance with a focus in cryptocurrency based News headline

I constructed a cryptocurrency based dictionary that is made of the top 200 crypto currencies and their abbreviations according to thier market capitalization. 192 crypto exchanges also made the list. This dictionary was then used to identify crypto currency entities from news headlines datasets and their sentiments determined using. Textblob. DistilBERT,FinBERT, RoBERTa and BERT models were then fine tuned using this dataset with their performance evaluated.  DistilBERT performed best here and was then used for sentiment analysis using streamlit web interface.

### Contributors
Terdoo M. Dugeri

### Data

The dataset comprises of 2374 news headlines related to cryptocurrencies: using the beautiful soup library, 2060 news were scraped from cryptopotato.com, 60 news were scraped from Coinjournal.com, 29 news were scraped from cryptotimes.io, 73 news were scraped from newsbtc.com, 152 news were scraped from cnbc.com. Based on the task at hand a flowchart for the methodology is as shown below.
The dataset contains 50,000 reviews from IMDB. There are 25,000 training reviews and 25,000 testing reviews. The sentiment labels are balanced between positive and negative.  the code for this activity is contained in the script - scrapping_crypto_news.py


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

the model_training.py file is where the models are trained and saved.

### Directory
1- AnnotatedDictionary-  folder contains the dictionary  "  annotataion_dict.json " used in the crypto currency labelling task.  The source used in scraping the web for top cryptocurrencies, exchanges and abbreviations  is the "scrapping_crypto_news.py"

2- data-  Extracted and processed data are saved here

3. Streamlit_app(Interface) _  the script that opens the web interface for app usage "app.py"
          
          
           Dependencies Setup- refer to the "requirements.txt" file for dependencies needed to successfully run the app
 
 streamlit run app.py to run the app after all dependencies have been installed.

### Evaluation

All models are evaluated on the remaining 20% test set using accuracy and F1 score.
Classification reports are printed showing precision, recall and F1 for each sentiment class.


### Results
The DistilBERT showed better performance compared to the other models accuracy of 85% and F1 of 0.75 
There is still room for improvement by using more increasing the size of the training data, and increasing the list entities used in the dataset, more  diverse source of news data and regularization techniques.





