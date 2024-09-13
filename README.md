
# Sentiment Analysis of Cryptocurrency News Headline using a Fine-Tuned Distilbert Model.

    
## About
The project intends to do sentiments analysis for crypto-currency news sentiment analysis by finetune DistilBERT and using it for said purpose. Its performanc would also be compared with fintuned versions of FinBERT, BERT, Roberta. LSTM model is also trained to see how it compares to the pre-trained models.

After scraping of the news headlines from select crypto-currency related websites. the news headline was preprocessed . A manually annotated dictionary was then used to identify the crypto currencies present before the sentiments in the dataset was manually labeled.

 DistilBERT performed best here and was then used for sentiment analysis using streamlit web interface.

## Scraping Crypto-currency News Headlines from select sites

 Please follow the steps below in your terminal : 

1. Clone the repo

```
git clone https://github.com/1NTRO4/crypto_news
cd crypto_news
```

2. Run the scrapping_crypto_news.py script

```
python scrapping_crypto_news.py
```


## Fine tuning the BERT-base models and training LSTM model

 I recommend using a gpu or google colab for model finetuning and training.

 Please follow the steps below: 

1. Clone the repo

```
!git clone https://github.com/1NTRO4/crypto_news.git
%cd crypto_news
```

2. Install the necessary packages

```
!pip install -r requirements.txt
```

3.  Run the training_run.py script

```
!python training_run.py
```

## Making Predictions with the Fine-Tuned Model


A streamlit app interface is used for sentiments prediction.

To run this model take note of the following steps

- Clone the repo

```
git clone https://github.com/1NTRO4/crypto_news
cd crypto_news
```

- Create a folder in  `streamlit_app` folder named `saved_models`

- Download the already fine-tuned model titled `distilbert_class_head` from [google drive ](https://drive.google.com/drive/folders/1uUW5ftdOXXPz4_g41ijPNkecDOi6GSel?usp=sharing) link into [saved_models](streamlit_app/saved_models/)  folder

- Setup a virtual enviroment

- Install the packages

```
pip install -r requirements.txt
```

- Launch sentiment analysis streamlit app

```
streamlit run streamlit_app/app.py
```




## Data

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

## Training
Models are trained on 80% of the dataset.

They are trained for 7 epochs with a batch size of 16.

the model_training.py file is where the models are trained and saved.





## Directory
1- AnnotatedDictionary-  folder contains the dictionary  "  annotataion_dict.json " used in the crypto currency labelling task.  The source used in scraping the web for top cryptocurrencies, exchanges and abbreviations  is the "scrapping_crypto_news.py"

2- data-  Extracted and processed data are saved here

3. Streamlit_app(Interface) _  the script that opens the web interface for app usage "app.py"
          
          
           Dependencies Setup- refer to the "requirements.txt" file for dependencies needed to successfully run the app
 
 streamlit run app.py to run the app after all dependencies have been installed.

## Evaluation

All models are evaluated on the remaining 20% test set using accuracy and F1 score.
Classification reports are printed showing precision, recall and F1 for each sentiment class.


## Results
The DistilBERT showed better performance compared to the other models accuracy of 85% and F1 of 0.75 
There is still room for improvement by using more increasing the size of the training data, and increasing the list entities used in the dataset, more  diverse source of news data and regularization techniques.



## Contributors
Terdoo M. Dugeri





