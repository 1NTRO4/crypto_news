
# Sentiment Analysis of Cryptocurrency News Headline using a Fine-Tuned Distilbert Model.
  
This project focuses on sentiment analysis of cryptocurrency news by fine-tuning various transformer models, with DistilBERT ultimately being used for the final analysis. The performance of fine-tuned models, including DistilBERT, FinBERT, BERT, and RoBERTa, is compared to an LSTM model, which is also trained for the same task.

## Workflow
Data Collection: News headlines were scraped from selected cryptocurrency-related websites.

Data Preprocessing: Headlines were preprocessed, and a manually annotated dictionary was used to identify the specific cryptocurrencies mentioned in the text. Sentiments for the dataset were manually labeled based on the headlines.

Model Training:  The labeled dataset was split into training and test sets. Selected models (DistilBERT, FinBERT, BERT, RoBERTa) were fine-tuned on the training set. An LSTM model was also trained for comparison.

Model Evaluation:
The performance of each model was evaluated on the test set.
DistilBERT outperformed the other models and was selected for further analysis.

Deployment: The fine-tuned models are saved in the Streamlit_app/saved_models directory. DistilBERT is used in a Streamlit web interface for real-time sentiment analysis of cryptocurrency news.

Key Features
Fine-tuning of state-of-the-art transformer models (DistilBERT, FinBERT, BERT, RoBERTa).
LSTM as a baseline model.
Manual sentiment labeling and cryptocurrency identification in the dataset.
Web interface for sentiment analysis using Streamlit.


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





