
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

- Download the already fine-tuned models titled `saved_models.zip` from [google drive ](https://drive.google.com/file/d/1rmx9awq2AgnHI0noOBbcNRUdbfJqnMql/view?usp=sharing) and unzip into the [streamlit_app](streamlit_app/) folder

- Setup a virtual enviroment

- Install the packages

```
pip install -r requirements.txt
```

- Launch sentiment analysis streamlit app

```
streamlit run streamlit_app/app.py
```

## Contributors
Terdoo M. Dugeri





