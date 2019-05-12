# StockPredict

CMPT 733 Final Project: 

Our project a stock movement prediction data science pipeline and platform, including binary(up/down) classification and regression (price prediction). For details please check out our report, video (https://www.youtube.com/watch?v=b5wkez0ENVk&feature=youtu.be) and website (http://nml-cloud-231.cs.sfu.ca:8088/).

To run our data science pipeline, simply follow the instructions in each stage as following. Here are the list of dependencies for running the python code: AlphaVantage, Quandl, tweepy, gensim, spacy, pyLDAvis, nltk, seaborn, pandas, numpy, PIL, wordcloud, tqdm, keras, tensorflow, sklearn.

Before running our code, please download files from Google Drive:

1. Code/3ModelDevelopment/Doc2Vec/doc2vec_model.docvecs.vectors_docs.npy -- Doc2Vec model

https://drive.google.com/open?id=1XhfAJkx6Ign9AMc5eHe3pp_cg5W5YSmg

2. Data/ProcessedNews.pkl -- Preprocessed News Data

https://drive.google.com/open?id=1hMsiEZoP5ITTy7B-5fr_NH0W4Vb46YQu

Details for each folder and files:

## Data Collection & Preprocess

Path: Code/1DataCollection&Preprocess

AllSymbolScraper.py -- Scrape all nasdaq company name and stock symbol.

ExtractNYTNews.py -- Parse HTML files and store to json.

Text_Preprocessing.ipynb -- Prepross Json files and store to pickle file.

stockDataGetter.ipynb -- Get company stock price and nasdaq index.

Twitter_api.py -- Extracting twitter data from api (reference : https://github.com/tweepy/tweepy)

Twitter_parse.py & Twitter_reformat.py -- Twitter data cleaning and preprocessing

Guardian_scrape.ipynb -- Scrape the Guardian news.

news_api_extract.ipynb -- Extract HTML files from news_api.

news_extract_Guardian.ipynb -- Extract HTML files from guardian.

NYT_scrape.ipynb -- Scrape news from New York Times.

## Data Analysis

Path: Code/2DataAnalysis

TopicModeling -- Train LDA model and visualize result.

Twitter-Wordcloud.py - Twitter data visualation.

News_EDA.ipynb -- EDA for news dataset.

tsNe.ipynb -- t-SNE for word embedding.

## Model Development

Path: Code/3ModelDevelopment

Doc2Vec -- Doc2Vec model development and LSTM model for binary prediction.

base_model_nasdaq.ipynb -- Baseline model for logistic regression and linear regression.

deploy_model.py -- Integrate model to web application.

Glove.ipynb -- Train glove model and cnn+rnn model.

Linear SVM_CountNgram.ipynb -- SVM model for binary classification.

lstm_regression.ipynb -- Train LSTM model for regression predition.

model/ -- glove model, deep learning models.

Naive Bayes model_CountNgram.ipynb -- Naive Bayes for binary classification.

Random Forest model_CountNgram.ipynb -- Random Forest for binary classification.

## Web Application

Path: WebServer/

Sprintboot Web Server code. Java code and all static resources included. Maven based project.

WebServer/src/main/java/com/sfu/an3di/  -- Server Java code

WebServer/src/main/resources/template/  -- html pages

WebServer/src/main/resources/static/    -- js, css, images, etc

## Our Data

Path: Data/

AllSymbolScrapper -- Stock symbols html file.

HTMLNews -- New Yort Times and part of the Gardian HTML news.

JsonNews -- Extraced json from HTML.

StockPrice -- CSV file for stock price data.

NewsAPI -- Data from NewsAPI.

processed_data -- Processed headlines from different news sources.

stocknews -- News data from Kaggle.

The Guardian -- News (raw data) from Guardian.
