
# coding: utf-8

# In[5]:


import quandl
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import date, timedelta
from pandas.tseries.offsets import BDay
import warnings
warnings.filterwarnings('ignore')
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

def stocks(start_date, end_date):
    # use quandl to acquire nasdaq composite
    ndq = quandl.get("NASDAQOMX/COMP-NASDAQ",
                        start_date = start_date, 
                        end_date = end_date)
    ndq_df = ndq.reset_index()
    val = ndq_df['Index Value'].tolist()
    return ndq_df, val

def analyze_sentiment(day_headline_lst):
    sentiments = []
    sid = SentimentIntensityAnalyzer()
    for headline in day_headline_lst:
        lst = []
        if len(headline) >3:
            for each in headline.split('$@$'):
                sentiment = sid.polarity_scores(each)['compound']
                lst.append(sentiment)
            arr_mean = np.mean(lst, axis=0)
        else:
            arr_mean = 0.0
        sentiments.append(arr_mean)
    return sentiments
    
def main(str1, str2, day):
    # filter string len >5
    if len(str1) <= 5:
        today = pd.datetime.today()
        prev_day = today - BDay(30)
        prev_date = prev_day.strftime('%Y-%m-%d')
        today_date = today.strftime('%Y-%m-%d')
        df, price = stocks(prev_date, today_date)
    else:
        price = [float(a) for a in str1.split('|')]
        
    model = load_model('model/model_15.hdf5')
    data_stock = np.asarray(price).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    min_max_price = scaler.fit_transform(data_stock)
    
    # calculate sentiment
    day_headline_lst = str2.split('|||')
    senti_day = analyze_sentiment(day_headline_lst)
    data_senti = np.asarray(senti_day).reshape(15, -1)
    
    # concate senti and stock price
    data = np.concatenate((data_senti, min_max_price),axis=1)
    test_data = data.reshape(-1, 15, 2)
    
    predict = []
    for i in range(day):
        Xt = model.predict(test_data)
        val = scaler.inverse_transform(Xt)
        price = ''.join([str(price) for i in val for price in i])
        predict.append(price)
        c = np.asarray([[0, float(price)]])
        new_test_data = test_data[:, 1:, :]
        aa = new_test_data.reshape(-1, 2)
        test_1 = np.concatenate((aa, c))
        test_data = test_1.reshape(-1, 15, 2)
    return predict

if __name__ == "__main__":
    price = sys.argv[1]
    headline = sys.argv[2]
    predict_day = sys.argv[3]
    main(price, headline, predict_day)

