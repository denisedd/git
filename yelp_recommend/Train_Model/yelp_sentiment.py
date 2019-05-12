
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql.types import *
from pyspark.sql.functions import udf, lit, sum
from pyspark import SparkConf
import uuid
import json
import keras
import nltk
import os
from nltk.corpus import stopwords
from keras.preprocessing import sequence
from keras.preprocessing import text as txt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras import backend as K
import tensorflow
from sklearn.model_selection import train_test_split
from keras.models import load_model
conf = SparkConf().set('spark.dynamicAllocation.maxExecutors', 16)
spark = SparkSession.builder.appName('example code').config(conf=conf).getOrCreate()
sc = spark.sparkContext


def binarize(x):
    # create a function to categorize low and high reviews as 0 or 1 for training
    if x <= 2:
        return 0
    if x == 5:
        return 1

def list_tuple(my_list):
  if my_list[0] != None:
    return [[x, my_list[1]] for x in my_list[0]]
  else:
    return [['Empty', my_list[1]]]

def sentiment_predictor(tk, model, input_str):
    if input_str !=None:
        dummy = ''
        sequences = tk.texts_to_sequences([dummy, input_str])
        padded_sequences = sequence.pad_sequences(sequences, maxlen = 500, padding = 'post')
        preds = model.predict(padded_sequences)
        return float(preds[1][0])
    else:
        return 0
    

def main(input_file, input_model, output):
    LVreview_schema = StructType([
        types.StructField('business_id', types.StringType(), True),    
        types.StructField('cool', types.IntegerType(), True),
        types.StructField('date', types.DateType(), True),
        types.StructField('funny', types.IntegerType(), True),
        types.StructField('review_id', types.StringType(), True),
        types.StructField('stars', types.LongType(), True),
        types.StructField('text', types.StringType(), True)
        ])
    reviewall = spark.read.json(input_file, schema=LVreview_schema)
    reviewall_df = reviewall.na.fill('')
    #reviewall_df.describe('stars').show() 
    review_low3 = reviewall_df.filter(reviewall_df['stars'] < 3)
    review_5stars = reviewall_df.filter(reviewall_df['stars'] == 5)
    #reviewall_df.groupBy('stars').count().show()
    # grab the low & high reviews for training
    user_reviews = reviewall_df.filter((reviewall_df['stars'] <3) | (reviewall_df['stars'] == 5))
    udfBinarize=udf(binarize, LongType())
    user_reviews = user_reviews.withColumn('overall', udfBinarize('stars'))
    user_reviews.groupBy('overall').count()
    subset = user_reviews.select('business_id', 'stars', 'text', 'overall')

    # make a list of all the documents to train the model on
    review_list = subset.select('text').rdd.map(lambda x: x[0]).collect()
    # tokensize the reviews
    tk = txt.Tokenizer(split=" ")
    tk.fit_on_texts(review_list)
    
    #create data set, both features and labels
    x = tk.texts_to_sequences(review_list)      # converts the text to numbers
    y = subset.select('overall').rdd.map(lambda x: x[0]).collect()

    # assign attributes to variables for consistency
    max_features = 20000   # the more the better
    max_length = 500  # cut texts after this number of words

    # pad the sequences so all arrays of same length
    x = sequence.pad_sequences(x, maxlen = max_length, padding = 'post')  

    #Take a look at the original subset versus the x array x[1]
    #subset.select('text').show(1)
    
    # Train & Fit the NN for Sentiment Analysis
    # Divide dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    #print(len(X_train),len(y_train),len(X_test),len(y_test))

    # check keras NN runs with gpu
    #print(K.tensorflow_backend._get_available_gpus())

    if os.path.exists(input_model):
        model = load_model(input_model)

    else:
        # if saved model not exist, create new model
        embedding_vector_length = 32
        model = Sequential()
        # Embedding layer with a vocabulary of 100000 (e.g. integer encoded words from 0 to 99999), 
        model.add(Embedding(100000, embedding_vector_length, input_length = max_length))
        # convolution filters, the number of rows in each convolution kernel
        model.add(Convolution1D(activation="relu", filters=32, kernel_size=4, padding="same"))
        # reduce the number of parameters in the model
        model.add(MaxPooling1D(pool_size=4))       # pooling (max) after convoluting
        model.add(Flatten())
        # every input shape of review in first layer is 500 (defined above)
        model.add(Dense(500, activation='relu'))     # relu and sigmoid
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        # target is in binary format, optimizer can be changed to 'sgd', 'RMSprop', 'Adagrad'
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #print(model.summary())
        
        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=32, verbose=1)   # batch size powers of two
        model.save('keras_nn_model')
        
    
    scores = model.evaluate(X_test, y_test, verbose=0)
    #print("Accuracy: %.2f%%" % (scores[1]*100))
    #Accuracy: 96.61%

    # test it
    sentiment_predictor(tk, model,"nothing special")
    sentiment_predictor(tk, model,'hello love hate')
    sentiment_predictor(tk, model,'disgusting')

    reviews_rdd = user_reviews.select('business_id', 'text').rdd.map(lambda x: x[:])
    #Each array is converted to a set of tuples.
    reviews_tuple = reviews_rdd.flatMap(lambda x: list_tuple(x))

    # user_reviews
    all_review = reviewall.select('review_id', 'text').rdd.map(lambda x: (x[0],x[1])).collect()
    score = sc.parallelize((i[0], sentiment_predictor(tk, model,i[1])) for i in all_review)
    sentiment = spark.createDataFrame(score).withColumnRenamed('_1', 'review_id').withColumnRenamed('_2', 'sentiment')
    allreview_df = reviewall.join(sentiment, 'review_id')
    df_sentiment_score = allreview_df.select('business_id', 'sentiment').groupby('business_id').avg().withColumnRenamed('avg(sentiment)', 'sentiment_score')
    df_sentiment_score.repartition(1).write.json(output, mode='overwrite')

if __name__ == '__main__':
    input_file = sys.argv[1]
    input_model = sys.argv[2]
    output = sys.argv[3]
    main(input_file, input_model, output)

