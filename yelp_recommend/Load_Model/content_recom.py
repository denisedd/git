import sys, re, os, operator
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row
from pyspark import SparkConf
import numpy as np
from pyspark.sql.types import *
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.feature import VectorAssembler, IDF
from pyspark.ml.clustering import LDA, LocalLDAModel
import TFIDF_LDA1 as TF

conf = SparkConf().set('spark.dynamicAllocation.maxExecutors', 16)
spark = SparkSession.builder.appName('example code').config(conf=conf).getOrCreate()
sc = spark.sparkContext



def main(file1, file2, tfidf_model, tfidf_lda_model, sentiment_file, all_business_parquet, key_words):
    #data = spark.read.parquet('review_parquet')
    #df_business = spark.read.parquet('yelp_bus_parquet')
    data = spark.read.json(file1)
    df_business = spark.read.parquet(file2)
    
    df = data.select('business_id', 'text')
    #df_review = df.groupby('business_id').agg(functions.collect_set('text')).show(100)
    review_rdd = df.rdd.map(tuple).reduceByKey(operator.add)
    review_df = spark.createDataFrame(review_rdd).withColumnRenamed('_1', 'business_id').withColumnRenamed('_2', 'text')

    #tfidf_model = PipelineModel.load('tfidf_model')
    tfidf_model = PipelineModel.load(tfidf_model)
    result_tfidf = tfidf_model.transform(review_df) 
    yelp = result_tfidf

    lda = LDA(k=15, maxIter=100)
    model = LocalLDAModel.load(tfidf_lda_model)
    #model = LocalLDAModel.load('tfidf_lda_model')
    # lda output column topicDistribution
    lda_df = model.transform(yelp)
    lda_vec = lda_df.select('business_id', 'topicDistribution').rdd.map(lambda x: (x[0], x[1])).collect()
    
    #result = TF.getKeyWordsRecoms('chicken cheese burger', 20, tfidf_model, model, lda_vec)
    result = TF.getKeyWordsRecoms(key_words, 20, tfidf_model, model, lda_vec)
    df_sentiment = spark.read.json(sentiment_file)
    #df_sentiment = spark.read.json('yelp_review_sentiment')
    df_content_rest = df_sentiment.join(result, 'business_id', 'inner').orderBy("sentiment_score", ascending = False).limit(6)
    #all_busi_df = spark.read.parquet('all_business_parquet')
    all_busi_df = spark.read.parquet(all_business_parquet)
    df_rest_result = all_busi_df.join(df_content_rest, 'business_id', 'right').select('business_id', 'sentiment_score', 'name', 'categories','score', 'latitude', 'longitude')
    df_rest_result.show()

if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    tfidf_model = sys.argv[3]
    tfidf_lda_model = sys.argv[4]
    sentiment_file = sys.argv[5]
    all_business_parquet = sys.argv[6]
    key_words = sys.argv[7]
    main(file1, file2, tfidf_model, tfidf_lda_model, sentiment_file, all_business_parquet, key_words)
