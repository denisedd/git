
# coding: utf-8

# In[ ]:


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

conf = SparkConf().set('spark.dynamicAllocation.maxExecutors', 16)
spark = SparkSession.builder.appName('example code').config(conf=conf).getOrCreate()
sc = spark.sparkContext

def list_tuple(my_list):
    if my_list[0] != None:
        return [[x, my_list[1]] for x in my_list[0]]
    else:
        return [['Empty', my_list[1]]]

def CosineSim(vec1, vec2):
    return np.dot(vec1, vec2) / np.sqrt(np.dot(vec1, vec1)) / np.sqrt(np.dot(vec2, vec2))

def main(input_file, input_model):
    data = spark.read.parquet(input_file)
    df = data.select('business_id', 'text')
    #df_review = df.groupby('business_id').agg(functions.collect_set('text')).show(100)
    review_rdd = df.rdd.map(tuple).reduceByKey(operator.add)
    review_df = spark.createDataFrame(review_rdd).withColumnRenamed('_1', 'business_id').withColumnRenamed('_2', 'text')
    
    # create text preprocessing pipeline
    # Build the pipeline
    # tokenize review
    regexTokenizer = RegexTokenizer(gaps=False, pattern='\w+', inputCol='text', outputCol='text_token')
    yelpTokenDF = regexTokenizer.transform(review_df)
    
    # filter stopwords
    stopWordsRemover = StopWordsRemover(inputCol='text_token', outputCol='nonstopwrd')
    yelp_remove_df = stopWordsRemover.transform(yelpTokenDF)

    # TF
    countVectorizer = CountVectorizer(inputCol = 'nonstopwrd', outputCol='raw_features', minDF=2)
    cv = countVectorizer.fit(yelp_remove_df)
    yelp_CountVec = cv.transform(yelp_remove_df)

    # IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    idfModel = idf.fit(yelp_CountVec)
    result_tfidf = idfModel.transform(yelp_CountVec) 
    yelp = result_tfidf
    
    if not os.path.exists(input_model):
        #Train LDA model
        lda = LDA(k=20, maxIter=100)
        model = lda.fit(yelp)
        # save model
        model.write().overwrite().save('tfidf_lda_model')
    else:
        model = LocalLDAModel.load(input_model)
    # lda output column topicDistribution
    lda_df = model.transform(yelp)

    # test result
    x = sc.parallelize([('aaa', 'breakfast omelet')]).toDF(['business_id', 'text'])
    x_token = regexTokenizer.transform(x)
    x_swr = stopWordsRemover.transform(x_token)
    x_count = cv.transform(x_swr)
    x_tfidf = idfModel.transform(x_count)
    lda_x = model.transform(x_tfidf)
    input_vec = lda_x.select('topicDistribution').collect()[0][0]
    lda_vec = lda_df.select('business_id', 'topicDistribution').rdd.map(lambda x: (x[0], x[1])).collect()

    # compute similarity
    t = sc.parallelize((i[0], float(CosineSim(input_vec, i[1]))) for i in lda_vec)

    # recommendation's cosine values
    similarity = spark.createDataFrame(t).withColumnRenamed('_1', 'business_id').withColumnRenamed('_2', 'similarity').orderBy('similarity', ascending = False)
    similarity.show(10)
    
    
if __name__ == '__main__':
    input_file = sys.argv[1]
    input_model = sys.argv[2]
    main(input_file, input_model)

