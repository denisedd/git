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

def getKeyWordsRecoms(key_words, sim_bus_limit, tfidf_model, model, lda_vec):
    
    input_words_df = sc.parallelize([('abc', key_words)]).toDF(['business_id', 'text'])
    
    # transform the the key words to vectors
    input_tfidf = tfidf_model.transform(input_words_df)
    lda_x = model.transform(input_tfidf)
    
    # choose lda vectors
    input_key_wordsvec = lda_x.select('topicDistribution').collect()[0][0]
    
    # get similarity
    sim_bus_byword_rdd = sc.parallelize((i[0], float(CosineSim(input_key_wordsvec, i[1]))) for i in lda_vec)

    sim_bus_byword_df = spark.createDataFrame(sim_bus_byword_rdd) \
         .withColumnRenamed('_1', 'business_id') \
         .withColumnRenamed('_2', 'score') \
         .orderBy("score", ascending = False)
    
    # return top 10 similar businesses
    a = sim_bus_byword_df.limit(sim_bus_limit)
    return a

def main(input_file, bus_parquet, input_model):
    data = spark.read.parquet(input_file)
    df_business = spark.read.parquet(bus_parquet)
    
    df = data.select('business_id', 'text')
    #df_review = df.groupby('business_id').agg(functions.collect_set('text')).show(100)
    review_rdd = df.rdd.map(tuple).reduceByKey(operator.add)
    review_df = spark.createDataFrame(review_rdd).withColumnRenamed('_1', 'business_id').withColumnRenamed('_2', 'text')
    
    # create text preprocessing pipeline
    # Build the pipeline
    # tokenize review
    regexTokenizer = RegexTokenizer(gaps=False, pattern='\w+', inputCol='text', outputCol='text_token')
    #yelpTokenDF = regexTokenizer.transform(review_df)
    
    # filter stopwords
    stopWordsRemover = StopWordsRemover(inputCol='text_token', outputCol='nonstopwrd')
    #yelp_remove_df = stopWordsRemover.transform(yelpTokenDF)

    # TF
    countVectorizer = CountVectorizer(inputCol = 'nonstopwrd', outputCol='raw_features', minDF=2)
    #yelp_CountVec = cv.transform(yelp_remove_df)

    # IDF
    idf = IDF(inputCol="raw_features", outputCol="features")

    pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, countVectorizer, idf])
    #tfidf_model = pipeline.fit(review_df)
    #tfidf_model.write().overwrite().save('tfidf_model')

    tfidf_model = PipelineModel.load('tfidf_model')
    result_tfidf = tfidf_model.transform(review_df) 
    yelp = result_tfidf

    lda = LDA(k=15, maxIter=100)
    # already saved model
    #model = lda.fit(yelp)
    # save model
    #model.write().overwrite().save(input_model)

    model = LocalLDAModel.load(input_model)
    # lda output column topicDistribution
    lda_df = model.transform(yelp)

    # test result
    x = sc.parallelize([('aaa', 'chicken cheese burger')]).toDF(['business_id', 'text'])
    x_tfidf = tfidf_model.transform(x)
    lda_x = model.transform(x_tfidf)
    
    input_vec = lda_x.select('topicDistribution').collect()[0][0]
    lda_vec = lda_df.select('business_id', 'topicDistribution').rdd.map(lambda x: (x[0], x[1])).collect()

    # compute similarity
    t = sc.parallelize((i[0], float(CosineSim(input_vec, i[1]))) for i in lda_vec)

    # recommendation's cosine values
    similarity = spark.createDataFrame(t).withColumnRenamed('_1', 'business_id').withColumnRenamed('_2', 'similarity_score')
    df_result = df_business.join(similarity, 'business_id', 'right').select(similarity['business_id'] ,'similarity_score', 'categories').orderBy('similarity_score', ascending = False)
    
    result = getKeyWordsRecoms('chicken cheese burger', 20, tfidf_model, model, lda_vec)
    result.show()
    
if __name__ == '__main__':
    input_file = sys.argv[1]
    bus_parquet = sys.argv[2]
    input_model = sys.argv[3]
    main(input_file, bus_parquet, input_model)
