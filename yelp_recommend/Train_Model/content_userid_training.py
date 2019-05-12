import sys, re, os, operator
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row
from pyspark import SparkConf
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.feature import VectorAssembler, IDF
from pyspark.ml.feature import Word2Vec, Word2VecModel

conf = SparkConf().set('spark.dynamicAllocation.maxExecutors', 16)
spark = SparkSession.builder.appName('example code').config(conf=conf).getOrCreate()
sc = spark.sparkContext

def CosineSim(vec1, vec2):
    return np.dot(vec1, vec2) / np.sqrt(np.dot(vec1, vec1)) / np.sqrt(np.dot(vec2, vec2))



def main(file1, file2, output_model):
    data = spark.read.parquet(file1)
    data.createOrReplaceTempView('review')
    df_business = spark.read.parquet(file2)
    schema = StructType([   
                            StructField("business_id", StringType(), True)
                            ,StructField("score", IntegerType(), True)
                            ,StructField("input_business_id", StringType(), True)
                        ])
    
    similar_businesses_df = spark.createDataFrame([], schema)
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
    idf = IDF(inputCol="raw_features", outputCol="idf_vec")
    word2Vec = Word2Vec(vectorSize = 500, minCount = 5, inputCol = 'nonstopwrd', outputCol = 'word_vec', seed=123)
    pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, countVectorizer, idf, word2Vec])
    pipeline_model = pipeline.fit(review_df)
    pipeline_model.write().overwrite().save(output_model)


if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_model = sys.argv[3]
    main(file1, file2, output_model)
