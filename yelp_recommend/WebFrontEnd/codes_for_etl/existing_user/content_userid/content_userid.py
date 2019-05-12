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



def main(file1, file2, input_model, u_id, sim_bus_limit=3):
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
    #vectorAssembler = VectorAssembler(inputCols=['idf_vec', 'word_vec'], outputCol='comb_vec')
    pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, countVectorizer, idf, word2Vec])
    #pipeline_model = pipeline.fit(review_df)
    #pipeline_model.write().overwrite().save('content_userid')

    pipeline_model = PipelineModel.load(input_model)
    reviews_by_business_df = pipeline_model.transform(review_df) 
    all_business_vecs = reviews_by_business_df.select('business_id', 'word_vec').rdd.map(lambda x: (x[0], x[1])).collect()
    usr_rev_bus = spark.sql('SELECT distinct business_id FROM review where stars >= 3.0 and user_id = "{}"'.format(u_id))
    
    bus_list = [i for i in usr_rev_bus.collect()]

    for b_id in bus_list:
        input_vec = [(r[1]) for r in all_business_vecs if r[0] == b_id[0]][0]
        similar_business_rdd = sc.parallelize((i[0], float(CosineSim(input_vec, i[1]))) for i in all_business_vecs)
        similar_business_df = spark.createDataFrame(similar_business_rdd).withColumnRenamed('_1', 'business_id').withColumnRenamed('_2', 'score').orderBy("score", ascending = False)
        similar_business_df = similar_business_df.filter(col("business_id") != b_id[0]).limit(10)
        similar_business_df = similar_business_df.withColumn('input_business_id', lit(b_id[0]))
        # get restaurants similar to the user_id
        result = similar_businesses_df.union(similar_business_df)
    result.cache()
    # filter out those have been reviewd before by the user
    d = [i[0] for i in usr_rev_bus.collect()]
    df_1 = result.filter(~(col('business_id').isin(d))).select('business_id', 'score')
    #df_1= result.join(usr_rev_bus, 'business_id', 'left_outer').where(col("usr_rev_bus.business_id").isNull()).select([col('result.business_id'),col('result.score')])
    df_2 = df_1.orderBy("score", ascending = False).limit(sim_bus_limit)
    df_result = df_business.join(df_2, 'business_id', 'right').select('business_id', 'score', 'name', 'categories', 'latitude', 'longitude')
    df_result.show()

if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    input_model = sys.argv[3]
    u_id = sys.argv[4]
    main(file1, file2, input_model, u_id)
