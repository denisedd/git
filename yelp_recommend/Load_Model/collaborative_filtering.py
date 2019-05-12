
# coding: utf-8

# In[ ]:


import sys, re
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row
from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
conf = SparkConf().set('spark.dynamicAllocation.maxExecutors', 16)
spark = SparkSession.builder.appName('example code').config(conf=conf).getOrCreate()
sc = spark.sparkContext

def getCollabRecom(u_id, all_userRecoms, business_new_df):
    userRecom_df =  spark.createDataFrame(all_userRecoms.filter(col('user_id') == u_id).rdd.flatMap(lambda p: p[1]))
    collab_df = business_new_df.join(userRecom_df, 'businessId', 'inner').drop('businessId')

    return collab_df

def main(input_1, input_2, input_3):

    business_df = spark.read.json(input_1)
    user_df = spark.read.json(input_2)
    review_df = spark.read.json(input_3)
    
    # Spark ALS implementation requires the rating matrix to have the follwoing data types
    user_df_schema = StructType(
        [StructField("user_id", StringType(), True),
        StructField("userId", IntegerType(), True)])
        
    user_id = user_df.select('user_id')
    user_newid_df = spark.createDataFrame(user_id.rdd.map(lambda x: x[0]).zipWithIndex(), user_df_schema)
    
    # add the new userId column the user dataframe
    user_new_df = user_df.join(user_newid_df, 'user_id', 'inner').select('userId', 'user_id', 'name')
    
    bus_df_schema = StructType(
            [StructField("business_id", StringType(), True),
            StructField("businessId", IntegerType(), True)])
    bus_id = business_df.select('business_id')
    business_newid_df = spark.createDataFrame(bus_id.rdd.map(lambda x: x[0]).zipWithIndex(), bus_df_schema)
    business_new_df = business_df.join(business_newid_df, 'business_id', 'inner').select('businessId', 'business_id', 'name', 'categories', 'latitude', 'longitude')
    
    # map new userId and businessId in the review dataframe
    review_df = review_df.select('user_id', 'business_id', 'stars')
    review_userId_df = review_df.join(user_newid_df, "user_id", 'inner').select('business_id', 'userId', 'user_id', 'stars')
    # map the businessId
    review_userId_df = review_userId_df.join(business_newid_df, "business_id", 'inner').select('user_id', 'business_id', 'stars', 'userId', 'businessId')
    
    #create the rating dataframe required by the ALS model
    rating_df = review_userId_df.select('userId', 'businessId', review_userId_df.stars.cast('float').alias('rating'))
    rating_df.cache()
    #print(' Rating matrx no. of rows :', rating_df.count())

    (train, test) = rating_df.randomSplit([0.8, 0.2], seed=123)
    # Cross Validation
    als = ALS(userCol="userId", itemCol="businessId", ratingCol="rating", coldStartStrategy="drop")
    param_grid = ParamGridBuilder().addGrid(als.rank, [10, 15, 20]).addGrid(als.maxIter, [10, 15, 20]).build()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating")
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5, seed=123)
    cv_als_model = cv.fit(train)

    # Evaluate the model by compu
    als_predictions = cv_als_model.bestModel.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(als_predictions)
    print("Root-mean-square error"+ str(rmse))
    #rmse = 1.559099

    #best_model = cv_als_model.bestModel
    #best_rank is 20
    #best_model.rank

    #best_maxIter is 20
    #(best_model._java_obj.parent().getMaxIter())
    # drop columns for Nan values (ColdStrategy parameter) and tune ALS model
    als = ALS(rank=20, maxIter=20, regParam=0.3, userCol="userId", itemCol="businessId", ratingCol="rating", coldStartStrategy="drop", seed=123)
    alsb_model = als.fit(train)
    alsb_predictions = alsb_model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(alsb_predictions)
    # save the ALS model
    alsb_model.write().overwrite().save('als_model')
    print("alsb_model Root-mean-square error = " + str(rmse))
    # rmse is 1.45023

    alsn_model = ALSModel.load('als_model')
    # generate top 10 business recommendations for each user
    userRecoms = alsn_model.recommendForAllUsers(10)

    all_userRecoms = userRecoms.join(user_newid_df, 'userId', 'inner').select('userId', 'recommendations', 'user_id')
    all_userRecoms.cache()
    

    # test and show recommendations
    u_id = 'ZWD8UH1T7QXQr0Eq-mcWYg'

    userFlatRec =  spark.createDataFrame(all_userRecoms.filter(col('user_id') == u_id).rdd.flatMap(lambda p: p[1]))
    # businessId|            rating|
    #+----------+------------------+
    #|    171476|5.4555559158325195|
    #|     25624|5.3495965003967285|
    #|     14049| 5.271500110626221|

    #show the recommeded restaurants details
    collab_df = business_new_df.join(userFlatRec, 'businessId', 'inner').drop('businessId')

    result = getCollabRecom(u_id, all_userRecoms, business_new_df)
    result.show()

    
if __name__ == '__main__':
    input_1 = sys.argv[1]
    input_2 = sys.argv[2]
    input_3 = sys.argv[3]
    main(input_1, input_2, input_3)

