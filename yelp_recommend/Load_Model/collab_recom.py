import sys, re, os, operator
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row
from pyspark import SparkConf
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS, ALSModel
import collaborative_filtering as CF

conf = SparkConf().set('spark.dynamicAllocation.maxExecutors', 16)
spark = SparkSession.builder.appName('example code').config(conf=conf).getOrCreate()
sc = spark.sparkContext



def main(model, input_1, input_2, u_id, sim_bus_limit=3):
    #business_df = spark.read.json(input_1)
    #bus_df_schema = StructType(
          # [StructField("business_id", StringType(), True),
          # StructField("businessId", IntegerType(), True)])
    #bus_id = business_df.select('business_id')
    #business_newid_df = spark.createDataFrame(bus_id.repartition(16).rdd.map(lambda x: x[0]).zipWithIndex(), bus_df_schema)
    #business_new_df = business_df.join(business_newid_df, 'business_id', 'inner').select('businessId', 'business_id', 'name', 'categories', 'latitude', 'longitude')
    # saved as json file
    #business_new_df.write.json('collab_business_JSON', compression='gzip', mode='overwrite')
    
    # saved above method into parquet to boost the code running time
    business_new_df = spark.read.parquet(input_1)
    business_new_df.cache()

    #user_df = spark.read.json(input_2)
    #user_df_schema = StructType(
        #[StructField("user_id", StringType(), True),
       # StructField("userId", IntegerType(), True)])
        
    #user_id = user_df.select('user_id')
    #user_newid_df = spark.createDataFrame(user_id.repartition(16).rdd.map(lambda x: x[0]).zipWithIndex(), user_df_schema)
    #saved as json file
    #user_newid_df.write.json('collab_user_JSON', compression='gzip', mode='overwrite')
    #user_newid_df.write.mode('overwrite').parquet('collab_user_Idnum')

    # saved above method into parquet to boost the code running time
    user_newid_df = spark.read.parquet(input_2)
    user_newid_df.cache()

    bid = business_new_df.select('businessId').distinct()
    uid =  user_newid_df.filter(col('user_id') == u_id).select('userId').collect()[0][0]
    predDF = bid.withColumn("userId", lit(uid))
    alsn_model = ALSModel.load(model)
    predictions_ureq = alsn_model.transform(predDF)
    recommend_df = predictions_ureq.sort(predictions_ureq.prediction.desc()).select('businessId','prediction').limit(sim_bus_limit)
    result_df = business_new_df.join(recommend_df, 'businessId', 'right').select('business_id', 'name', 'categories', 'latitude', 'longitude', 'prediction')
    result_df.orderBy('prediction', ascending = False).show()

if __name__ == '__main__':
    model = sys.argv[1]
    input_1 = sys.argv[2]
    input_2 = sys.argv[3]
    u_id = sys.argv[4]
    main(model, input_1, input_2, u_id)