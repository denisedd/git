
# coding: utf-8

# In[ ]:


import sys, re
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql.types import MapType, StructType
from pyspark.sql.functions import udf, explode
from pyspark import SparkConf
import uuid
import json
cluster_seeds = ['199.60.17.188', '199.60.17.216']
conf = SparkConf().set('spark.cassandra.connection.host', ','.join(cluster_seeds)).set('spark.dynamicAllocation.maxExecutors', 16)
spark = SparkSession.builder.appName('example code').config(conf=conf).getOrCreate()
sc = spark.sparkContext

def json_key_value_2(file):
    data = json.loads(file)
    business_id=data['business_id']
    time=data['time']
    
    return (time, business_id)

def json_key_value_1(file):
    data = json.loads(file)
    business_id=data['business_id']
    attributes=data['attributes']
    categories=data['categories']
    hours=data['hours']
    
    return (business_id, attributes, categories, hours)
    
def main(inputs, keyspace, table):
    if table == "yelp_business":
        business_schema = StructType([
            types.StructField('business_id', types.StringType(), True),    
            types.StructField('name', types.StringType(), True),
            types.StructField('neighborhood', types.StringType(), True),
            types.StructField('address', types.StringType(), True),
            types.StructField('city', types.StringType(), True),
            types.StructField('state', types.StringType(), True),
                types.StructField('postal_code', types.StringType(), True),
            types.StructField('latitude', types.FloatType(), True),
            types.StructField('longitude', types.FloatType(), True),
            types.StructField('stars', types.FloatType(), True),
            types.StructField('review_count', types.LongType(), True),
            types.StructField('is_open', types.IntegerType(), True)
            ])
        business = spark.read.json(inputs, schema=business_schema)
        df = business.drop('neighborhood').filter(business.is_open==1)
        df.cache()
        business_data = sc.textFile(inputs).map(json_key_value_1).map(lambda x: Row(x[0], x[1], x[2], x[3]))
        df_1 = business_data.toDF()
        df_2 = df_1.withColumnRenamed("_1", "bus_id").withColumnRenamed("_2", "attributes").withColumnRenamed("_3", "categories").withColumnRenamed("_4", "hours")
        df_2.cache()
        result = df.join(df_2, df.business_id==df_2.bus_id,how='inner').drop(df_2.bus_id)
        
    elif table == "yelp_checkin":
        
        checkin_data = sc.textFile(inputs).map(json_key_value_2).map(lambda x: Row(str(uuid.uuid1()), x[0], x[1]))
        df = checkin_data.toDF().cache()
        df_1 = df.withColumnRenamed("_1", "id").withColumnRenamed("_2", "time").withColumnRenamed("_3", "business_id")
        result = df_1
        
        
    if table == "yelp_review":
        reviews_schema = types.StructType([
            types.StructField('business_id', types.StringType(), True),    
            types.StructField('cool', types.LongType(), True),
            types.StructField('date', types.DateType(), True),
            types.StructField('funny', types.LongType(), True),
            types.StructField('review_id', types.StringType(), True),
            types.StructField('stars', types.LongType(), True),
            types.StructField('text', types.StringType(), True),
            types.StructField('useful', types.LongType(), True),
            types.StructField('user_id', types.StringType(), True)
            ])
        
        reviews = spark.read.json(inputs, schema=reviews_schema)
        uuidUdf= udf(lambda : str(uuid.uuid1()),types.StringType())
        result = reviews.withColumn("id",uuidUdf())
    result.repartition(300).write.format("org.apache.spark.sql.cassandra").options(table=table, keyspace=keyspace).save()
    
if __name__ == '__main__':
    inputs = sys.argv[1]
    keyspace = sys.argv[2]
    table = sys.argv[3]
    #output = sys.argv[3]
    main(inputs, keyspace, table)

