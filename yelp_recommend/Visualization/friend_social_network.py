
# coding: utf-8

# In[ ]:


import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row
from pyspark import SparkConf
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window  
from random import randrange
conf = SparkConf().set('spark.dynamicAllocation.maxExecutors', 16)
spark = SparkSession.builder.appName('example code').config(conf=conf).getOrCreate()
sc = spark.sparkContext

def main(user_inputs):
    df_user = spark.read.parquet(user_inputs)
    df_user1 = df_user.drop('friends')
    #df_review = spark.read.json(review_inputs)
    #df_business = spark.read.json(business_inputs)
    #df_review.cache()
    df_1 = df_user.filter(df_user['friends']!='None').select('user_id','friends', 'review_count')
    df_1.cache()
    df_2 = df_1.orderBy('review_count', ascending=False).limit(1)
    df_2 = df_2.drop('review_count').cache()
    df_friend = df_2.withColumn('friends', explode(split(col('friends'), ',')))
    w = Window.orderBy("friends")
    df_new = df_friend.withColumn('ID', row_number().over(w))
    df_new = df_new.withColumn('Index', df_new.ID.cast('string')).drop('ID')
    df_new1 = df_new
    df_new2 = df_new1.withColumnRenamed('friends', 'user_id1').withColumnRenamed('user_id', 'friends').withColumnRenamed('user_id1', 'user_id')

    df_new_1 = df_new.withColumn('Relationship', functions.concat(functions.lit('A-'), functions.col('Index')))
    
    df_new_2 = df_new_1.withColumn('LineX', functions.lit(1200)).withColumn('LineY', functions.lit(2500))
    df_new_3 = df_new_2.withColumn('CircleY', df_new_2['LineY']).withColumn('INDEX1', functions.lit(1)).withColumn('INDEX2', df_new_2['user_id']).drop('friends')



    x = udf(lambda: randrange(500, 8000, 50), IntegerType())
    y = udf(lambda: randrange(1000, 8000, 20), IntegerType())
    df_new2_1 = df_new2.withColumn('Relationship', functions.concat(functions.lit('A-'), functions.col('Index')))
    df_new2_2 = df_new2_1.withColumn('LineX', x()).withColumn('LineY', y())
    df_new2_3 = df_new2_2.withColumn('CircleY', df_new2_2['LineY']).drop('friends')

    df_result = df_new_3.join(df_new2_3, ['user_id', 'Index', 'Relationship', 'LineX', 'LineY', 'CircleY'], 'full')
    df_result1 = df_result.orderBy('Index').cache()
    #df_result1.repartition(1).write.json('friend_df1', mode='overwrite')

    ##########################################################################
    df_2 = df_1.orderBy('review_count', ascending=False).limit(2)
    df_2 = df_2.drop('review_count').filter(df_2['user_id']!='Ud6j2HN40qEhycq2PuwcBA').cache()
    df_friend = df_2.withColumn('friends', explode(split(col('friends'), ',')))
    w = Window.orderBy("friends")
    df_new = df_friend.withColumn('ID', row_number().over(w))
    df_new = df_new.withColumn('Index1', df_new.ID.cast('string'))
    df_new = df_new.withColumn('Index', df_new['Index1']+300).drop('ID', 'Index1')
    df_new1 = df_new
    df_new2 = df_new1.withColumnRenamed('friends', 'user_id1').withColumnRenamed('user_id', 'friends').withColumnRenamed('user_id1', 'user_id')

    df_new_1 = df_new.withColumn('Relationship', functions.concat(functions.lit('B-'), functions.col('Index')))
    
    df_new_2 = df_new_1.withColumn('LineX', functions.lit(1700)).withColumn('LineY', functions.lit(6500))
    df_new_3 = df_new_2.withColumn('CircleY', df_new_2['LineY']).withColumn('INDEX1', functions.lit(2)).withColumn('INDEX2', df_new_2['user_id']).drop('friends')

    #####################################################################
    x = udf(lambda: randrange(500, 8000, 25), IntegerType())
    y = udf(lambda: randrange(1000, 8000, 35), IntegerType())
    df_new2_1 = df_new2.withColumn('Relationship', functions.concat(functions.lit('B-'), functions.col('Index')))
    df_new2_2 = df_new2_1.withColumn('LineX', x()).withColumn('LineY', y())
    df_new2_3 = df_new2_2.withColumn('CircleY', df_new2_2['LineY']).drop('friends')

    df_result = df_new_3.join(df_new2_3, ['user_id', 'Index', 'Relationship', 'LineX', 'LineY', 'CircleY'], 'full')
    df_result2 = df_result.orderBy('Index').cache()
    
    df_result3 = df_result1.join(df_result2, ['user_id', 'Index', 'Relationship', 'LineX', 'LineY', 'CircleY', 'INDEX1', 'INDEX2'], 'full')
    #df_result3.repartition(1).write.json('friend_df2', mode='overwrite')

      ##########################################################################
      ##########################################################################
    df_2 = df_1.orderBy('review_count', ascending=False).limit(3)
    df_2 = df_2.drop('review_count').filter((df_2['user_id']!='Ud6j2HN40qEhycq2PuwcBA')|(df_2['user_id']!='zXQuFIgNgARtX6Nf5hIWIQ')).cache()
    df_friend = df_2.withColumn('friends', explode(split(col('friends'), ',')))
    w = Window.orderBy("friends")
    df_new = df_friend.withColumn('ID', row_number().over(w))
    df_new = df_new.withColumn('Index1', df_new.ID.cast('string'))
    df_new = df_new.withColumn('Index', df_new['Index1']+650).drop('ID', 'Index1')
    df_new1 = df_new
    df_new2 = df_new1.withColumnRenamed('friends', 'user_id1').withColumnRenamed('user_id', 'friends').withColumnRenamed('user_id1', 'user_id')

    df_new_1 = df_new.withColumn('Relationship', functions.concat(functions.lit('C-'), functions.col('Index')))
    
    df_new_2 = df_new_1.withColumn('LineX', functions.lit(3700)).withColumn('LineY', functions.lit(7200))
    df_new_3 = df_new_2.withColumn('CircleY', df_new_2['LineY']).withColumn('INDEX1', functions.lit(3)).withColumn('INDEX2', df_new_2['user_id']).drop('friends')

    #####################################################################
    x = udf(lambda: randrange(500, 8000, 60), IntegerType())
    y = udf(lambda: randrange(1000, 8000, 18), IntegerType())
    df_new2_1 = df_new2.withColumn('Relationship', functions.concat(functions.lit('C-'), functions.col('Index')))
    df_new2_2 = df_new2_1.withColumn('LineX', x()).withColumn('LineY', y())
    df_new2_3 = df_new2_2.withColumn('CircleY', df_new2_2['LineY']).drop('friends')

    df_result = df_new_3.join(df_new2_3, ['user_id', 'Index', 'Relationship', 'LineX', 'LineY', 'CircleY'], 'full')
    df_result5 = df_result.orderBy('Index').cache()
    
    df_result6 = df_result3.join(df_result5, ['user_id', 'Index', 'Relationship', 'LineX', 'LineY', 'CircleY', 'INDEX1', 'INDEX2'], 'full')
    #df_result6.repartition(1).write.json('friend_df3', mode='overwrite')

          ##########################################################################
      ##########################################################################
    df_2 = df_1.orderBy('review_count', ascending=False).limit(4)
    df_2 = df_2.drop('review_count').filter((df_2['user_id']!='Ud6j2HN40qEhycq2PuwcBA')|(df_2['user_id']!='zXQuFIgNgARtX6Nf5hIWIQ')|(df_2['user_id']!='_72QdatN4fPKO5QSNGFYRA')).cache()
    df_friend = df_2.withColumn('friends', explode(split(col('friends'), ',')))
    w = Window.orderBy("friends")
    df_new = df_friend.withColumn('ID', row_number().over(w))
    df_new = df_new.withColumn('Index1', df_new.ID.cast('string'))
    df_new = df_new.withColumn('Index', df_new['Index1']+1100).drop('ID', 'Index1')
    df_new1 = df_new
    df_new2 = df_new1.withColumnRenamed('friends', 'user_id1').withColumnRenamed('user_id', 'friends').withColumnRenamed('user_id1', 'user_id')

    df_new_1 = df_new.withColumn('Relationship', functions.concat(functions.lit('D-'), functions.col('Index')))
    
    df_new_2 = df_new_1.withColumn('LineX', functions.lit(4700)).withColumn('LineY', functions.lit(2000))
    df_new_3 = df_new_2.withColumn('CircleY', df_new_2['LineY']).withColumn('INDEX1', functions.lit(4)).withColumn('INDEX2', df_new_2['user_id']).drop('friends')

    #####################################################################
    x = udf(lambda: randrange(500, 8000, 35), IntegerType())
    y = udf(lambda: randrange(1000, 8000, 39), IntegerType())
    df_new2_1 = df_new2.withColumn('Relationship', functions.concat(functions.lit('D-'), functions.col('Index')))
    df_new2_2 = df_new2_1.withColumn('LineX', x()).withColumn('LineY', y())
    df_new2_3 = df_new2_2.withColumn('CircleY', df_new2_2['LineY']).drop('friends')

    df_result = df_new_3.join(df_new2_3, ['user_id', 'Index', 'Relationship', 'LineX', 'LineY', 'CircleY'], 'full')
    df_result7 = df_result.orderBy('Index').cache()
    
    df_result8 = df_result6.join(df_result7, ['user_id', 'Index', 'Relationship', 'LineX', 'LineY', 'CircleY', 'INDEX1', 'INDEX2'], 'full')

             ##########################################################################
      ##########################################################################
    df_2 = df_1.orderBy('review_count', ascending=False).limit(5)
    df_2 = df_2.drop('review_count').filter((df_2['user_id']!='Ud6j2HN40qEhycq2PuwcBA')|(df_2['user_id']!='zXQuFIgNgARtX6Nf5hIWIQ')|(df_2['user_id']!='_72QdatN4fPKO5QSNGFYRA')|(df_2['user_id']!='TvRqHTHzTRLry_4u2nafAA')).cache()
    df_friend = df_2.withColumn('friends', explode(split(col('friends'), ',')))
    w = Window.orderBy("friends")
    df_new = df_friend.withColumn('ID', row_number().over(w))
    df_new = df_new.withColumn('Index1', df_new.ID.cast('string'))
    df_new = df_new.withColumn('Index', df_new['Index1']+1700).drop('ID', 'Index1')
    df_new1 = df_new
    df_new2 = df_new1.withColumnRenamed('friends', 'user_id1').withColumnRenamed('user_id', 'friends').withColumnRenamed('user_id1', 'user_id')

    df_new_1 = df_new.withColumn('Relationship', functions.concat(functions.lit('E-'), functions.col('Index')))
    
    df_new_2 = df_new_1.withColumn('LineX', functions.lit(7200)).withColumn('LineY', functions.lit(5500))
    df_new_3 = df_new_2.withColumn('CircleY', df_new_2['LineY']).withColumn('INDEX1', functions.lit(5)).withColumn('INDEX2', df_new_2['user_id']).drop('friends')

    #####################################################################
    x = udf(lambda: randrange(500, 8000, 70), IntegerType())
    y = udf(lambda: randrange(1000, 8000, 23), IntegerType())
    df_new2_1 = df_new2.withColumn('Relationship', functions.concat(functions.lit('E-'), functions.col('Index')))
    df_new2_2 = df_new2_1.withColumn('LineX', x()).withColumn('LineY', y())
    df_new2_3 = df_new2_2.withColumn('CircleY', df_new2_2['LineY']).drop('friends')

    df_result = df_new_3.join(df_new2_3, ['user_id', 'Index', 'Relationship', 'LineX', 'LineY', 'CircleY'], 'full')
    df_result9 = df_result.orderBy('Index').cache()
    
    df_result10 = df_result8.join(df_result9, ['user_id', 'Index', 'Relationship', 'LineX', 'LineY', 'CircleY', 'INDEX1', 'INDEX2'], 'full')

    df_result10.repartition(1).write.json('friend_df6', mode='overwrite')
    
if __name__ == '__main__':
    user_inputs = sys.argv[1]
    #review_inputs = sys.argv[2]
    #business_inputs = sys.argv[3]
    #u_id = sys.argv[4]
    main(user_inputs)

