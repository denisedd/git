
# coding: utf-8

# In[ ]:


import sys, re
import ast
from pyspark.sql import SparkSession, functions, types, Row
from pyspark import SparkConf
from pyspark.sql.functions import lit, sum

cluster_seeds = ['199.60.17.188', '199.60.17.216']
conf = SparkConf().set('spark.cassandra.connection.host', ','.join(cluster_seeds)).set('spark.dynamicAllocation.maxExecutors', 16)
spark = SparkSession.builder.appName('example code').config(conf=conf).getOrCreate()
sc = spark.sparkContext

def cate_tuple(my_list):
    if my_list[0] != None:
        return [[a, my_list[1]] for a in my_list[0].split(',')]
    else:
        return [['Empty', my_list[1]]]
    
def att_time_split(my_list):
    (a, b)=my_list
    if a != None:
        return [[x, b] for x in a]
    else:
        return [['Empty', b]]

def dict_split(list_in):
    (a,b)=list_in
    try:
        match = re.search(r'([\w*]+),([\w*]+)', a)
        if match:
            return [[str(match.group(1)), str(match.group(2)), b]]
        else:
            match = re.search(r'([\w*]+),{(.*)}', a)
            if match:
                f=match.group(2).split(',')
                for a in f:
                    dict_value = re.search(r"('[\w*]+'):([\s\w*]+)", a)
                    return [['{}_{}'.format(match.group(1), dict_value.group(1)), str(dict_value.group(2)), b]]
    except:
        return [['','', b]]

def hours_split(my_list):
    (a, b)=my_list
    if a != '':
        temp = re.search(r"([\w*]+),(\d+):(\d+)-(\d+):(\d+)", a)
        return [[temp.group(1), int(temp.group(2))+int(temp.group(3))/60, int(temp.group(4))+int(temp.group(5))/60, b]]
    else:
        #Keep Empty as Empty
        return [['', '', '', my_list[1]]]

def main(keyspace, table):
    # create dataframes for order, lineitem and part
    df=spark.read.format("org.apache.spark.sql.cassandra").options(table='yelp_business', keyspace=keyspace).load()
    df.cache()
    # save as Allbusinesses table
    city_review = df.select('city', 'review_count').groupby('city').sum().orderBy('sum(review_count)', ascending=False).withColumnRenamed('sum(review_count)', 'ttl_reviews/City')
    # set up search grid around regions in Las Vegas
    # the final city we grab  Las Vegas, North Las Vegas, Henderson, Boulder City, las vegas
    lat, lon = 36.181271, -115.134132
    lat_range = 0.015
    lon_range = 0.015
    #DF = df.filter((df.city=='Las Vegas') | (df.city=='North Las Vegas')).select('latitude', 'longitude').orderBy('latitude')
    #DF.show()
    # save as champDF table
    DF = df.filter('latitude between {} and {}'.format(lat-lat_range,lat+lat_range))    .filter('longitude between {} and {}'.format(lon-lon_range,lon+lon_range)).cache()
    las_vegas_df = DF.select('city', 'review_count').groupby('city').sum().orderBy('sum(review_count)', ascending=False).withColumnRenamed('sum(review_count)', 'ttl_reviews/las_vegas')
    DF.cache()
    #Split categories
    cate_rdd = DF.select('categories', 'business_id').rdd.map(lambda x: x[:])
    # convert into a tuple of each category with one business_id
    categories = cate_rdd.flatMap(cate_tuple)
    # schemaString = "category business_id"
    # fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
    # schema = StructType(fields)
    observation_schema = types.StructType([
    types.StructField('category', types.StringType(), True),
    types.StructField('business_id', types.StringType(), True)
    ])
    # save as categoryBusiness table
    categoryDF = spark.createDataFrame(categories, observation_schema)
    categoryDF = categoryDF.withColumn("cate_count", lit(1))
    categoryDF.cache()
    categoryDF.createOrReplaceTempView("cate_restaurant")
    # Looking at all of the categories listed by frequency (each business can have multiple)
    df_cate_count = categoryDF.select('category', 'cate_count').groupby('category').sum().orderBy('sum(cate_count)', ascending=False).withColumnRenamed('sum(cate_count)', 'count')
    # filter business with categories as food or restaurants
    food_rest_df = spark.sql("SELECT count(*) AS num_category_restaurants FROM cate_restaurant WHERE lower(category) LIKE '%food%' OR lower(category) LIKE '%restaurant%'")
    # save table as foodbusiness
    food_rest_business = spark.sql("SELECT count(category) AS num_category_restaurants, business_id FROM cate_restaurant WHERE lower(category) LIKE '%food%' OR lower(category) LIKE '%restaurant%' GROUP BY business_id")
    # saved as businessFoodOnly table
    business_food_rest_df = DF.join(food_rest_business, "business_id", "right")
    business_food_rest_df.groupby('state').count()
    # convert each attributes with business_id
    attri = business_food_rest_df.select('attributes', 'business_id').rdd.map(lambda x: x[:])
    attri_restaurant = attri.flatMap(lambda x: att_time_split(x))
    schema_1 = types.StructType([
    types.StructField('attributes', types.StringType(), True),
    types.StructField('business_id', types.StringType(), True)])
    attri_df = spark.createDataFrame(attri_restaurant, schema_1)
    # Extract dictionaries from attributes column
    attri_df2 = attri_df.rdd.map(lambda x: x[:]).flatMap(lambda x: dict_split(x))
    schema_2 = types.StructType([
    types.StructField('attribute', types.StringType(), True),
    types.StructField('attribute_value', types.StringType(), True),
    types.StructField('business_id', types.StringType(), True)])
    attri_df3 = spark.createDataFrame(attri_df2, schema_2)
    # saved as attributeFinal
    hours_rdd = business_food_rest_df.select('hours','business_id').rdd.map(lambda x: x[:]).flatMap(lambda x: att_time_split(x))
    schema_3 = types.StructType([
    types.StructField('hours', types.StringType(), True),
    types.StructField('business_id', types.StringType(), True)])
    # saves as hoursBusiness table
    hours_df = spark.createDataFrame(hours_rdd, schema_3)
    hours_df1 = hours_df.groupby('hours').count().orderBy('count', ascending=False)
    # clean hours column
    hours_rdd1 = hours_df.rdd.map(lambda x: x[:]).flatMap(lambda x: hours_split(x))
    schema_5 = types.StructType([
    types.StructField('day', types.StringType(), True),
    types.StructField('opening_hour', types.FloatType(), True),
    types.StructField('closing_hour', types.FloatType(), True),
    types.StructField('business_id', types.StringType(), True)])
    # saved as openCloseBusiness table
    hours_df2 = spark.createDataFrame(hours_rdd1, schema_5)
    # most popular opening hours
    popular_hour_df = hours_df.groupby('day', 'opening_hour').count().orderBy('count', ascending=False)
    # Check-in dataset cleaning (saved as checkinAll table)
    df_checkin=spark.read.format("org.apache.spark.sql.cassandra").options(table='yelp_checkin', keyspace=keyspace).load()
    df_checkin.cache()
    checkin_rdd = df_checkin.select('time', 'business_id').rdd.map(lambda x: x[:]).flatMap(lambda x: att_time_split(x))
    schema_5 = types.StructType([
    types.StructField('checkin', types.StringType(), True),
    types.StructField('business_id', types.StringType(), True)])
    # saved as checkinCount table
    checkin_df = spark.createDataFrame(checkin_rdd, schema_5)
    # each business separated checkin hours count
    checkin_count_df = checkin_df.groupby('business_id').count().orderBy('business_id', ascending=False).withColumnRenamed('count', 'num_checkin')
    # merge num of checkins to business df (saved as cleanBusiness table) 190 restaurants in total
    cleanBusinessDF = business_food_rest_df.join(checkin_count_df, 'business_id', 'left').drop('hours', 'categories', 'attributes', 'type', 'is_open')
    df_review=spark.read.format("org.apache.spark.sql.cassandra").options(table='yelp_review', keyspace=keyspace).load()
    review_lasvegas_DF = df_review.join(cleanBusinessDF, 'business_id', 'right').drop(cleanBusinessDF['stars']).drop('address', 'latitude', 'longitude', 'postal_code', 'review_count', 'state', 'num_category_restaurants', 'num_checkin')
    if table == 'yelp_business_lasvegas':
        cleanBusinessDF.repartition(300).write.format("org.apache.spark.sql.cassandra").options(table=table, keyspace=keyspace).save()
    elif table == 'yelp_review_lasvegas':
        review_lasvegas_DF.repartition(300).write.format("org.apache.spark.sql.cassandra").options(table=table, keyspace=keyspace).save()
    
if __name__ == '__main__':
    keyspace = sys.argv[1]
    table = sys.argv[2]
    main(keyspace, table)

