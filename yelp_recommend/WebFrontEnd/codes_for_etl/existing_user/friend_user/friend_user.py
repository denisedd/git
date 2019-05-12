import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row
from pyspark import SparkConf
from pyspark.sql.functions import *
from pyspark.sql.types import *
conf = SparkConf().set('spark.dynamicAllocation.maxExecutors', 16)
spark = SparkSession.builder.appName('example code').config(conf=conf).getOrCreate()
sc = spark.sparkContext

def main(user_inputs, review_inputs, business_inputs, u_id, sim_bus_limit=3):
    df_user = spark.read.json(user_inputs)
    df_review = spark.read.json(review_inputs)
    df_business = spark.read.json(business_inputs)
    df_review.cache()
    df_1 = df_user.filter(df_user['friends']!='None').select('user_id','friends')
    df_2 = df_1.withColumn('friends', explode(split(col('friends'), ',')))
    df_2.cache()
    review_df = df_review.withColumn("date", df_review["date"].cast(DateType()))
    df_2.createOrReplaceTempView('friends')
    df_user.createOrReplaceTempView('users')
    review_df.createOrReplaceTempView("reviews")

    # Friends Recommendation
    # From the reviews, get top 50 restaurants that the user did not visit (reviewd), and that rated as 4 or 5 by his friends;
    
    
    df_friends_recom = spark.sql('select business_id, count(*) as 4_5_stars_count from reviews where user_id in \
                (select f.friends from friends f inner join users u on f.friends = u.user_id where f.user_id = "{}") \
                and stars >= 4 and business_id not in (select business_id from reviews where user_id = "{}") group by business_id \
                limit 50'.format(u_id, u_id))
    df_result = df_friends_recom.orderBy("4_5_stars_count", ascending = False).limit(sim_bus_limit)
    df_result.cache()
    df = df_business.join(df_result, 'business_id', 'right').select('business_id', 'name', 'categories', 'latitude', 'longitude', '4_5_stars_count')
    # order by count(*) desc 
    df.show()
    

    
if __name__ == '__main__':
    user_inputs = sys.argv[1]
    review_inputs = sys.argv[2]
    business_inputs = sys.argv[3]
    u_id = sys.argv[4]
    main(user_inputs, review_inputs, business_inputs, u_id)