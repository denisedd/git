from pyspark.sql import SparkSession, functions, types, SQLContext
from pyspark import SparkConf, SparkContext
import re
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
import os
import unidecode 

bdenv_loc = 'file:///media/qianzhang/My Book/Angel'
#bdata = os.path.join(bdenv_loc,'twitter_parse_2018_01.json')


try:
    import json
except ImportError:
    import simplejson as json
    
class EntityResolution:
    def __init__(self, dataFile1, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))

        self.stopWordsBC = sc.broadcast(self.stopWords).value

        self.df1 = sqlContext.read.json(dataFile1)
        self.df1 = self.df1.withColumn('reformat1', functions.regexp_replace('text', 'RT', '{RT|RETWEET}'))
        
        self.df1 = self.df1.withColumn('reformat1', functions.regexp_replace('reformat1', 'http\S+', '{URL}'))
        self.df1 = self.df1.withColumn('reformat1', functions.regexp_replace('reformat1', '(\$)(\d+)', '{NUMBER|$2\}'))
        self.df1 = self.df1.withColumn('reformat1', functions.regexp_replace('reformat1', '(\d+)', '{NUMBER|$1\}'))
        self.df1 = self.df1.withColumn('reformat1', functions.regexp_replace('reformat1', '(\$)(\w+)', '{TICKER|$2\}'))
        self.df1 = self.df1.withColumn('reformat1', functions.regexp_replace('reformat1', '(\d+)(%)', '{PERCENT|$1}'))
        self.df1 = self.df1.withColumn('reformat1', functions.regexp_replace('reformat1', '(\#)(\w+)', '{HASH|$2\}'))  
        self.df1 = self.df1.withColumn('reformat1', functions.regexp_replace('reformat1', '(\@)(\w+)', '{USER|$2\}')) 
        self.df1 = self.df1.withColumn('reformat1', rem_udf('reformat1'))
        
        self.df1 = self.df1.withColumn('reformat2', functions.regexp_replace('reformat1', '\{\w+\|\w+\}', ''))
        self.df1 = self.df1.withColumn('reformat2', functions.regexp_replace('reformat2', '[^\sa-zA-Z0-9]', '')) 
        self.df1 = self.df1.drop('lang')      
        self.df2 = self.df1.select('user.favourites_count')
        self.df2.show(2)                          
        
        

    def preprocessDF(self):
        reTokenizer = RegexTokenizer(pattern=r'\W+', inputCol='reformat2', outputCol='tokenKey', toLowercase=True)
        df_token = reTokenizer.transform(self.df)
        remover = StopWordsRemover(inputCol='tokenKey', outputCol='tokens')
        remover.setStopWords(list(self.stopWords))
        df_token = remover.transform(df_token)
        
        df_token=df_token.select('categories','created_at','favorite_count','quote_count','reply_count','retweet_count','user.followers_count',
        'user.favourites_count','user.friends_count','tokens')
        
        self.df1.write.json(os.path.join(bdenv_loc,'twitter_parse_reformat_2018_01-06.json'), mode='append')
        df_token.write.json(os.path.join(bdenv_loc,'twitter_parse_tokens_2018_01-06.json'), mode='append')
        


if __name__ == "__main__":
    conf = SparkConf().setAppName('Twitter Parse')

    sc = SparkContext(conf=conf).getOrCreate()

    assert sc.version >= '2.3'  # make sure we have Spark 2.3+

    sqlContext = SQLContext(sc)

    spark = SparkSession.builder.appName('Twitter Parse').getOrCreate()

    spark.sparkContext.setLogLevel('WARN')


    sc.setLogLevel('WARN')
    
    def remove_diacritics(s):
        return unidecode.unidecode(s)

    rem_udf = functions.udf(remove_diacritics, functions.StringType())

    
    #er = EntityResolution(os.path.join(bdenv_loc,'twitter_parse_2018_01.json'), "stopwords.txt")               
    #er.preprocessDF()
    for i in range(1,7):
        er = EntityResolution(os.path.join(bdenv_loc,'twitter_parse_2018_0{}.json'.format(i)), "stopwords.txt")               
        er.preprocessDF()
    
    
    
