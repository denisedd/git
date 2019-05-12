from pyspark.sql import SparkSession, functions, types, SQLContext
from pyspark import SparkConf, SparkContext
import re
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from tweet_parser.tweet import Tweet
import os


try:
    import json
except ImportError:
    import simplejson as json

bdenv_loc = 'file:///media/qianzhang/My Book'
bdata = os.path.join(bdenv_loc,'archiveteam-twitter-stream-2018-07')
download = 'file:///home/qianzhang/Downloads/archiveteam-twitter-stream-2018-09'
pycharm = 'file:///home/qianzhang/PycharmProjects/BigDataA2/FinalProject/archiveteam-twitter-stream-2018-06'

class EntityResolution:
    def __init__(self, dataFile1, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))

        self.stopWordsBC = sc.broadcast(self.stopWords).value

        self.df1 = sqlContext.read.json(dataFile1)
        OUTPUT_FORMAT = 'yyyy-MM-dd HH:mm:ss Z'
        INPUT_FORMAT = 'EEE MMM dd HH:mm:ss Z yyyy'
        self.df = self.df1.select([functions.date_format(functions.to_timestamp('created_at', INPUT_FORMAT), OUTPUT_FORMAT).alias('created_at'), 
        'text',  'favorite_count', 'retweet_count', 'reply_count', 'quote_count', 'lang', 'user' ])
        
       
        
        self.df = self.df.withColumn('categories',
                                    functions.when((self.df.text.like('%$AAPL%'))  |
                                        (self.df.text.like('%$ABBV%'))  |
                                        (self.df.text.like('%$ABT%'))  |
                                        (self.df.text.like('%$ACN%'))  |
                                        (self.df.text.like('%$AGN%'))  |
                                        (self.df.text.like('%$AIG%'))  |
                                        (self.df.text.like('%$ALL%'))  |
                                        (self.df.text.like('%$AMGN%'))  |
                                        (self.df.text.like('%$AMZN%'))  |
                                        (self.df.text.like('%$AXP%'))  |
                                        (self.df.text.like('%$BA%'))  |
                                        (self.df.text.like('%$BAC%'))  |
                                        (self.df.text.like('%$BIIB%'))  |
                                        (self.df.text.like('%$BK%'))  |
                                        (self.df.text.like('%$BKNG%'))  |
                                        (self.df.text.like('%$BLK%'))  |
                                        (self.df.text.like('%$BMY%'))  |
                                        (self.df.text.like('%$BRK.B%'))  |
                                        (self.df.text.like('%$C%'))  |
                                        (self.df.text.like('%$CAT%'))  |
                                        (self.df.text.like('%$CELG%'))  |
                                        (self.df.text.like('%$CHTR%'))  |
                                        (self.df.text.like('%$CL%'))  |
                                        (self.df.text.like('%$CMCSA%'))  |
                                        (self.df.text.like('%$COF%'))  |
                                        (self.df.text.like('%$COP%'))  |
                                        (self.df.text.like('%$COST%'))  |
                                        (self.df.text.like('%$CSCO%'))  |
                                        (self.df.text.like('%$CVS%'))  |
                                        (self.df.text.like('%$CVX%'))  |
                                        (self.df.text.like('%$DHR%'))  |
                                        (self.df.text.like('%$DIS%'))  |
                                        (self.df.text.like('%$DUK%'))  |
                                        (self.df.text.like('%$DWDP%'))  |
                                        (self.df.text.like('%$EMR%'))  |
                                        (self.df.text.like('%$EXC%'))  |
                                        (self.df.text.like('%$F%'))  |
                                        (self.df.text.like('%$FB%'))  |
                                        (self.df.text.like('%$FDX%'))  |
                                        (self.df.text.like('%$FOX%'))  |
                                        (self.df.text.like('%$FOXA%'))  |
                                        (self.df.text.like('%$GD%'))  |
                                        (self.df.text.like('%$GE%'))  |
                                        (self.df.text.like('%$GILD%'))  |
                                        (self.df.text.like('%$GM%'))  |
                                        (self.df.text.like('%$GOOG%'))  |
                                        (self.df.text.like('%$GOOGL%'))  |
                                        (self.df.text.like('%$GS%'))  |
                                        (self.df.text.like('%$HAL%'))  |
                                        (self.df.text.like('%$HD%'))  |
                                        (self.df.text.like('%$HON%'))  |
                                        (self.df.text.like('%$IBM%'))  |
                                        (self.df.text.like('%$INTC%'))  |
                                        (self.df.text.like('%$JNJ%'))  |
                                        (self.df.text.like('%$JPM%'))  |
                                        (self.df.text.like('%$KHC%'))  |
                                        (self.df.text.like('%$KMI%'))  |
                                        (self.df.text.like('%$KO%'))  |
                                        (self.df.text.like('%$LLY%'))  |
                                        (self.df.text.like('%$LMT%'))  |
                                        (self.df.text.like('%$LOW%'))  |
                                        (self.df.text.like('%$MA%'))  |
                                        (self.df.text.like('%$MCD%'))  |
                                        (self.df.text.like('%$MDLZ%'))  |
                                        (self.df.text.like('%$MDT%'))  |
                                        (self.df.text.like('%$MET%'))  |
                                        (self.df.text.like('%$MMM%'))  |
                                        (self.df.text.like('%$MO%'))  |
                                        (self.df.text.like('%$MRK%'))  |
                                        (self.df.text.like('%$MS%'))  |
                                        (self.df.text.like('%$MSFT%'))  |
                                        (self.df.text.like('%$NEE%'))  |
                                        (self.df.text.like('%$NFLX%'))  |
                                        (self.df.text.like('%$NKE%'))  |
                                        (self.df.text.like('%$NVDA%'))  |
                                        (self.df.text.like('%$ORCL%'))  |
                                        (self.df.text.like('%$OXY%'))  |
                                        (self.df.text.like('%$PEP%'))  |
                                        (self.df.text.like('%$PFE%'))  |
                                        (self.df.text.like('%$PG%'))  |
                                        (self.df.text.like('%$PM%'))  |
                                        (self.df.text.like('%$PYPL%'))  |
                                        (self.df.text.like('%$QCOM%'))  |
                                        (self.df.text.like('%$RTN%'))  |
                                        (self.df.text.like('%$SBUX%'))  |
                                        (self.df.text.like('%$SLB%'))  |
                                        (self.df.text.like('%$SO%'))  |
                                        (self.df.text.like('%$SPG%'))  |
                                        (self.df.text.like('%$T%'))  |
                                        (self.df.text.like('%$TGT%'))  |
                                        (self.df.text.like('%$TXN%'))  |
                                        (self.df.text.like('%$UNH%'))  |
                                        (self.df.text.like('%$UNP%'))  |
                                        (self.df.text.like('%$UPS%'))  |
                                        (self.df.text.like('%$USB%'))  |
                                        (self.df.text.like('%$UTX%'))  |
                                        (self.df.text.like('%$V%'))  |
                                        (self.df.text.like('%$VZ%'))  |
                                        (self.df.text.like('%$WBA%'))  |
                                        (self.df.text.like('%$WFC%'))  |
                                        (self.df.text.like('%$WMT%'))  |
                                        (self.df.text.like('%$XOM%')), 'General').when(
                                        (self.df.text.like('%$GOOG%')) | (self.df.text.like('%$GOOGL%')) ,     'Google').when(
                                        self.df.text.like('%$FB%'), 'Facebook').when(
                                        self.df.text.like('%$APPL%'), 'Apple').when(
                                        self.df.text.like('%$MSFT%'), 'Microsoft').when(
                                        self.df.text.like('%$BABA%'), 'Alibaba').when(
                                        self.df.text.like('%$CSCO%'), 'cisco').when(
                                        self.df.text.like('%$EBAY%'), 'ebay').when(
                                        self.df.text.like('%$NFLX%'), 'Netflix').when(
                                        self.df.text.like('%$TSLA%'), 'Tasla').when(
                                        self.df.text.like('%$AMZN%'), 'Amazon').otherwise('Others'))
                                        
                                        
        self.df = self.df.filter((self.df.categories != 'Others') & (self.df.lang == 'en'))

    def preprocessDF(self):
  
        self.df.write.json(os.path.join(bdenv_loc,'twitter_parse_2018_07.json'), mode='append')
        


if __name__ == "__main__":
    conf = SparkConf().setAppName('Twitter Parse')

    sc = SparkContext(conf=conf).getOrCreate()

    assert sc.version >= '2.3'  # make sure we have Spark 2.3+

    sqlContext = SQLContext(sc)

    spark = SparkSession.builder.appName('entity resolution').getOrCreate()

    spark.sparkContext.setLogLevel('WARN')


    sc.setLogLevel('WARN')
     
    
    for i in ['01','02','03','11','12','13','14','15','16','17','18','19']:
        er = EntityResolution(os.path.join(bdata,'2018 ({})/07/{}/*'.format(i,i)), "stopwords.txt")
         
        er.preprocessDF()
        
        
        
