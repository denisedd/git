
# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# Import the tweepy library
import tweepy
import csv
import datetime

# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN = 'XXX'
ACCESS_SECRET = 'XXXX'
CONSUMER_KEY = 'XXXX'
CONSUMER_SECRET = 'XXXX'

# Setup tweepy to authenticate with Twitter credentials:

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

# Create the api to connect to twitter with your creadentials
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
#---------------------------------------------------------------------------------------------------------------------
# wait_on_rate_limit= True;  will make the api to automatically wait for rate limits to replenish
# wait_on_rate_limit_notify= Ture;  will make the api  to print a notification when Tweepyis waiting for rate limits to replenish
#---------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------
# The following loop will print most recent statuses, including retweets, posted by the authenticating user and that user’s friends. 
# This is the equivalent of /timeline/home on the Web.
#---------------------------------------------------------------------------------------------------------------------

startDate = datetime.datetime(2019, 1, 1, 0, 0, 0)
endDate =   datetime.datetime(2019, 3, 13, 23, 15, 53)

tweets = tweepy.Cursor(api.search, q='$GOOG OR $APPL OR $FB OR $MSFT OR $BABA OR $CSCO OR $EBAY OR $NFLX OR $TSLA OR AMZN', since= '2019-01-01', until='2019-03-13', lang='en').items()

with open('data_finance.csv','a', newline='') as outfile:
    writer = csv.writer(outfile)
    for tweet in tweets:
        line = [tweet.text.encode('utf-8'), tweet.created_at, tweet.retweet_count, tweet.favorite_count]
        writer.writerow(line)
        #json.dump(tweet._json, outfile)
        #print(tweet._json)
    
#for status in tweepy.Cursor(api.home_timeline).items(200):
#	print(status._json)
	
#---------------------------------------------------------------------------------------------------------------------
# Twitter API development use pagination for Iterating through timelines, user lists, direct messages, etc. 
# To help make pagination easier and Tweepy has the Cursor object.
#---------------------------------------------------------------------------------------------------------------------



