# coding: utf-8

from flask import Flask, render_template, request, flash
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map, icons
from celery import Celery
from spark_celery import SparkCeleryApp, SparkCeleryTask, cache, main
import sys, re, os, operator
from kombu import Queue
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired
import paramiko
import getpass
from pyspark.sql import SparkSession, functions, types, Row
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.feature import VectorAssembler, IDF
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.ml.clustering import LDA, LocalLDAModel
#import sys
#sys.path.insert(0, "codes_for_etl/new_user/content_recom/TFIDF_LDA1")
#from codes_for_etl.new_user.content_recom.TFIDF_LDA1 import TFIDF_LDA1 as TF

#from codes_for_etl.existing_user.collab_recom.collaborative_filtering import collaborative_filtering as CF
#from spark_celery_main import return_app
#from etl_codes import collab_recom, collaborative_filtering, content_recom, content_userid, friend_user

BROKER_URL = "amqp://mhhau:yelp@localhost:5672/mhhau_vhost"
BACKEND_URL = "rpc://"

# app = flask_app
app = Flask(__name__, template_folder="templates", static_url_path="")

class Config(object):
    SECRET_KEY = os.environ.get("SECRET_KEY") or "this_is_a_secret_key_lol"

app.config.from_object(Config)
app.config.update(
    CELERY_BROKER_URL=BROKER_URL,
    CELERY_RESULT_BACKEND=BACKEND_URL
)
     
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

def sparkconfig_builder():
    from pyspark import SparkConf
    return SparkConf().setAppName('Yelp2018') \
        .set('spark.dynamicAllocation.enabled', 'true') \
        .set('spark.dynamicAllocation.schedulerBacklogTimeout', 1) \
        .set('spark.dynamicAllocation.minExecutors', 1) \
        .set('spark.dynamicAllocation.executorIdleTimeout', 20) \
        .set('spark.dynamicAllocation.cachedExecutorIdleTimeout', 60)


spark_celery_app = SparkCeleryApp(broker=BROKER_URL, backend=BACKEND_URL, sparkconf_builder=sparkconfig_builder)

priority = int(os.environ.get('CONSUMER_PRIORITY', '10'))
spark_celery_app.conf['CELERY_QUEUES'] = (
    Queue('celery', consumer_arguments={'x-priority': priority}),
)

GoogleMaps(
    app,
    key="AIzaSyBhUlLqKDSPctLcVDBUB3AraaRGTOQXDo4"
)

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/collaborative", methods=["GET", "POST"])
def collaborative():
    form = CollaborativeForm()
    if form.validate_on_submit():

        friend_user_rows = friend_user("WebFrontend/yelp_json/yelp_academic_dataset_user.json", "WebFrontend/yelp_json/yelp_academic_dataset_review.json", "WebFrontend/yelp_json/yelp_academic_dataset_business.json", form.userid.data, int(form.max_num.data))
        collab_recom_rows = collab_recom("WebFrontend/codes_for_etl/existing_user/collab_recom/als_model", "WebFrontend/codes_for_etl/existing_user/collab_recom/collab_business_Idnum", "WebFrontend/codes_for_etl/existing_user/collab_recom/collab_user_Idnum", form.userid.data, int(form.max_num.data))
        content_userid_rows = content_userid("WebFrontend/codes_for_etl/existing_user/content_userid/reviewLV_parquet", "WebFrontend/codes_for_etl/existing_user/content_userid/yelp_bus_parquet", "WebFrontend/codes_for_etl/existing_user/content_userid/content_userid", form.userid.data, int(form.max_num.data))

        marker_list = []

        for friend_user_row in friend_user_rows:
            marker_dict = {}
            marker_dict["icon"] = "//maps.google.com/mapfiles/ms/icons/blue-dot.png"
            marker_dict["lat"] = float(friend_user_row["latitude"])
            marker_dict["lng"] = float(friend_user_row["longitude"])
            marker_dict["infobox"] = "Your Friend Gives This 5 Stars!<br>" + \
                        "Your Friend Has Visited This Place " + str(friend_user_row["4_5_stars_count"]) + " Time(s)<br>" + \
                        friend_user_row["name"] + "<br>" + \
                        friend_user_row["categories"]
            marker_list.append(marker_dict)
        
        for collab_recom_row in collab_recom_rows:
            marker_dict = {}
            marker_dict["icon"] = "//maps.google.com/mapfiles/ms/icons/green-dot.png"
            marker_dict["lat"] = float(collab_recom_row["latitude"])
            marker_dict["lng"] = float(collab_recom_row["longitude"])
            marker_dict["infobox"] = "Collaborative Filtering Recommendation<br>" + \
                        "Prediction Score: " + str(collab_recom_row["prediction"]) + "<br>" + \
                        collab_recom_row["name"] + "<br>" + \
                        collab_recom_row["categories"]
            marker_list.append(marker_dict)

        for content_userid_row in content_userid_rows:
            marker_dict = {}
            marker_dict["icon"] = "//maps.google.com/mapfiles/ms/icons/purple-dot.png"
            marker_dict["lat"] = float(content_userid_row["latitude"])
            marker_dict["lng"] = float(content_userid_row["longitude"])
            marker_dict["infobox"] = "Content-based Similarity Based On Your Reviews<br>" + \
                        "Score: " + str(content_userid_row["score"]) + "<br>" + \
                        content_userid_row["name"] + "<br>" + \
                        content_userid_row["categories"]
            marker_list.append(marker_dict)

        #collab_recom_row, friend_user_row, content_userid_row = submit_collaborative(form.userid.data)
        # show top 10 of each category
        collaborative = Map(
            identifier="collaborative",
            varname="collaborative",
            style=(
                "height:90%;"
                "width:100%;"
                "top:10%;"
                "left:0;"
                "position:absolute;"
                "z-index:200;"
            ),
            lat=marker_list[0]["lat"],#36.114647,
            lng=marker_list[0]["lng"],#-115.172813,
            center_on_user_location=False,
            markers=marker_list
            # zoom="5"
        )
        return render_template(
            'collaborative.html',
            collaborative=collaborative,
            GOOGLEMAPS_KEY=request.args.get('apikey'), form=form)
    if request.method == "GET":
        collaborative = Map(
            identifier="collaborative",
            varname="collaborative",
            style=(
                "height:90%;"
                "width:100%;"
                "top:10%;"
                "left:0;"
                "position:absolute;"
                "z-index:200;"
            ),
            lat=49.267132,
            lng=-122.968941,
            center_on_user_location=True,
            markers=[
                {
                    'icon': '//maps.google.com/mapfiles/ms/icons/green-dot.png',
                    'lat': 49.277220, 
                    'lng': -122.914304,
                    'infobox': "Simon Fraser University Big Data Atrium"
                }
            ]
            # maptype = "TERRAIN",
            # zoom="5"
        )
        return render_template(
            'collaborative.html',
            collaborative=collaborative,
            GOOGLEMAPS_KEY=request.args.get('apikey'), form=form)


@app.route("/content_based", methods=["GET", "POST"])
def content_based():
    form = ContentBasedForm()
    if form.validate_on_submit():
        content_recom_rows = content_recom("WebFrontend/codes_for_etl/new_user/content_recom/review_json.json", \
                "WebFrontend/codes_for_etl/new_user/content_recom/yelp_bus_parquet", \
                "WebFrontend/codes_for_etl/new_user/content_recom/tfidf_model", \
                "WebFrontend/codes_for_etl/new_user/content_recom/tfidf_lda_model", \
                "WebFrontend/codes_for_etl/new_user/content_recom/yelp_review_sentiment", \
                "WebFrontend/codes_for_etl/new_user/content_recom/all_business_parquet", \
                form.query.data, int(form.max_num.data))
      

        marker_list = []
        count = 1 #start counting from 1st result

        for content_recom_row in content_recom_rows:
            marker_dict = {}
            if count < 27:
            
                pin_char = chr(ord('A') + count - 1)
                marker_dict["icon"] = "//www.google.com/mapfiles/marker" + pin_char + ".png"
            
            else:
            
                marker_dict["icon"] = "//maps.google.com/mapfiles/ms/icons/red-dot.png"
            
            marker_dict["lat"] = float(content_recom_row["latitude"])
            marker_dict["lng"] = float(content_recom_row["longitude"])
            marker_dict["infobox"] = "Sentiment Analysis Ranking: " + str(count) + "<br>" + \
                        "Content-based Similarity Score: " + str(content_recom_row["score"]) + "<br>" + \
                        "Sentiment Analysis Score: " + str(content_recom_row["sentiment_score"]) + "<br>" + \
                        content_recom_row["name"] + "<br>" + \
                        content_recom_row["categories"]
            count = count + 1
            marker_list.append(marker_dict)
        
        
        content_based = Map(
            identifier="content_based",
            varname="content_based",
            style=(
                "height:90%;"
                "width:100%;"
                "top:10%;"
                "left:0;"
                "position:absolute;"
                "z-index:200;"
            ),
            lat=content_recom_rows[0]["latitude"], #36.114647,
            lng=content_recom_rows[0]["longitude"], #-115.172813,
            center_on_user_location=False,
            markers=marker_list
            # zoom="5"
        )
        return render_template(
            'content_based.html',
            content_based=content_based,
            GOOGLEMAPS_KEY=request.args.get('apikey'), form=form)
    if request.method == "GET":
        content_based = Map(
            identifier="content_based",
            varname="content_based",
            style=(
                "height:90%;"
                "width:100%;"
                "top:10%;"
                "left:0;"
                "position:absolute;"
                "z-index:200;"
            ),
            lat=49.267132,
            lng=-122.968941,
            center_on_user_location=True,
            markers=[
                {
                    'icon': '//maps.google.com/mapfiles/ms/icons/green-dot.png',
                    'lat': 49.277220, 
                    'lng': -122.914304,
                    'infobox': "Simon Fraser University Big Data Atrium"
                }
            ]
            # maptype = "TERRAIN",
            # zoom="5"
        )
        return render_template(
            'content_based.html',
            content_based=content_based,
            GOOGLEMAPS_KEY=request.args.get('apikey'), form=form)

    

@app.route('/clickpost/', methods=['POST'])
def clickpost():
    # Now lat and lon can be accessed as:
    lat = request.form['lat']
    lng = request.form['lng']
    print(lat)
    print(lng)
    return "ok"

'''@app.route("/search_by_userid")
def search_by_userid():

    result = WordCount.get_data("/user/dca98/als_model", "/user/dca98/collab_business_Idnum", "/user/dca98/collab_user_Idnum", "gJhzYU76x7-U23jODY-MRQ")
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=result.id)}

@app.route('/status/<task_id>')
def taskstatus(task_id):
    result = WordCount.get_data.AsyncResult(task_id)
    if result.state == 'PENDING':
        response = {
            'state': result.state,
          
        }
    elif result.state != 'FAILURE':
        response = {
            'state': result.state,
           
        }
      
    else:
        # something went wrong in the background job
        response = {
            'state': result.state,
           
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)'''



@spark_celery_app.task(bind=True, base=SparkCeleryTask, name="collaborative.submit_collaborative")
def submit_collaborative(self, userID):
    content_userid_row = content_userid("WebFrontend/codes_for_etl/existing_user/content_userid/reviewLV_parquet", "WebFrontend/codes_for_etl/existing_user/content_userid/yelp_bus_parquet", "WebFrontend/codes_for_etl/existing_user/content_userid/content_userid", userID)
    friend_user_row = friend_user("WebFrontend/yelp_json/yelp_academic_dataset_user.json", "WebFrontend/yelp_json/yelp_academic_dataset_review.json", "WebFrontend/yelp_json/yelp_academic_dataset_business.json", userID)
    collab_recom_row = collab_recom("WebFrontend/codes_for_etl/existing_user/collab_recom/als_model", "WebFrontend/codes_for_etl/existing_user/collab_recom/collab_business_Idnum", "WebFrontend/codes_for_etl/existing_user/collab_recom/collab_user_Idnum", userID)
    return collab_recom_row, friend_user_row, content_userid_row

@spark_celery_app.task(bind=True, base=SparkCeleryTask, name="collaborative.collab_recom")

def collab_recom(self, model, input_1, input_2, u_id, sim_bus_limit=3):
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
    sparkconf_builder = spark_celery_app.sparkconf_builder 
    spark_conf = sparkconf_builder()
    sc = SparkContext.getOrCreate(conf=spark_conf)
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
    
    business_new_df = spark.read.parquet(input_1)
    business_new_df.cache()

    # saved above method into parquet to boost the code running time
    user_newid_df = spark.read.parquet(input_2)
    user_newid_df.cache()
    #print(type(u_id))
    #print("UID is ...")
    #print(u_id)

    bid = business_new_df.select('businessId').distinct()
    uid =  user_newid_df.filter(col('user_id') == u_id).select('userId').collect()[0][0]
    predDF = bid.withColumn("userId", lit(uid))
    alsn_model = ALSModel.load(model)
    predictions_ureq = alsn_model.transform(predDF)
    recommend_df = predictions_ureq.sort(predictions_ureq.prediction.desc()).select('businessId','prediction').limit(sim_bus_limit)
    result_df = business_new_df.join(recommend_df, 'businessId', 'right').select('business_id', 'name', 'categories', 'latitude', 'longitude', 'prediction')
    result_df = result_df.orderBy('prediction', ascending = False) #.show()
    result_df.show()
    result_df = result_df.collect()
    #result_json = result_df.to_json#name1 = result_df[0]["name"]
    return result_df #result_json #result_df, name1

@spark_celery_app.task(bind=True, base=SparkCeleryTask, name="collaborative.friend_user")

def friend_user(self, user_inputs, review_inputs, business_inputs, u_id, sim_bus_limit=3):
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
    sparkconf_builder = spark_celery_app.sparkconf_builder 
    spark_conf = sparkconf_builder()
    sc = SparkContext.getOrCreate(conf=spark_conf)
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()

    #print(type(user_inputs))
    #print(user_inputs)
    #print(type(review_inputs))
    #print(review_inputs)
    #print(type(business_inputs))
    #print(business_inputs)
    df_user = spark.read.json(user_inputs)
    df_review = spark.read.json(review_inputs)
    df_business = spark.read.json(business_inputs)
    #df_review.cache()
    df_1 = df_user.filter(df_user['friends']!='None').select('user_id','friends')
    df_2 = df_1.withColumn('friends', explode(split(col('friends'), ',')))
    #df_2.cache()
    review_df = df_review.withColumn("date", df_review["date"].cast(DateType()))
    df_2.createOrReplaceTempView('friends')
    df_user.createOrReplaceTempView('users')
    review_df.createOrReplaceTempView("reviews")
    print("achieved here")

    # Friends Recommendation
    # From the reviews, get top 50 restaurants that the user did not visit (reviewd), and that rated as 4 or 5 by his friends;
    
    df_friends_recom = spark.sql('select business_id, count(*) as 4_5_stars_count from reviews where user_id in \
                (select f.friends from friends f inner join users u on f.friends = u.user_id where f.user_id = "{}") \
                and stars >= 4 and business_id not in (select business_id from reviews where user_id = "{}") group by business_id \
                limit 50'.format(u_id, u_id))
    df_result = df_friends_recom.orderBy("4_5_stars_count", ascending = False).limit(sim_bus_limit)
    #df_result.cache()
    df = df_business.join(df_result, 'business_id', 'right').select('business_id', 'name', 'categories', 'latitude', 'longitude', '4_5_stars_count')
    # order by count(*) desc 
    #final_df.show()
    df_collected = df.collect()
    return df_collected


def CosineSim(vec1, vec2):
    return np.dot(vec1, vec2) / np.sqrt(np.dot(vec1, vec1)) / np.sqrt(np.dot(vec2, vec2))


@spark_celery_app.task(bind=True, base=SparkCeleryTask, name="collaborative.content_userid")

def content_userid(self, file1, file2, input_model, u_id, sim_bus_limit=3):
    
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
    sparkconf_builder = spark_celery_app.sparkconf_builder 
    spark_conf = sparkconf_builder()
    sc = SparkContext.getOrCreate(conf=spark_conf)
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
    
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
    #result.cache()
    # filter out those have been reviewd before by the user
    d = [i[0] for i in usr_rev_bus.collect()]
    df_1 = result.filter(~(col('business_id').isin(d))).select('business_id', 'score')
    #df_1= result.join(usr_rev_bus, 'business_id', 'left_outer').where(col("usr_rev_bus.business_id").isNull()).select([col('result.business_id'),col('result.score')])
    df_2 = df_1.orderBy("score", ascending = False).limit(sim_bus_limit)
    df_result = df_business.join(df_2, 'business_id', 'right').select('business_id', 'score', 'name', 'categories', 'latitude', 'longitude')
    df_result.show()
    df_result = df_result.collect()
    return df_result


@spark_celery_app.task(bind=True, base=SparkCeleryTask, name="collaborative.content_recom")

def content_recom(self, file1, file2, tfidf_model, tfidf_lda_model, sentiment_file, all_business_parquet, key_words, num_results=20):
    
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
    sparkconf_builder = spark_celery_app.sparkconf_builder 
    spark_conf = sparkconf_builder()
    sc = SparkContext.getOrCreate(conf=spark_conf)
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()

    data = spark.read.json(file1)
    df_business = spark.read.parquet(file2)
    
    df = data.select('business_id', 'text')
    review_rdd = df.rdd.map(tuple).reduceByKey(operator.add)
    review_df = spark.createDataFrame(review_rdd).withColumnRenamed('_1', 'business_id').withColumnRenamed('_2', 'text')

    tfidf_model = PipelineModel.load(tfidf_model)
    result_tfidf = tfidf_model.transform(review_df) 
    yelp = result_tfidf

    lda = LDA(k=15, maxIter=100)
    model = LocalLDAModel.load(tfidf_lda_model)
    # lda output column topicDistribution
    lda_df = model.transform(yelp)
    lda_vec = lda_df.select('business_id', 'topicDistribution').rdd.map(lambda x: (x[0], x[1])).collect()
    
    result = get_keywords_recoms(key_words, num_results, tfidf_model, model, lda_vec)
    df_sentiment = spark.read.json(sentiment_file)
    df_content_rest = df_sentiment.join(result, 'business_id', 'inner').orderBy("sentiment_score", ascending = False).limit(num_results)
    all_busi_df = spark.read.parquet(all_business_parquet)
    df_rest_result = all_busi_df.join(df_content_rest, 'business_id', 'right').select('business_id', 'sentiment_score', 'name', 'categories','score', 'latitude', 'longitude')
    df_rest_result.show()
    collected_df_rest_result = df_rest_result.collect()
    return collected_df_rest_result


class CollaborativeForm(FlaskForm):
    userid = StringField("Your User ID", validators=[DataRequired()])
    submit = SubmitField("Show Results")
    max_num = StringField("Max")

class ContentBasedForm(FlaskForm):
    query = StringField("Your Query", validators=[DataRequired()])
    submit = SubmitField("Search")
    max_num = StringField("Max")



#TFIDF_LDA1

def CosineSim(vec1, vec2):
    return np.dot(vec1, vec2) / np.sqrt(np.dot(vec1, vec1)) / np.sqrt(np.dot(vec2, vec2))

def get_keywords_recoms(key_words, sim_bus_limit, tfidf_model, model, lda_vec):
    
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
    sparkconf_builder = spark_celery_app.sparkconf_builder 
    spark_conf = sparkconf_builder()
    sc = SparkContext.getOrCreate(conf=spark_conf)
    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()

    input_words_df = sc.parallelize([('abc', key_words)]).toDF(['business_id', 'text'])
    
    # transform the the key words to vectors
    input_tfidf = tfidf_model.transform(input_words_df)
    lda_x = model.transform(input_tfidf)
    
    # choose lda vectors
    input_key_wordsvec = lda_x.select('topicDistribution').collect()[0][0]
    
    # get similarity
    sim_bus_byword_rdd = sc.parallelize((i[0], float(CosineSim(input_key_wordsvec, i[1]))) for i in lda_vec)

    sim_bus_byword_df = spark.createDataFrame(sim_bus_byword_rdd) \
         .withColumnRenamed('_1', 'business_id') \
         .withColumnRenamed('_2', 'score') \
         .orderBy("score", ascending = False)
    
    # return top 10 similar businesses
    a = sim_bus_byword_df.limit(sim_bus_limit)
    return a

if __name__ == "__main__":
    
    app.run(debug=True, use_reloader=True)
