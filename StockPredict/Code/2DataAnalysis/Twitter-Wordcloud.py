# -*- coding: utf-8 -*-

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np
import urllib
import requests
import matplotlib.pyplot as plt

import pandas as pd
#from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from itertools import chain
import json
import os

mask = np.array(Image.open("gdrive/My Drive/BigDataFinalReport/img/images.jpeg"))
df_total = pd.DataFrame()
directory = 'gdrive/My Drive/Programming in Big Data 2/data/wordcould.json/wordcloud.json'

print(directory)

with open(directory, 'r') as f:
    df = pd.read_json(f, lines=True)  
     
f.close()      

print(len(df))

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np
import urllib
import requests
import matplotlib.pyplot as plt

import pandas as pd
#from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from itertools import chain
import json
import os

mask = np.array(Image.open("gdrive/My Drive/BigDataFinalReport/img/images.jpeg"))
df_total = pd.DataFrame()
directory = 'gdrive/My Drive/Programming in Big Data 2/data/wordcould_2018.json/nltk_2018.json'

print(directory)

word_list = []

with open(directory, 'r') as f:
    df = pd.read_json(f, lines=True)  
    df_total = df_total.append(df) 
f.close()      
print(df_total.head(2))      
   
word_list = []
print(len(df_total))
for i in range(len(df_total)):
  word_list = word_list + df_total.iloc[i]['tokens']

 
words = ' '.join(word_list)
 
stopwords = set(STOPWORDS)
stopwords.update(["food", "now", "one", "rt", "$","#","URL","aapl", "{", "}","AT_USER","https","http"])
word_cloud = WordCloud(width = 412, height = 412, background_color='white', max_words=1000, stopwords=stopwords, mask=mask).generate(words)
plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
             
plt.axis('off')
plt.tight_layout(pad=0)
             
word_cloud.to_file("gdrive/My Drive/Programming in Big Data 2/data/tweet/img/aapl_2018.jpeg")


def transform_format(val):
    if val == 0:
        return 255
    else:
        return val


