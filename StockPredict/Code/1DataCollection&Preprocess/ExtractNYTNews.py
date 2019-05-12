"""
Extract New York Times news data from HTML files.
"""
from bs4 import BeautifulSoup
import os
from datetime import datetime
import re
import json

root = os.path.join('../../Data/HTMLNews', 'NYT')
years = ['2018'] #'2016', '2017','2018'
months = ['08','09','10','11','12'] #'01','02','03','04','05','06','07','08','09','10','11','12'
generals = ['economy', 'Finance', 'Stocks', 'Stocks Market']

for year in years:
    for month in months:
        for cat in generals:
            path = os.path.join(root, year, month, cat)
            if os.path.exists(path):
                path_list = os.listdir(path)
                print(len(path_list), 'files in the directory:', path)
                for web in path_list:
                    if web.endswith('.html'):
                        with open(os.path.join(path, web)) as file:
                            item = {'Headline': None, 'Date': None, 'News': None}
                            html = file.read()
                            soup = BeautifulSoup(html, 'html.parser')

                            # Find headline of HTML news
                            headline = soup.find('h1', attrs={"itemprop": "headline"})
                            if headline:
                                item['Headline'] = headline.string

                            # Find time of HTML news
                            time = soup.find('time')
                            if time:
                                try:
                                    dt = time['datetime'].split('T')[0]
                                    if re.match('\d+', dt):
                                        dt = datetime.strptime(dt, "%Y-%m-%d").date()
                                        # print(type(dt))
                                        item['Date'] = str(dt)
                                except:
                                    print('error on:', web)

                            # Find content of HTML news
                            contents = soup.find_all('p')
                            content = ''
                            for c in contents:
                                if c.string:
                                    content += c.string
                            # print(type(contents))
                            item['News'] = content

                            # Store item as json
                            dict = os.path.join('../../Data/JsonNews', 'NYT', year, month)
                            if not os.path.exists(dict):
                                os.makedirs(dict)

                            with open(os.path.join(dict, str(cat) + '.json'), 'a') as f:
                                # f.write(json.dumps(item).encode("utf-8"))
                                json.dump(item, f)
                                f.write('\n')
