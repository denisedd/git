"""
Scrap stock symbol for all nasdaq company.
"""
from bs4 import BeautifulSoup
import os
from datetime import datetime
import re
import json
import requests

# symbol_starts = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# for i in symbol_starts:
#     response = requests.get('http://eoddata.com/stocklist/NASDAQ/'+ str(i) + '.htm')
#     if not os.path.exists('AllSymbolScrapper'):
#         os.makedirs('AllSymbolScrapper')
#     with open('AllSymbolScrapper'+i+'.htm','w') as file:
#         file.write(response.text)

res = {}
for file in os.listdir('../../Data/AllSymbolScrapper'):
    if file.endswith('.htm'):
        with open(os.path.join('../../Data/AllSymbolScrapper', file)) as f:
            html = f.read()
            soup = BeautifulSoup(html, 'html.parser')
            tr_all = soup.find_all('tr', attrs={"class": "ro"}) + soup.find_all('tr', attrs={"class": "re"})
            for tr in tr_all:
                code = list(tr.children)[0].string
                name = list(tr.children)[1].string
                res[code] = name

# with open(os.path.join('../Data/AllSymbolScrapper', 'symbols.json'), 'a') as f:
#     json.dump(res, f)
print(len(res))
