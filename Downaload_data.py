import quandl
import bs4 as bs
import datetime as dt
import pandas as pd
import pickle
import requests
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os

quandl.ApiConfig.api_key = "Hs1z6Q-Lv5NoTrsByZ1o"
style.use('ggplot')


def sp500_symbols():
    response = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    symbols = []
    for row in table.findAll('tr')[1:]:
        symbol = row.findAll('td')[1].text
        symbols.append(symbol)
    with open("sp500symbols.pickle", "wb") as sp:
        pickle.dump(symbols, sp)
    
    return symbols


# sp500_symbols()
def getdata(reload_sp500=False):
    if reload_sp500:
        symbols = sp500_symbols()
    else:
        with open("sp500symbols.pickle", "rb") as sp:
            symbols = pickle.load(sp)
    if not os.path.exists('sp500stocks'):
        os.makedirs('sp500stocks')

    for symbol in symbols:
        
        if not os.path.exists('sp500stocks/{}.csv'.format(symbol)):
            df  = quandl.get('WIKI/{}'.format(symbol),start_date='2000-1-1', end_date='2017-12-29')
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('sp500stocks/{}.csv'.format(symbol))
        else:
            print('Already have {}'.format(symbol))


# getdata()
def final_data():
    with open("sp500symbols.pickle", "rb") as sp:
        symbols = pickle.load(sp)

    main_df = pd.DataFrame()

    for count,symbol in enumerate(symbols[:6]):
        df = pd.read_csv('sp500stocks/{}.csv'.format(symbol))
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj. Close': symbol}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume','Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 
        'Adj. Low','Adj. Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    main_df.to_csv('sp500_adjcloses.csv')

final_data()
