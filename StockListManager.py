import os
import pandas as pd
import numpy as np
import datetime
import urllib
from bs4 import BeautifulSoup
import sqlite3


class StockListManager:
    def __init__(self, root):
        self.root = root

    def load_stockList(self):
        fpath = os.path.join(self.root, 'database', 'stockList.txt')
        with open(fpath, 'rt') as f:
            stockList = f.read()
        return np.array(stockList.split('\n'))

    def load_dateList(self, label='screen_table_3min', format='string'):
        fpath = os.path.join(self.root, 'database', 'dateList.txt')
        df = pd.read_csv(fpath, index_col=0, header=None)
        dateList = df.loc[label, 1].split(';')
        if format == 'datetime':
            dateList = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in dateList]
        return np.array(dateList)
