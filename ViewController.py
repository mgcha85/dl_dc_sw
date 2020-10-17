from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, QDate
import os
import json
import datetime
import numpy as np
import pandas as pd
import sqlite3
import math
from StockListManager import StockListManager
from Database import Database

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc

form_class = uic.loadUiType("dl_dc_sw.ui")[0]


class ViewController(QMainWindow, form_class):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        with open('parameter.json') as f:
            self.user_param = json.load(f)

        self.slm = StockListManager(self.user_param['path']['root'])
        self.dateList = self.slm.load_dateList(format='datetime')

        self.connect_btn()
        self.symbolList = None
        self.current_page = 0
        self.current_code = ''

        today = datetime.date.today()
        self.dateEdit.setDate(QDate(today.year, today.month, today.day))
        self.current_date = datetime.date(*self.dateEdit.date().getDate())
        self.didx = len(self.dateList) - 1
        self.last_date = self.dateList[self.didx]
        self.limit = 100
        self.lineEdit.setText(str(self.limit))

        # for chart
        self.figure = plt.figure()
        gs = self.figure.add_gridspec(6, 1)

        self.canvas = FigureCanvas(self.figure)
        self.ax = [self.figure.add_subplot(gs[:2]),
                   self.figure.add_subplot(gs[2]),
                   self.figure.add_subplot(gs[3:5]),
                   self.figure.add_subplot(gs[5])]

        toolbar = NavigationToolbar(self.canvas, self)

        self.gridLayout_2.addWidget(self.canvas, 2, 0, 1, 2)
        self.gridLayout_2.addWidget(toolbar, 1, 0, 1, 2)
        self.figure.tight_layout()

        self.report = self.xaxis = self.df = self.df_int = self.xtickLabel = self.xaxis_int = self.xtickLabel_int = None
        self.clicked = False
        self.n = 0
        self.num_task = 0
        self.isSet = False

        self.radioButton.toggled.connect(self.onClicked)
        self.radioButton_2.toggled.connect(self.onClicked)
        self.lineEdit_2.returnPressed.connect(self.on_returned)

        self.type = '돌파'
        self.colors = {'돌파': 'r', '지지': 'g'}
        fpath = os.path.join(self.user_param['path']['root'], 'database', 'day_candle.db')
        self.con = sqlite3.connect(fpath)

        fpath = os.path.join(self.user_param['path']['root'], 'database', 'intra_day_3min_{}-{:02d}.db'.format(today.year, today.month))
        self.db_int = Database(self.user_param['path']['root'], 'intra_day_3min')
        self.con_int = sqlite3.connect(fpath)

    def connect_btn(self):
        btn_map = {self.pushButton: self.go_prev, self.pushButton_2: self.go_next, self.pushButton_3: self.save_task,
                   self.pushButton_4: self.load_task}

        for btn, func in btn_map.items():
            btn.clicked.connect(func)

    def load_task(self):
        self.current_date = datetime.date(*self.dateEdit.date().getDate())
        fpath = os.path.join(self.user_param['path']['root'], 'database/screen_tables', 'swing_table_{}.db'.format(self.current_date.year))
        con = sqlite3.connect(fpath)

        self.didx = np.where(self.dateList == self.current_date)[0][0]
        self.last_date = self.dateList[self.didx]

        rep_path = os.path.join(self.user_param['path']['root'], 'database/dl_data', self.type, str(self.current_date) + '.xlsx')

        df = pd.read_sql("SELECT * FROM '{}'".format(self.last_date), con, index_col='Symbol')
        self.n = df.shape[0]
        self.symbolList = list(df.index)
        if len(self.symbolList) > 0:
            if os.path.exists(rep_path):
                self.report = pd.read_excel(rep_path)
                if self.report.empty:
                    self.report = pd.DataFrame(index=self.symbolList, columns=['date', 'important_price'])
                    self.report.index.name = 'code'
                else:
                    self.report['code'] = self.report['code'].astype(str).str.zfill(6)
                    self.report = self.report.set_index('code', drop=True)
                    self.num_task = self.report.shape[0]
                    self.current_page = self.symbolList.index(self.report.index[-1]) + 1
            else:
                self.report = pd.DataFrame(index=self.symbolList, columns=['date', 'important_price'])
                self.report.index.name = 'code'
            self.current_code = self.symbolList[self.current_page]
            self.show_candle()
        else:
            print('해당 날짜에 해당 주식이 없음')

        self.label_2.setText('{} / 100'.format(self.num_task+1))
        # connect event handler for figures
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_press_event', self.on_click)

    # radio button
    def onClicked(self):
        radioButton = self.sender()
        if radioButton.text() == '돌파':
            self.type = '돌파'
        else:
            self.type = '지지'

    # for graph
    def on_click(self, event):
        if self.clicked is False and self.report is not None:
            x, y = event.xdata, event.ydata
            self.report.loc[self.current_code, 'date'] = self.current_date
            self.report.loc[self.current_code, 'important_price'] = y
            self.clicked = True
            self.isSet = True
        else:
            self.ax[0].clear()
            self.ax[2].clear()
            self.report.loc[self.current_code, 'important_price'] = float('nan')
            self.draw_candle([0, 2])

            self.canvas.draw()
            self.clicked = False

    # for stock name
    def on_returned(self):
        self.current_code = self.lineEdit_2.text()
        idx = self.symbolList.index(self.current_code)
        self.current_page = idx
        self.show_candle()

    def on_motion(self, event):
        if self.clicked is False and event.inaxes and self.xaxis is not None:
            if event.inaxes.rowNum == 0 or event.inaxes.rowNum == 3:
                x, y = event.xdata, event.ydata
                self.ax[0].clear()
                self.ax[2].clear()

                self.draw_candle([0, 2])
                self.ax[0].plot(self.xaxis, [y]*len(self.xaxis), color=self.colors[self.type])
                self.ax[2].plot(self.xaxis_int, [y]*len(self.xaxis_int), color=self.colors[self.type])
                self.canvas.draw()

    def show_candle(self):
        for ax in self.ax:
            ax.clear()
        rows = range(len(self.ax))

        self.df = pd.read_sql("SELECT * FROM (SELECT * FROM '{}' WHERE Date<='{}' ORDER BY Date DESC LIMIT {}) "
                    "ORDER BY Date".format(self.current_code, self.last_date, self.limit+20), self.con, index_col='Date')
        if self.df is None or self.df.empty:
            return
        self.df['MA_20'] = self.df['Close'].rolling(20).mean()
        self.df.dropna(how='any', inplace=True)

        pdate = self.dateList[self.didx-2]
        if Database.checkTableExists(self.con_int, self.current_code):
            try:
                self.df_int = self.db_int.get_data_by_datetime(self.current_code, pdate, self.last_date, '09:00:00', '15:20:00')
                if self.df_int is None or self.df_int.empty:
                    return
                self.df_int['MA_20'] = self.df_int['Close'].rolling(20).mean()
                self.df_int = self.df_int[self.df_int.index.date == self.last_date]
            except Exception as e:
                print(e)

        self.draw_candle(rows)
        self.canvas.draw()

    def draw_candle(self, rows):
        if self.df is None or self.df_int is None:
            return

        if 0 in rows:
            self.xaxis = np.arange(self.df.shape[0])
            candlestick_ohlc(self.ax[0], zip(self.xaxis, self.df['Open'], self.df['High'], self.df['Low'], self.df['Close']),
                             width=1, colorup='r', colordown='b')
            self.xtickLabel = [str(x) for x in self.df.index]
            self.ax[0].plot(self.xaxis, self.df['MA_20'], color='y')
            self.ax[0].set_xticks(self.xaxis[::4])
            self.ax[0].set_xticklabels(self.xtickLabel[::4], rotation=30, fontsize=6)
            self.ax[0].set_ylabel('Price')

        if 1 in rows:
            if self.df is None or self.df.empty:
                return
            self.ax[1].bar(self.xaxis, self.df['Volume'])
            self.ax[1].set_ylabel('Volume')

        if 2 in rows:
            if self.df_int is None or self.df_int.empty:
                return

            self.xaxis_int = np.arange(self.df_int.shape[0])
            candlestick_ohlc(self.ax[2], zip(self.xaxis_int, self.df_int['Open'], self.df_int['High'], self.df_int['Low'], self.df_int['Close']), width=1, colorup='r', colordown='b')

            self.xtickLabel_int = [str(x) for x in self.df_int.index.time]
            self.ax[2].plot(self.xaxis_int, self.df_int['MA_20'], color='y')
            self.ax[2].set_xticks(self.xaxis_int[::4])
            self.ax[2].set_xticklabels(self.xtickLabel_int[::4], rotation=30, fontsize=6)
            self.ax[2].set_ylabel('Price')

        if self.current_code in self.report.index and math.isnan(self.report.loc[self.current_code, 'important_price']) is False:
            self.ax[0].plot(self.xaxis, [self.report.loc[self.current_code, 'important_price']] * len(self.xaxis), color=self.colors[self.type])
            self.ax[2].plot(self.xaxis_int, [self.report.loc[self.current_code, 'important_price']] * len(self.xaxis_int), color=self.colors[self.type])
            self.clicked = True

        if 3 in rows:
            self.ax[3].bar(self.xaxis_int, self.df_int['Volume'])
            self.ax[3].set_ylabel('Volume')

        self.lineEdit_2.setText('{}'.format(self.current_code))
        self.figure.tight_layout()

    def save_task(self):
        self.report.dropna(how='any').to_excel(os.path.join(self.user_param['path']['root'], 'database/dl_data', self.type, str(self.current_date) + '.xlsx'))
        if self.isSet:
            self.num_task += 1

    def go_next(self):
        if len(self.symbolList) == 0:
            print('오늘은 작업은 없네요')
            return

        if self.current_page + 1 == self.n:
            return

        self.save_task()
        self.current_page += 1
        self.current_code = self.symbolList[self.current_page]
        self.show_candle()

        self.clicked = False
        self.isSet = False
        self.label_2.setText('{} / 100'.format(self.num_task + 1))

    def go_prev(self):
        if len(self.symbolList) == 0:
            print('오늘은 작업은 없네요')
            return
        if self.current_page == 0:
            return

        self.current_page -= 1
        self.current_code = self.symbolList[self.current_page]
        self.show_candle()

        self.clicked = False
        self.label_2.setText('{} / 100'.format(self.num_task+1))
