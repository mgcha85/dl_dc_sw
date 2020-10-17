import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Input, Dense, BatchNormalization, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, Bidirectional
from keras.optimizers import SGD
from keras.models import Model, Sequential, load_model

import sqlite3
import pandas as pd
import numpy as np
import socket
import os

import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc


class deeplearning:
    def __init__(self, root):
        self.root = root

    def set_candle_data(self, df):
        open = (1 + df['Open'].pct_change()).cumprod()
        open[0] = 1
        for col in ['High', 'Low', 'Close']:
            df[col] = (df[col] - df['Open']) / df['Open'] + open
        df['Open'] = open
        return df.dropna(how='any').drop(columns=['Volume'])

    def get_data(self):
        dirname = os.path.join(self.root, 'database/dl_data/지지')
        dfs = []
        for fname in [x for x in os.listdir(dirname) if '_1.xlsx' in x]:
            fpath = os.path.join(dirname, fname)
            df = pd.read_excel(fpath)
            df['code'] = df['code'].astype(str).str.zfill(6)
            df = df.set_index('code', drop=True)
            dfs.append(df)
        df = pd.concat(dfs)
        df = df[~df.index.duplicated(keep='first')]

        con = sqlite3.connect(os.path.join(self.root, 'database', 'day_candle.db'))
        trn_data = {'data': [], 'label': []}
        for code in df.index:
            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
            date = df.loc[code, 'date'].date()
            sustain = df.loc[code, 'important_price']

            df_day = pd.read_sql("SELECT * FROM (SELECT * FROM '{}' WHERE Date<='{}' ORDER BY Date DESC LIMIT 100) "
                                 "ORDER BY Date".format(code, date), con, index_col='Date')
            df_day.index = pd.to_datetime(df_day.index).date
            spos = 1 + (sustain - df_day['Open'][0]) / df_day['Open'][0]

            # xaxis = np.arange(df_day.shape[0])
            # candlestick_ohlc(ax1, zip(xaxis, df_day['Open'], df_day['High'], df_day['Low'], df_day['Close']), width=1, colorup='r', colordown='b')
            # ax1.plot(xaxis, [sustain] * len(xaxis))

            df_day = self.set_candle_data(df_day)[-100:]
            if df_day.shape[0] != 100:
                continue

            # candlestick_ohlc(ax2, zip(xaxis, df_day['Open'], df_day['High'], df_day['Low'], df_day['Close']), width=1, colorup='r', colordown='b')
            # ax2.plot(xaxis, [spos] * len(xaxis))
            # plt.savefig(os.path.join(self.root, 'database/dl_data/지지', '{}.png'.format(code)))
            # plt.close()

            trn_data['data'].append(df_day.values)
            trn_data['label'].append(spos)
        trn_data['data'] = np.transpose(np.dstack(trn_data['data']), (2, 0, 1))
        trn_data['label'] = np.array(trn_data['label'])
        return trn_data

    def model(self, candle__):
        candle = candle__
        for fsize in [32, 64, 128]:
            candle = Conv1D(filters=fsize, kernel_size=3, activation="relu")(candle)
            candle = Conv1D(filters=fsize, kernel_size=3, activation="relu")(candle)
            candle = BatchNormalization()(candle)
            candle = MaxPooling1D(pool_size=2, strides=2)(candle)
            candle = Dropout(rate=0.25)(candle)

        # candle = Bidirectional(LSTM(32))(candle)
        candle = Flatten()(candle)
        candle = Dense(1024, activation='relu')(candle)
        candle = Dropout(rate=0.5)(candle)
        candle = Dense(1)(candle)

        model = Model(inputs=candle__, outputs=candle)
        model.summary()
        return model

    def split_data(self, trn_data__, val=0.1, test=0.1):
        n = len(trn_data__['label'])
        vn = int(n * val)
        vt = int(n * test)

        ridx = np.arange(n)
        np.random.seed(0)
        np.random.shuffle(ridx)

        trn_data, val_data, test_data = {}, {}, {}
        for key in trn_data__.keys():
            trn_data__[key] = trn_data__[key][ridx]
            val_data[key] = trn_data__[key][:vn]
            test_data[key] = trn_data__[key][-vt:]
            trn_data[key] = trn_data__[key][vn:-vt]
        return trn_data, val_data, test_data

    def run(self):
        import matplotlib.pyplot as plt
        from mpl_finance import candlestick_ohlc

        trn_data = self.get_data()
        trn_data, val_data, test_data = self.split_data(trn_data)

        candle_in = Input(trn_data['data'].shape[1:])
        model = self.model(candle_in)

        model.compile(loss='mean_absolute_error', optimizer='adam')

        history = model.fit(trn_data['data'], trn_data['label'], validation_data=(val_data['data'], val_data['label']), epochs=300, verbose=0)

        df_hist = pd.DataFrame(history.history)
        print(df_hist)
        df_hist.plot()
        plt.show()

        model.save('dl_dc.model')

        model = load_model('dl_dc.model')
        result = model.predict(test_data['data']).flatten()

        for i in range(len(result)):
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
            xaxis = np.arange(test_data['data'].shape[1])
            candlestick_ohlc(ax1, zip(xaxis, test_data['data'][i, :, 0], test_data['data'][i, :, 1],
                                      test_data['data'][i, :, 2], test_data['data'][i, :, 3]), width=1, colorup='r', colordown='b')
            ax1.plot(xaxis, [test_data['label'][i]]*len(xaxis), label='real')
            ax1.plot(xaxis, [result[i]]*len(xaxis), label='pred')

            plt.legend()
            plt.savefig(os.path.join(self.root, 'database/dl_data/지지', str(i) + '.png'))
            plt.close()

        diff = np.abs((test_data['label'] - result))
        print('mean: {:0.4f}, std: {:0.4f}'.format(diff.mean(), diff.std()))


if __name__ == '__main__':
    hostname = socket.gethostname()
    print(hostname)
    if hostname == 'mingyu-Precision-Tower-7810':
        root = '/home/mingyu/Indepedent_research/KR'
    elif hostname == 'DESKTOP-DLOOJR6' or hostname == 'DESKTOP-1NLOLK4':
        root = 'D:/Independent_research/KR'
    elif hostname == 'mingyu-Inspiron-7559':
        root = '/media/mingyu/8AB4D7C8B4D7B4C3/Indepedent_research/KR'
    else:
        root = '/lustre/fs0/home/mcha/mgfin/KR'

    dl = deeplearning(root)
    dl.run()
