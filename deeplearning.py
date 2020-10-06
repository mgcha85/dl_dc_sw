import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Input, Dense, BatchNormalization, Flatten, Conv1D, MaxPooling1D, Dropout
from keras.optimizers import SGD
from keras.models import Model, Sequential, load_model

import sqlite3
import pandas as pd
import numpy as np
import socket
import os


class deeplearning:
    def __init__(self, root):
        self.root = root

    def set_candle_data(self, df):
        df['ma20'] = df['Close'].rolling(20).mean()
        # df['AVG_VOL'] = df['Volume'].rolling(20).mean()
        # df['VRATE'] = df['Volume'] / df['AVG_VOL']

        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = (1 + df[col].pct_change()).cumprod()
        df.iloc[0, :] = 1
        return df.dropna(how='any').drop(columns=['Volume'])

    def get_data(self):
        fpath = os.path.join(self.root, 'database/dl_data/지지', '2020-10-06.xlsx')
        con = sqlite3.connect(os.path.join(self.root, 'database', 'day_candle.db'))

        df = pd.read_excel(fpath)
        df['code'] = df['code'].astype(str).str.zfill(6)
        df = df.set_index('code', drop=True)

        trn_data = {'data': [], 'label': []}
        for code in df.index:
            date = df.loc[code, 'date']
            sustain = df.loc[code, 'important_price']

            df_day = pd.read_sql("SELECT * FROM (SELECT * FROM '{}' WHERE Date<='{}' ORDER BY Date DESC LIMIT 120) "
                                 "ORDER BY Date".format(code, date), con, index_col='Date')
            df_day.index = pd.to_datetime(df_day.index).date
            spos = 1 + (sustain - df_day['Open'][0]) / df_day['Open'][0]

            df_day = self.set_candle_data(df_day)[-100:]
            if df_day.shape[0] != 100:
                continue

            trn_data['data'].append(df_day.values)
            trn_data['label'].append(spos)
        trn_data['data'] = np.transpose(np.dstack(trn_data['data']), (2, 0, 1))
        trn_data['label'] = np.array(trn_data['label'])
        return trn_data

    def model(self, candle__):
        candle = candle__
        for fsize in [128, 256, 512, 1024]:
            candle = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(candle)
            candle = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(candle)
            candle = BatchNormalization()(candle)
            candle = MaxPooling1D(pool_size=2, strides=2)(candle)

        candle = Flatten()(candle)

        s = 4096
        for _ in range(2):
            candle = Dense(s, activation='relu')(candle)
            candle = Dropout(rate=0.5)(candle)
        candle = Dense(1, activation='sigmoid')(candle)

        model = Model(inputs=candle__, outputs=candle)
        model.summary()
        return model

    def split_data(self, trn_data__, val=0.1, test=0.1):
        n = len(trn_data__['label'])
        vn = int(n * val)
        vt = int(n * test)

        trn_data, val_data, test_data = {}, {}, {}
        for key in trn_data__.keys():
            np.random.shuffle(trn_data__[key])
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
        model.compile(loss='binary_crossentropy', optimizer='adam')
        history = model.fit(trn_data['data'], trn_data['label'], validation_data=(val_data['data'], val_data['label']), epochs=200, verbose=1)

        df_hist = pd.DataFrame(history.history)
        print(df_hist)
        model.save('dl_dc.model')

        model = load_model('dl_dc.model')
        result = model.predict(test_data['data']).argmax(axis=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        xaxis = np.arange(test_data['data'].shape[1])
        candlestick_ohlc(ax1, zip(xaxis, test_data['data'][0, :, 0], test_data['data'][0, :, 1],
                                  test_data['data'][0, :, 2], test_data['data'][0, :, 3]), width=1, colorup='r', colordown='b')
        ax1.plot(xaxis, result[0])
        plt.show()
        diff = (test_data['label'] - result) / result
        print('mean: {:04f}, std: {:04f}'.format(diff.mean(), diff.std()))


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
