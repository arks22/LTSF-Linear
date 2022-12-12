import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv('../dataset/bitstamp/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)

df['weekday'] = df['date'].dt.weekday

periods = [3, 10, 30, 60, 60*3, 60*6, 60*24, 60*24*3, 60*24*7]

# Moving Average
for period in periods:
    if period < 60:
        col = 'ma_{}m'.format(period)
    elif period < 24 * 60:
        col = 'ma_{}h'.format(round(period / 60))
    else:
        col = 'ma_{}d'.format(round(period / (60 * 24)))

    df[col] = df['Close'].rolling(period, min_periods=1).mean()

# 変化率
for period in periods:
    if period < 60:
        col = 'change_rate_{}m'.format(period)
    elif period < 24 * 60:
        col = 'change_rate_{}h'.format(round(period / 60))
    else:
        col = 'change_rate_{}d'.format(round(period / (60 * 24)))

    df[col] = df['Close'].pct_change(period)

# ボラティリティ
for period in periods:
    if period < 60:
        col = 'volatility_{}m'.format(period)
    elif period < 24 * 60:
        col = 'volatility_{}h'.format(round(period / 60))
    else:
        col = 'volatility_{}d'.format(round(period / (60 * 24)))

    df[col] = np.log(df["Close"]).diff().rolling(period, min_periods=1).std()

# 最小値
for period in periods:
    if period < 60:
        col = 'min_{}m'.format(period)
    elif period < 24 * 60:
        col = 'min_{}h'.format(round(period / 60))
    else:
        col = 'min_{}d'.format(round(period / (60 * 24)))

    df[col] = df['Close'].rolling(period, min_periods=1).min()

# 最大値
for period in periods:
    if period < 60:
        col = 'max_{}m'.format(period)
    elif period < 24 * 60:
        col = 'max_{}h'.format(round(period / 60))
    else:
        col = 'max_{}d'.format(round(period / (60 * 24)))

    df[col] = df['Close'].rolling(period, min_periods=1).max()

print(df)
df.to_csv('bitstamp_indices2.csv')


"""
pdf = PdfPages('btc.pdf')

plt.figure(figsize=(80, 20))
plt.plot(df["date"],df['Close'], label='price')
plt.plot(df["date"],df['min_100d'], label='min_100d')
plt.plot(df["date"],df['max_100d'], label='max_100d')
plt.ylabel("price")
plt.xlabel("datetime")
plt.grid(True)
plt.legend()
pdf.savefig()

plt.figure(figsize=(80, 20))
plt.plot(df["date"],df['volatility_1h'], label='volatility_1h')
plt.plot(df["date"],df['volatility_1d'], label='volatility_1d')
plt.plot(df["date"],df['volatility_25d'], label='volatility_25d')
plt.plot(df["date"],df['volatility_100d'], label='volatility_100d')
plt.ylabel("price")
plt.xlabel("datetime")
plt.grid(True)
plt.legend()
pdf.savefig()

pdf.close()

"""
