import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 目的変数を価格変動率にする　
# 3mのperiod削除

df = pd.read_csv('../dataset/bitstamp/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)

df['weekday'] = df['date'].dt.weekday

periods = [10, 30, 60, 60*3, 60*6, 60*24, 60*24*3, 60*24*7]

# 目的変数を価格変動率にする　
df['price_fluctuation_rate'] = df['Weighted_Price'].pct_change(1)

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

print(df.isnull().sum())
print(len(df))
df.to_csv('bitstamp_indices3.csv')


"""
pdf = PdfPages('btc3.pdf')

plt.figure(figsize=(60,15))
plt.plot(df["date"],df['Close'], label='price')
plt.ylabel("price")
plt.xlabel("datetime")
plt.grid(True)
plt.legend()
pdf.savefig()

plt.figure(figsize=(60,15))
plt.plot(df["date"],df['price_fluctuation_rate'])
plt.ylabel("price_fluctuation_rate")
plt.xlabel("datetime")
plt.grid(True)
pdf.savefig()

pdf.close()

"""
