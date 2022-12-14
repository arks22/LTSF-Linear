import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 目的変数を価格のlogスケールにする
# 長期指標をやっぱり追加
# MA -> EMA
# MACD

df = pd.read_csv('../dataset/bitstamp/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)
df['weekday'] = df['date'].dt.weekday

#対数変換
df['High']  = np.log10(df['High'] + 1)
df['Low']   = np.log10(df['Low'] + 1)
df['Open']  = np.log10(df['Open'] + 1)
df['Close'] = np.log10(df['Close'] + 1)
df['Weighted_Price'] = np.log10(df['Weighted_Price'] + 1)
df['Volume_(Currency)'] = np.log10(df['Volume_(Currency)'] + 1)

periods = [7, 25, 100, 60*6, 60*24, 60*24*7]

# Exponential Moving Average
for period in periods:
    col = 'EMA_{}m'.format(period)
    df[col] = df['Close'].ewm(alpha=(2/(1+period)), adjust=False).mean()

ema_periods = [9, 12, 26]
for period in ema_periods:
    col = 'EMA_{}m'.format(period)
    df[col] = df['Close'].ewm(alpha=(2/(1+period)), adjust=False).mean()

df['MACD'] = df['EMA_9m'] - (df['EMA_12m'] - df['EMA_26m'])

# remove metrics for MACD
for period in ema_periods:
    df = df.drop('EMA_{}m'.format(period), axis=1)

# 変化率
for period in periods:
    col = 'Change_rate_{}m'.format(period)
    df[col] = df['Close'].pct_change(period)

# ボラティリティ
for period in periods:
    col = 'Volatility_{}m'.format(period)
    df[col] = np.log(df["Close"]).diff().rolling(period, min_periods=1).std()

# 最小値
for period in periods:
    col = 'Low_{}m'.format(period)
    df[col] = df['Close'].rolling(period, min_periods=1).min()

# 最大値
for period in periods:
    col = 'High_{}m'.format(period)
    df[col] = df['Close'].rolling(period, min_periods=1).max()


df.to_csv('../dataset/bitstamp/bitstamp_indices6.csv')

print(df.isnull().sum())
print(len(df))


"""
pdf = PdfPages('btc5_log.pdf')

plt.figure(figsize=(60,15))
plt.title('Price')
plt.plot(df["date"],df['Close'], label='price')
plt.xlabel("datetime")
plt.grid(True)
plt.legend()
pdf.savefig()

plt.figure(figsize=(60,15))
plt.title('Volume(Currency)')
plt.plot(df["date"],df['Volume_(Currency)'])
plt.xlabel("datetime")
plt.grid(True)
pdf.savefig()

plt.figure(figsize=(60,15))
plt.title('Volume(BTC)')
plt.plot(df["date"],df['Volume_(BTC)'])
plt.xlabel("datetime")
plt.grid(True)
pdf.savefig()

plt.figure(figsize=(60,15))

pdf.close()
"""
