import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv('../dataset/bitstamp/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)

df['weekday'] = df['date'].dt.weekday

#  1時間、6時間、1日、7日、25日、100日
# Moving Average
periods = [1, 6, 24, 24*7, 24*25, 24*100]
for period in periods:
    if period < 24:
        col = 'ma_{}h'.format(period)
    else:
        col = 'ma_{}d'.format(round(period / 24))
    period *= 60
    df[col] = df['Close'].rolling(period, min_periods=1).mean()

# 変化率
periods = [1, 6, 24, 24*7, 24*25, 24*100]
for period in periods:
    if period < 24:
        col = 'change_rate_{}h'.format(period)
    else:
        col = 'change_rate_{}d'.format(round(period / 24))
    period *= 60
    df[col] = df['Close'].pct_change(period)

# ボラティリティ
periods = [1, 6, 24, 24*7, 24*25, 24*100]
for period in periods:
    if period < 24:
        col = 'volatility_{}h'.format(period)
    else:
        col = 'volatility_{}d'.format(round(period / 24))
    period *= 60
    df[col] = np.log(df["Close"]).diff().rolling(period, min_periods=1).std()

# 最小値
periods = [1, 6, 24, 24*7, 24*25, 24*100]
for period in periods:
    if period < 24:
        col = 'min_{}h'.format(period)
    else:
        col = 'min_{}d'.format(round(period / 24))
    period *= 60
    df[col] = df['Close'].rolling(period, min_periods=1).min()

# 最大値
periods = [1, 6, 24, 24*7, 24*25, 24*100]
for period in periods:
    if period < 24:
        col = 'max_{}h'.format(period)
    else:
        col = 'max_{}d'.format(round(period / 24))
    period *= 60
    df[col] = df['Close'].rolling(period, min_periods=1).max()

print(df)
df.to_csv('bitstamp.csv')


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
