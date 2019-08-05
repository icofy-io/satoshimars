
import pandas as pd
import numpy as np


def bollinger(close, last_day):
    '''
    1. Bollinger Bands
    calculate standard deviation, or vol, for the bollinger bands
    sell signal is when price is above upper band
    buy signal is when price is below lower band
    '''
    std_dev = close.rolling(window=5).std() # calculate rolling 5 day std
    sma = close.rolling(5).mean()  # calculate 5 period SMA
    upper_band = sma + std_dev # 1 standard deviations above the 5 day sma
    lower_band = sma - std_dev #1 standard deviations below the 5 day sma
    if last_day.close > upper_band[-1]:
        return 1
    elif last_day.close < lower_band[-1]:
        return -1
    else:
        return 0


def sma(close, last_day):
    '''
    2: Simple Moving Average
    Buy Signal is when close Price is Above 100 day SMA
    Sell Signal is when close Price Below 100 day SMA
    calculate 100 day sma:
    '''
    last_sma = close.rolling(100).mean()[-1]
    if last_day.close > last_sma:
        return 1
    elif last_day.close < last_sma:
        return -1
    else:
        return 0


# 3.  RSI. set to default period of 14 days
# buy signal is when RSI < 50
# sell signal is when RSI > 55
# Define RSI Function
def rsi(close, n=14):
    delta = close.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[n - 1]] = np.mean(u[:n])  # first value is sum of avg gains
    u = u.drop(u.index[:(n - 1)])
    d[d.index[n - 1]] = np.mean(d[:n])  # first value is sum of avg losses
    d = d.drop(d.index[:(n - 1)])
    rs = pd.ewma(close, com=13,min_periods=0,adjust=False,ignore_na=False).mean() / \
         pd.ewma(close, com=13,min_periods=0,adjust=False,ignore_na=False).mean()
    rsi = 100 - 100 / (1 + rs)
    if rsi < 50:
        return 1
    elif rsi > 55:
        return -1
    else:
        return 0


def momentum(close, lookback):
    returns_series = up_or_down(close)
    mom = np.sign(returns_series.rolling(lookback).mean()).iloc[-1]
    if mom > 0:
        return 1
    elif mom < 0:
        return -1
    else:
        return 0


def MACD(close):
    fast_ema = close.rolling(12).mean()
    slow_ema = close.rolling(26).mean()
    macd = fast_ema - slow_ema
    signal =macd.rolling(9).mean()
    crossover = (macd - signal).iloc[-1]
    if crossover > 0:
        return 1
    elif crossover < 0:
        return -1
    else:
        return 0


def up_or_down(close):
    changes = (close - close.shift(1)).iloc[1:]
    return (changes / abs(changes)).apply(int)

