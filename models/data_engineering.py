# File to store common methods for creating usable dataframes ready to enter the modeling pipeline

import pandas as pd
import datetime

# reads from the complete.csv file and returns a dictionary of dataframes where the keys are the tickers
def separate_by_stock():
    df_all = pd.read_csv('../data/complete.csv')
    df_all['Market Time'] = pd.to_datetime(df_all['Market Time'], utc = True)
    df_all['Publishing Time'] = pd.to_datetime(df_all['Publishing Time'], utc = True)
    df_all['Date'] = pd.to_datetime(df_all['Date'])

    tickers = df_all['Ticker'].unique()

    ticker_frames = {}
    df_t = df_all.groupby(['Date', 'Ticker'])[['sentiment_tot', 'finvader', 'Open', 'Close']].mean().reset_index()
    for tick in tickers:
        ticker_frames[tick] = df_t[df_t['Ticker'] == tick].set_index('Date')
    
    return ticker_frames

def train_test_split(df):
    train = df.loc[df.index < datetime.datetime(2023,3,1)].copy()
    test = df.drop(train.index).copy()
    return (train, test)

