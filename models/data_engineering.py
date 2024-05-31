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

# separates the data into train and test sets, leaving out the last year of data as the test set
def train_test_split(df):
    train = df.loc[df.index < datetime.datetime(2023,3,1)].copy()
    test = df.drop(train.index).copy()
    return (train, test)

# creates a column in the dataframe called "Diff" which is the first differences of the Open price
def create_first_diff(df):
    df['Diff'] = df.Open.diff()
    return df

# takes in the training set and returns a list of indices for training and validation
def get_cv_splits(df):
    dates = [datetime.datetime(2022,3,1),
             datetime.datetime(2022,6,1),
             datetime.datetime(2022,9,1),
             datetime.datetime(2022,12,1),
             datetime.datetime(2023,3,1)]
    splits = []
    for i in range(len(dates)-1):
        train_idx = df.loc[df.index < dates[i]].index
        test_idx = df.loc[(df.index >= dates[i]) & (df.index < dates[i+1])].index
        splits.append((train_idx, test_idx))
    return splits

