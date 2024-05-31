# File to store common methods for creating usable dataframes ready to enter the modeling pipeline

import pandas as pd
import datetime

# determines if the sentiment of an article is positive, negative, or neutral
def _overall_sentiment(x:int):
    threshold = .1
    if x > threshold:
        return 'pos'
    elif x < -threshold:
        return 'neg'
    else:
        return 'neu'

# reads from the complete.csv file and returns a dictionary of dataframes where the keys are the tickers
def separate_by_stock():
    # read in full data set
    df = pd.read_csv('../data/complete_next_open.csv')

    # create overall sentiment column
    df['overall_sen'] = df['finvader_tot'].apply(_overall_sentiment)
    df['overall_sen'] = df['overall_sen'].astype('category')

    # value counts for overall sentiment by market date and ticker
    counts = df.groupby(['Market Date', 'Ticker'])['overall_sen'].value_counts()

    # we will take the mean of each of these features
    features = ['finvader_neg',
            'finvader_neu',
            'finvader_pos',
            'finvader_tot',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'Dividends',
            'Stock Splits']
    df_mean = df.groupby(['Market Date', 'Ticker'])[features].mean().reset_index()

    # add in the article counts to the df_mean dataframe
    labels = {'pos_art_count':'pos', 'neg_art_count':'neg', 'neu_art_count':'neu'}
    for l in labels:
        df_mean[l] = df_mean.apply(lambda x: counts.loc[x['Market Date'], x['Ticker']][labels[l]], axis = 1)
    df_mean['total_articles'] = df_mean['pos_art_count'] + df_mean['neg_art_count'] + df_mean['neu_art_count']

    # change market date to datetime format
    df_mean['Market Date'] = pd.to_datetime(df_mean['Market Date'])


    tickers = df_mean['Ticker'].unique()

    # create dictionary of data frames, one for each ticker
    ticker_frames = {}
    for tick in tickers:
        ticker_frames[tick] = df_mean.loc[df_mean['Ticker'] == tick].set_index('Market Date').drop(columns = ['Ticker', 'Dividends'])
    
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

