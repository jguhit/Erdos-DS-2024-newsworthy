import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_stock_data(csv_file, scatter_features, corr_features):
    """
    Analyze stock data from a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file containing the data.
        scatter_features (list): List of two strings representing the scatter plot features.
        corr_features (list): List of strings representing the features for correlation analysis.
    """

    # Load and preprocess data
    date_columns = ["Publishing Time", "Market Time", "Date"]
    df = pd.read_csv(csv_file, parse_dates=date_columns, low_memory=False)
    df = df.dropna(subset=["finvader"])
    float_columns = ["Open", "High", "Low"]
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.drop(["Unnamed: 20", "Unnamed: 21", "Unnamed: 22"], axis=1)

    # Feature engineering
    daily_data = df.groupby(['Date', 'Ticker']).agg({
        'finvader': [
            ('Daily_avg_sentiment', 'mean'),
            ('num_neg_articles', lambda x: (x < 0).sum()),
            ('num_pos_articles', lambda x: (x > 0).sum()),
            ('max_senti_articles', 'max'),
            ('min_senti_articles', 'min'),
            ('Tot_num_articles', lambda x: x.notnull().sum())
        ],
        'Open': 'mean'
    }).reset_index()
    daily_data.columns = ['Date', 'Ticker', 'Daily_avg_sentiment', 'num_neg_articles', 'num_pos_articles', 
                         'max_senti_articles', 'min_senti_articles', 'Tot_num_articles', 'Open']
    daily_data = daily_data.sort_values(by=['Ticker', 'Date'])

    daily_data['pct_change_open'] = daily_data.groupby('Ticker')['Open'].transform(lambda x: x.pct_change(fill_method=None) * 100)
    daily_data['pct_change_Finvader'] = daily_data.groupby('Ticker')['Daily_avg_sentiment'].transform(lambda x: x.pct_change(fill_method=None) * 100)
    daily_data['rolling_avg_open'] = daily_data.groupby('Ticker')['Open'].transform(lambda x: x.rolling(window=7).mean())
    daily_data['pct_change_open_7days_avg'] = (daily_data['Open'] - daily_data['rolling_avg_open'].shift(1)) / daily_data['rolling_avg_open'].shift(1) * 100
    daily_data['rolling_avg_finvader'] = daily_data.groupby('Ticker')['Daily_avg_sentiment'].transform(lambda x: x.rolling(window=7).mean())
    daily_data['pct_change_avg_finvader_7days_avg'] = (daily_data['Daily_avg_sentiment'] - daily_data['rolling_avg_finvader'].shift(1)) / daily_data['rolling_avg_finvader'].shift(1) * 100

    # Scatter Plot and Correlation Matrix

    def quantile_985(data):
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 2:
            return None, None
        lower_quantile = np.quantile(data, 0.0075)
        upper_quantile = np.quantile(data, 0.9925)
        return lower_quantile, upper_quantile

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 15), sharex=True, sharey=True)
    axes = axes.flatten() 

    num_stocks = len(daily_data['Ticker'].unique())
    for i, ticker in enumerate(daily_data['Ticker'].unique()[:num_stocks]):
        subset = daily_data[daily_data['Ticker'] == ticker]
        lower_bound_x, upper_bound_x = quantile_985(subset[scatter_features[0]])
        lower_bound_y, upper_bound_y = quantile_985(subset[scatter_features[1]])

        if lower_bound_x is None or upper_bound_x is None or lower_bound_y is None or upper_bound_y is None:
            continue 

        ax = axes[i]
        ax.scatter(subset[scatter_features[0]], subset[scatter_features[1]], alpha=0.5)
        ax.set_title(ticker)
        ax.set_xlabel(f'{scatter_features[0]}')
        ax.set_ylabel(f'{scatter_features[1]}')

        ax.set_xlim(lower_bound_x, upper_bound_x)
        ax.set_ylim(lower_bound_y, upper_bound_y)  

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Scatter Plot of {scatter_features[1]} vs {scatter_features[0]}', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  
    plt.savefig('scatter_plot.png')
    plt.show()

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 15), sharex=True, sharey=True)
    axes = axes.flatten() 

    for i, ticker in enumerate(daily_data['Ticker'].unique()[:num_stocks]):
        subset = daily_data[daily_data['Ticker'] == ticker]
        relevant_columns = corr_features
        subset = subset[relevant_columns]
        correlation_matrix = subset.corr()

        ax = axes[i]
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title(ticker)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Correlation Matrices for Tickers', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  
    plt.savefig('correlation_matrices.png')
    plt.show()
