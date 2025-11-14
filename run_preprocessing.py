import os
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# ======================================================================
# --- 1. CONFIGURATION ---
# ======================================================================

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "data")
print(f"Data directory set to: {DATA_DIRECTORY}")

# ---
# NOTE: Start with two files. Once this script succeeds,
# you can add all your other JSON files to this list.
# ---
JSON_FILES = [
    "2017_processed.json",
    "2018_processed.json",
    "2019_processed.json",
    "2020_processed.json"
    # "2021_processed.json",
    # "2022_processed.json",
    # "2023_processed.json"
]

DJIA_TICKERS = [
    'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON',
    'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
    'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW'
]

MARKET_CLOSE_UTC_HOUR = 21

# ======================================================================
# --- 2. DATA LOADING (Robust, One-by-One) ---
# ======================================================================

def load_all_news_data(directory, files_to_load):
    """
    Loads and combines all JSON files ONE BY ONE to save memory.
    Uses pd.read_json() which is robust to 'NaN' and other errors.
    """
    all_articles_dfs = []
    print(f"\nLoading all JSON files from {directory} (one by one)...")
    
    total_articles = 0
    for filename in files_to_load:
        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            print(f"  WARNING: File not found {filepath}. Skipping.")
            continue
            
        print(f"  Loading {filename}...")
        try:
            df = pd.read_json(filepath, orient='records')
            sentiment_df = pd.json_normalize(df['sentiment'])
            sentiment_df = sentiment_df.rename(columns={
                'positive': 'pos', 'negative': 'neg', 'neutral': 'neu'
            })
            
            df = df.join(sentiment_df)
            df = df[['date_publish', 'pos', 'neg', 'neu', 'mentioned_companies']]
            all_articles_dfs.append(df)
            total_articles += len(df)
            del df
            del sentiment_df
            
        except Exception as e:
            print(f"  ERROR: Could not process file {filepath}. Error: {e}")

    print(f"\nSuccessfully loaded {total_articles:,} total articles.")
    
    if not all_articles_dfs:
        print("CRITICAL ERROR: No data was loaded.")
        return pd.DataFrame() 
        
    final_df = pd.concat(all_articles_dfs, ignore_index=True)
    final_df['date_publish'] = pd.to_datetime(final_df['date_publish'], utc=True)
    final_df = final_df.dropna(subset=['date_publish', 'pos'])
    return final_df

def get_price_data(ticker="^DJI", start_date="2017-01-01", end_date="2020-12-31"):
    """
    Downloads DJIA (ticker ^DJI) price data from yfinance.
    --- THIS FUNCTION CONTAINS THE FINAL FIX ---
    """
    print(f"\nDownloading {ticker} price data from yfinance...")
    
    end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)
    
    price_df = yf.download(
        ticker, 
        start=start_date, 
        end=end_date_dt.strftime("%Y-%m-%d")
    )
    
    # --- FIX for KeyError: 'Close' ---
    # yfinance creates a MultiIndex header, e.g., ('Close', '^DJI').
    # We must "flatten" it to get a single header.
    if isinstance(price_df.columns, pd.MultiIndex):
        print("Detected MultiIndex columns. Flattening...")
        # This replaces ('Close', '^DJI') with just 'Close'
        price_df.columns = price_df.columns.get_level_values(0)
    # --- End of FIX ---

    # Now the columns are ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    
    # Reset the index to turn 'Date' into a column
    price_df = price_df.reset_index()
    
    # Clean the 'Date' column to be a date only (no time) and set to UTC
    price_df['Date'] = pd.to_datetime(price_df['Date'].dt.date, utc=True)
    
    print(f"Downloaded and flattened {len(price_df)} rows of price data.")
    return price_df

# ======================================================================
# --- 3. FEATURE ENGINEERING ---
# ======================================================================

def create_price_features(price_df):
    """
    Calculates baseline price features. 
    This should now work since 'Close' is a valid column.
    """
    print("Creating price features (returns, volatility, etc.)...")
    df_out = price_df.copy()
    
    # This line will no longer cause a KeyError
    df_out['return_1d'] = df_out['Close'].pct_change(1)
    df_out['return_3d'] = df_out['Close'].pct_change(3)
    df_out['return_5d'] = df_out['Close'].pct_change(5)
    df_out['vol_10d'] = df_out['return_1d'].rolling(10).std()
    df_out['vol_20d'] = df_out['return_1d'].rolling(20).std()
    df_out['ma_10d'] = df_out['Close'].rolling(10).mean()
    df_out['ma_20d'] = df_out['Close'].rolling(20).mean()
    
    # The Target Variable
    df_out['target'] = (df_out['Close'].shift(-1) > df_out['Close']).astype(int)
    
    df_out = df_out.dropna()
    return df_out

def create_daily_sentiment_features(news_df, trading_dates):
    """
    Aggregates news into daily Naive and FiTS sentiment features.
    """
    print("Building daily sentiment features (Naive vs. FiTS)...")
    
    news_df = news_df.set_index('date_publish')
    daily_features = []
    
    for date in trading_dates:
        try:
            day_group = news_df.loc[date.strftime('%Y-%m-%d')]
        except KeyError:
            daily_features.append({'Date': date})
            continue

        if day_group.empty:
            daily_features.append({'Date': date})
            continue

        naive_pos = day_group['pos'].mean()
        naive_neg = day_group['neg'].mean()
        
        is_relevant = day_group['mentioned_companies'].apply(
            lambda tickers: any(t in DJIA_TICKERS for t in tickers) if (tickers and isinstance(tickers, list)) else False
        )
        relevant_news = day_group[is_relevant]
        
        timely_and_relevant_news = relevant_news[
            relevant_news.index.hour < MARKET_CLOSE_UTC_HOUR
        ]
        
        if not timely_and_relevant_news.empty:
            fits_pos = timely_and_relevant_news['pos'].mean()
            fits_neg = timely_and_relevant_news['neg'].mean()
        else:
            fits_pos = np.nan
            fits_neg = np.nan
            
        daily_features.append({
            'Date': date,
            'naive_pos': naive_pos,
            'naive_neg': naive_neg,
            'fits_pos': fits_pos,
            'fits_neg': fits_neg
        })

    return pd.DataFrame(daily_features)

# ======================================================================
# --- 4. MAIN EXECUTION (This should now work) ---
# ======================================================================

if __name__ == "__main__":
    print("--- STARTING PREPROCESSING PIPELINE (FINAL) ---")
    
    # Step 1: Load Price Data (Header is now flat)
    price_data = get_price_data(
        ticker="^DJI",
        start_date="2017-01-01",
        end_date="2020-12-31" # Use file up to 2020
    )
    
    # Step 2: Create Price Features (KeyError is fixed)
    price_features_df = create_price_features(price_data)
    
    # Step 3: Load News Data
    news_df = load_all_news_data(DATA_DIRECTORY, JSON_FILES)
    
    if not news_df.empty:
        # Step 4: Build Daily Sentiment Features
        daily_sentiment_df = create_daily_sentiment_features(
            news_df,
            price_features_df['Date'] # Pass the clean 'Date' column
        )
        
        # Step 5: Merge All Data (This will work)
        print("Merging all data frames on 'Date' column...")
        final_df = pd.merge(
            price_features_df, 
            daily_sentiment_df, 
            on='Date', 
            how='left' 
        )
        
        # Fill missing sentiment with 0.0 (assuming "no news" is neutral)
        sentiment_cols = ['naive_pos', 'naive_neg', 'fits_pos', 'fits_neg']
        final_df[sentiment_cols] = final_df[sentiment_cols].fillna(0.0)
        
        # Step 6: Save Final Preprocessed Data
        output_filename = "final_preprocessed_data_full.csv"
        final_df.to_csv(output_filename, index=False) 
        
        print("\n--- PREPROCESSING COMPLETE ---")
        print(f"Successfully saved to {output_filename}")
        
        print("\nFinal DataFrame Info:")
        final_df.info()
        
        print("\nFinal DataFrame Head:")
        print(final_df.head())
    
    else:
        print("\n--- PREPROCESSING FAILED: No news data was loaded. ---")