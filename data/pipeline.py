import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class DataIngestion:
    """
    Handles the fetching, alignment, and cleaning of multi-modal financial data.
    Designed for the Hybrid AAA Architecture.
    """
    def __init__(self, start_date="2010-01-01", end_date=None):
        # Configuration parameters for the data fetch
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')
        
        # 1. The Strategic Asset Universe (Liquid ETFs for $10k account)
        self.assets = ['SPY', 'TLT', 'GLD', 'USO']
        
        # 2. The Macro "Context" Features (For XGBoost Regime Classifier)
        # ^VIX = Volatility Index, ^TNX = 10-Year Treasury Yield
        self.macro_tickers = ['^VIX', '^TNX'] 
        
        # Define the path to save raw data
        self.data_store_path = "data/raw"
        os.makedirs(self.data_store_path, exist_ok=True)

    def fetch_market_data(self):
        """Fetches Price and Volume data for Assets and Macro indicators."""
        print(f"Loading market data from {self.start_date} to {self.end_date}...")
        
        # Download all tickers at once to ensure alignment
        tickers = self.assets + self.macro_tickers
        # auto_adjust=True handles stock splits/dividends automatically
        raw_data = yf.download(tickers, start=self.start_date, end=self.end_date, group_by='ticker', auto_adjust=True)
        
        aligned_data = pd.DataFrame()
        
        # Process Asset Data (Close and Volume)
        for ticker in self.assets:
            df = raw_data[ticker].copy()
            df = df[['Close', 'Volume']]
            df.columns = [f"{ticker}_Close", f"{ticker}_Volume"]
            df = df.fillna(method='ffill') # Essential for filling in missing days (e.g., holidays)
            aligned_data = pd.concat([aligned_data, df], axis=1)

        # Process Macro Data (Close only)
        for ticker in self.macro_tickers:
            df = raw_data[ticker].copy()
            clean_name = ticker.replace('^', '')
            df = df[['Close']]
            df.columns = [f"{clean_name}_Close"]
            df = df.fillna(method='ffill')
            aligned_data = pd.concat([aligned_data, df], axis=1)

        # Final cleanup and indexing
        aligned_data.dropna(inplace=True)
        
        print(f"Success. Fetched {len(aligned_data)} days of aligned data.")
        return aligned_data

    def generate_mock_sentiment(self, index):
        """
        Placeholder function for the HARLF component. 
        Generates synthetic sentiment scores [-1, 1] for initial architectural testing.
        """
        print("Generating synthetic sentiment data (Placeholder)...")
        np.random.seed(42)
        # Simple random series, centered around 0 (neutral)
        sentiment = np.random.normal(0, 0.4, size=len(index)) 
        sentiment = np.clip(sentiment, -1, 1)
        
        return pd.Series(sentiment, index=index, name="News_Sentiment")

    def run_pipeline(self):
        # 1. Get Quantitative Data (Assets + Macro)
        market_df = self.fetch_market_data()
        
        # 2. Get Qualitative Data (MOCK Sentiment)
        sentiment_series = self.generate_mock_sentiment(market_df.index)
        
        # 3. Fuse Data into the primary raw time series
        full_df = pd.concat([market_df, sentiment_series], axis=1)
        
        # 4. Save to Disk (in the raw data folder)
        file_path = os.path.join(self.data_store_path, "hybrid_aaa_raw_data.csv")
        full_df.to_csv(file_path)
        print("-" * 50)
        print(f"Data Pipeline Complete. Raw aligned data saved to: {file_path}")
        print("--- Head of Data ---")
        print(full_df.head())
        print("-" * 50)
        return full_df

# --- Execution Example ---
if __name__ == "__main__":
    # Note: When you run this file directly, it assumes you are running from the
    # root directory of the Hybrid-AAA-Trader project.
    pipeline = DataIngestion()
    df = pipeline.run_pipeline()