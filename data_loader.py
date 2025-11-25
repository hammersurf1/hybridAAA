import yfinance as yf
import pandas as pd
import numpy as np
from typing import List

class MarketDataProcessor:
    """
    Upgraded Data Processor (v2.0).
    Now includes:
    1. Gold (GLD) for inflation hedging.
    2. Trend Signal (Price / SMA) to capture recoveries.
    """

    def __init__(self, start_date: str = "2015-01-01", end_date: str = "2024-01-01", 
                 tickers: List[str] = ["SPY", "TLT", "GLD"]): # Added Gold
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.raw_data = None
        self.processed_data = None

    def fetch_data(self) -> pd.DataFrame:
        """Downloads close prices from Yahoo Finance."""
        print(f"Fetching data for {self.tickers} from {self.start_date} to {self.end_date}...")
        
        # Auto_adjust=True handles splits/dividends
        df = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
        
        # Select the 'Close' column 
        data = df['Close']
        
        # Ensure correct shape if single ticker
        if len(self.tickers) == 1:
            data = data.to_frame()
            data.columns = self.tickers
            
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        self.raw_data = data
        print(f"Data fetched successfully. Shape: {self.raw_data.shape}")
        return self.raw_data

    def add_technical_features(self, window: int = 20) -> pd.DataFrame:
        """
        Engineers features. Now includes Trend Signal.
        """
        if self.raw_data is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")

        df_features = pd.DataFrame(index=self.raw_data.index)

        for ticker in self.tickers:
            price = self.raw_data[ticker]
            
            # 1. Log Returns
            log_ret = np.log(price / price.shift(1))
            df_features[f'{ticker}_log_ret'] = log_ret

            # 2. Rolling Volatility
            vol = log_ret.rolling(window=window).std()
            df_features[f'{ticker}_vol'] = vol

            # 3. Z-Score (Mean Reversion)
            rolling_mean = price.rolling(window=window).mean()
            rolling_std = price.rolling(window=window).std()
            z_score = (price - rolling_mean) / rolling_std
            df_features[f'{ticker}_z_score'] = z_score
            
            # 4. NEW: Trend Signal (Price / 50-Day SMA)
            # > 1.0 means Uptrend, < 1.0 means Downtrend
            # This gives the agent confidence to re-enter the market.
            sma_50 = price.rolling(window=50).mean()
            trend = price / sma_50
            df_features[f'{ticker}_trend'] = trend

        df_features.dropna(inplace=True)
        self.processed_data = df_features
        
        print(f"Feature Engineering Complete. Final Shape: {self.processed_data.shape}")
        return self.processed_data

    def get_data_for_gym(self) -> np.ndarray:
        if self.processed_data is None:
            self.add_technical_features()
        return self.processed_data.values

    def get_asset_returns(self) -> pd.DataFrame:
        return self.raw_data.pct_change().dropna().loc[self.processed_data.index]

if __name__ == "__main__":
    processor = MarketDataProcessor()
    processor.fetch_data()
    df = processor.add_technical_features()
    print(df.tail())