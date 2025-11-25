import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class PortfolioEnv(gym.Env):
    """
    The Gym Environment for the High-Level Controller (HLC).
    OPTIMIZED VERSION: Uses pre-computed features from 'data/processed/enriched_data.csv'.
    """
    def __init__(self, data_path="data/processed/enriched_data.csv", initial_balance=10000):
        super(PortfolioEnv, self).__init__()
        
        self.initial_balance = initial_balance
        self.data_path = data_path
        
        # --- Load Optimized Data ---
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Enriched data not found at {data_path}. Run src/data/processing.py first!")
            
        self.df = pd.read_csv(data_path, index_col=0)
        
        # Verify required columns exist
        if 'Z_0' not in self.df.columns or 'Regime' not in self.df.columns:
             raise ValueError("Datafile missing 'Z_0' or 'Regime' columns. Re-run preprocessing.")

        # --- Define Spaces ---
        # State: 32 (Latent) + 1 (Regime) + 1 (Sentiment) + 4 (Weights) = 38
        self.state_dim = 32 + 1 + 1 + 4
        self.action_dim = 4 # Number of assets
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        # --- DSR Variables (Online Sharpe Calculation) ---
        self.eta = 1/252 
        self.A_t = 0 
        self.B_t = 0 

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # We can start anywhere in the pre-processed file
        # (The file already accounts for the 60-day lag at the start)
        self.current_step = np.random.randint(0, len(self.df) - 252) # Ensure we have at least a year of data left
        self.end_step = len(self.df) - 1
        
        self.current_weights = np.array([0.25, 0.25, 0.25, 0.25]) 
        self.portfolio_value = self.initial_balance
        
        # Reset DSR memory
        self.A_t = 0
        self.B_t = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        # OPTIMIZED: Simple lookup instead of heavy computation
        row = self.df.iloc[self.current_step]
        
        # 1. Latent Features (Z_0 to Z_31)
        z_t = row[[f'Z_{i}' for i in range(32)]].values
        
        # 2. Regime
        regime = row['Regime']
        
        # 3. Sentiment
        sentiment = row.get('News_Sentiment', 0)
        
        # 4. Construct State
        obs = np.concatenate([
            z_t, 
            [regime], 
            [sentiment], 
            self.current_weights
        ])
        
        return obs.astype(np.float32)

    def step(self, action):
        # Normalize action (Softmax)
        weights = np.array(action).flatten()
        weights = np.exp(weights) / np.sum(np.exp(weights)) 
        
        # Get returns for the *next* day
        # Note: We need the raw Close prices to calculate returns.
        # Assuming original columns (SPY_Close, etc.) are still in the enriched dataframe.
        # Based on processing.py, we concatenated df_aligned, so they are there.
        
        # Filter columns that contain '_Close' and are the assets
        # Safe way: hardcode the asset list or rely on column position if strict
        # Here we rely on name matching for safety
        asset_cols = [c for c in self.df.columns if '_Close' in c and 'VIX' not in c and 'TNX' not in c]
        # Ensure we only get the 4 assets
        asset_cols = asset_cols[:4] 
        
        current_prices = self.df.iloc[self.current_step][asset_cols].values
        next_prices = self.df.iloc[self.current_step + 1][asset_cols].values
        
        # Asset Returns
        asset_returns = (next_prices - current_prices) / current_prices
        
        # Transaction Costs (10bps)
        turnover = np.sum(np.abs(weights - self.current_weights))
        cost = turnover * 0.0010 
        
        portfolio_return = np.sum(weights * asset_returns) - cost
        
        # Update Balance
        self.portfolio_value *= (1 + portfolio_return)
        
        # --- Calculate Reward: Differential Sharpe Ratio (DSR) ---
        delta_A = portfolio_return - self.A_t
        delta_B = (portfolio_return ** 2) - self.B_t
        
        old_A = self.A_t
        old_B = self.B_t
        
        self.A_t += self.eta * delta_A
        self.B_t += self.eta * delta_B
        
        if (old_B - old_A**2) <= 1e-8: # Added epsilon for stability
            D_t = 0 
        else:
            term1 = old_B - old_A * portfolio_return
            term2 = 0.5 * old_A * ((portfolio_return**2) - old_B)
            denom = (old_B - old_A**2)**(1.5)
            D_t = (term1 - term2) / denom
            
        reward = D_t * 0.1 

        # Advance Time
        self.current_step += 1
        self.current_weights = weights
        
        done = self.current_step >= self.end_step
        
        info = {
            "portfolio_value": self.portfolio_value,
            "return": portfolio_return,
            "regime": self.df.iloc[self.current_step]['Regime']
        }
        
        return self._get_observation(), reward, done, False, info