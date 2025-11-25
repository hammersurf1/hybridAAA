# ... existing imports ...
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class PortfolioEnv(gym.Env):
    def __init__(self, data_path="data/processed/enriched_data.csv", initial_balance=10000):
        super(PortfolioEnv, self).__init__()
        
        self.initial_balance = initial_balance
        self.data_path = data_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Enriched data not found at {data_path}. Run src/data/processing.py first!")
            
        self.df = pd.read_csv(data_path, index_col=0)
        
        # Dimensions: 32 Latent + 1 Regime + 1 Sentiment + 5 Weights
        self.state_dim = 32 + 1 + 1 + 5
        self.action_dim = 5 # SPY, TLT, GLD, USO, CASH
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        self.eta = 1/252 
        self.A_t = 0 
        self.B_t = 0 
        self.risk_free_daily = 0.04 / 252

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(0, len(self.df) - 252)
        self.end_step = len(self.df) - 1
        
        # Start safe (100% Cash)
        self.current_weights = np.array([0.0, 0.0, 0.0, 0.0, 1.0]) 
        self.portfolio_value = self.initial_balance
        self.A_t = 0
        self.B_t = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        z_t = row[[f'Z_{i}' for i in range(32)]].values
        regime = row['Regime']
        sentiment = row.get('News_Sentiment', 0)
        
        obs = np.concatenate([z_t, [regime], [sentiment], self.current_weights])
        return obs.astype(np.float32)

    def step(self, action):
        weights = np.array(action).flatten()
        weights = np.exp(weights) / np.sum(np.exp(weights)) 
        
        asset_cols = [c for c in self.df.columns if '_Close' in c and 'VIX' not in c and 'TNX' not in c][:4]
        
        current_prices = self.df.iloc[self.current_step][asset_cols].values
        next_prices = self.df.iloc[self.current_step + 1][asset_cols].values
        
        raw_asset_returns = (next_prices - current_prices) / current_prices
        all_returns = np.append(raw_asset_returns, self.risk_free_daily)
        
        # Calculate Metrics
        turnover = np.sum(np.abs(weights - self.current_weights))
        cost = turnover * 0.0010 
        
        portfolio_return = np.sum(weights * all_returns) - cost
        self.portfolio_value *= (1 + portfolio_return)
        
        # DSR Calculation
        delta_A = portfolio_return - self.A_t
        delta_B = (portfolio_return ** 2) - self.B_t
        old_A = self.A_t
        old_B = self.B_t
        self.A_t += self.eta * delta_A
        self.B_t += self.eta * delta_B
        
        if (old_B - old_A**2) <= 1e-8:
            D_t = 0 
        else:
            term1 = old_B - old_A * portfolio_return
            term2 = 0.5 * old_A * ((portfolio_return**2) - old_B)
            denom = (old_B - old_A**2)**(1.5)
            D_t = (term1 - term2) / denom
            
        # Reward
        turnover_penalty = turnover * 0.5 # Kept strict to force stability
        reward = (D_t * 0.1) - turnover_penalty

        self.current_step += 1
        self.current_weights = weights
        done = self.current_step >= self.end_step
        
        # --- NEW X-RAY DIAGNOSTICS ---
        info = {
            "portfolio_value": self.portfolio_value,
            "return": portfolio_return,
            "regime": self.df.iloc[self.current_step]['Regime'],
            "cash_weight": weights[4],        # Are we finding the exit?
            "turnover": turnover              # Are we churning?
        }
        
        return self._get_observation(), reward, done, False, info