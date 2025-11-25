import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioOptEnv(gym.Env):
    """
    The Master Agent's Environment.
    
    Goal: Allocate capital between Assets (SPY, TLT) and Cash.
    Constraint: Transaction Costs & $10,000 Capital.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 feature_data: np.ndarray, 
                 return_data: np.ndarray, 
                 initial_balance: float = 10000, 
                 lookback_window: int = 30,
                 transaction_cost_pct: float = 0.001): # 0.1% fee/slippage
        
        super(PortfolioOptEnv, self).__init__()

        self.data = feature_data   # The Z-scores/Log-returns (What the AI sees)
        self.returns = return_data # The actual % change (What calculates money)
        
        self.n_assets = return_data.shape[1] # e.g., 2 (SPY, TLT)
        self.n_features = feature_data.shape[1]
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost_pct = transaction_cost_pct

        # ACTION SPACE:
        # We need weights for n_assets + 1 (Cash).
        # Softmax normalization happens in the step function.
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32
        )

        # OBSERVATION SPACE:
        # A matrix of (Lookback Window x Features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(lookback_window, self.n_features),
            dtype=np.float32
        )
        
        # State variables
        self.current_step = None
        self.portfolio_value = None
        self.current_weights = None
        self.portfolio_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.portfolio_value = self.initial_balance
        
        # Start with 100% Cash to simulate a fresh account
        # Weights: [Asset1, Asset2, ... , Cash]
        self.current_weights = np.zeros(self.n_assets + 1)
        self.current_weights[-1] = 1.0 
        
        self.portfolio_history = [self.portfolio_value]
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Slice the data tensor for the window
        obs = self.data[self.current_step - self.lookback_window : self.current_step]
        return obs

    def step(self, action):
        # 1. Normalize Action to Weights (Softmax-like sum to 1)
        weights = np.array(action).flatten()
        weights = np.maximum(weights, 0) # Ensure non-negative
        if np.sum(weights) == 0:
            weights[-1] = 1.0 # Default to cash if agent output is 0
        else:
            weights = weights / np.sum(weights)
            
        # 2. Calculate Transaction Costs
        # Cost is incurred on the CHANGE in weights
        weight_change = np.abs(weights - self.current_weights)
        total_turnover = np.sum(weight_change)
        cost = total_turnover * self.portfolio_value * self.transaction_cost_pct
        
        # 3. Calculate Portfolio Return
        # We only apply returns to the invested assets, not cash (assuming 0% interest for simplicity)
        asset_weights = weights[:-1] 
        asset_returns = self.returns[self.current_step]
        
        # Portfolio return is weighted sum of asset returns
        # Cash return is 0, so we just sum the asset parts
        portfolio_return_pct = np.sum(asset_weights * asset_returns)
        
        # Update Value
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return_pct) - cost
        self.current_weights = weights
        self.portfolio_history.append(self.portfolio_value)
        
        # 4. Reward Engineering (Risk-Adjusted Return)
        # Use simple log return - penalty for volatility
        reward = portfolio_return_pct - (self.transaction_cost_pct * total_turnover) * 10 
        # *10 multiplier to make the agent hate fees
        
        # 5. Advance Step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Info dict for debugging
        info = {
            'portfolio_value': self.portfolio_value,
            'return': portfolio_return_pct,
            'cost': cost,
            'weights': weights
        }
        
        return self._next_observation(), reward, done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Val: ${self.portfolio_value:.2f} | "
              f"W: {[f'{w:.2f}' for w in self.current_weights]}")