import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ExecutionEnv(gym.Env):
    """
    The Gym Environment for the Low-Level Controller (LLC).
    
    Objective: Execute a specific 'Parent Order' (e.g., Sell 50 shares) over a fixed window 
    (e.g., 1 day) while minimizing Market Impact and Slippage.
    
    State Space:
    - Remaining Quantity to Trade (normalized)
    - Time Left in the window (normalized)
    - Current Spread / Volatility (from Regime)
    
    Action Space:
    - Execution Speed: Continuous [0, 1] (Fraction of remaining order to execute NOW)
    """
    def __init__(self, initial_order_size=1000, time_horizon=10):
        super(ExecutionEnv, self).__init__()
        
        self.initial_order_size = initial_order_size
        self.time_horizon = time_horizon # e.g., 10 "micro-steps" per day
        
        # State: [Remaining_Qty, Time_Left, Volatility_Factor]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Action: Fraction of remaining order to dump in this micro-step
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, volatility_factor=0.5):
        super().reset(seed=seed)
        
        self.remaining_qty = 1.0 # 100% of order left
        self.time_left = 1.0     # 100% of time left
        self.volatility = volatility_factor # Passed from HLC's regime
        
        self.current_step = 0
        self.total_slippage = 0
        
        return np.array([self.remaining_qty, self.time_left, self.volatility], dtype=np.float32), {}

    def step(self, action):
        # Action is % of remaining to trade
        # Clip action to valid range
        fraction_to_trade = np.clip(action[0], 0, 1)
        
        # Calculate actual volume executed
        executed_qty = self.remaining_qty * fraction_to_trade
        
        # --- COST SIMULATION (Almgren-Chriss Logic) ---
        # 1. Linear Impact: The more you trade, the more you push the price against you.
        # 2. Volatility Risk: The longer you wait, the more price might drift.
        
        # Impact Cost (Instant)
        market_impact = 0.1 * (executed_qty ** 2) * self.volatility
        
        # Update State
        self.remaining_qty -= executed_qty
        self.current_step += 1
        self.time_left = 1.0 - (self.current_step / self.time_horizon)
        
        # Reward: Negative Cost (We want to minimize cost)
        # We also penalize having inventory left at the very end
        reward = -market_impact
        
        done = self.current_step >= self.time_horizon
        
        if done and self.remaining_qty > 0.01:
            # Penalty for failing to execute the full order
            reward -= 10.0 * self.remaining_qty 
            
        info = {"executed_qty": executed_qty, "slippage": market_impact}
        
        return np.array([self.remaining_qty, self.time_left, self.volatility], dtype=np.float32), reward, done, False, info