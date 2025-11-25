import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- ABSOLUTE FIX: Ensure the project root is always in the path ---
# This resilient pathing logic guarantees Python can resolve 'src.envs'
# Finds the root directory by walking up two levels from the current file's location.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------------------------------------------------------

# Now the absolute imports will work correctly
from src.envs.portfolio_env import PortfolioEnv

def run_backtest(model_name="hybrid_aaa_ppo"):
    print(f"--- Starting Backtest for {model_name} ---")
    
    # 1. Paths
    # Construct absolute paths from the project root to avoid CWD issues.
    model_path = os.path.join(project_root, "models", "final", model_name)
    stats_path = os.path.join(project_root, "models", "final", f"{model_name}_vecnorm.pkl")
    
    if not os.path.exists(stats_path):
        print(f"Error: Stats file not found at {stats_path}. Train the model first!")
        return

    # 2. Setup Environment
    data_path = os.path.join(project_root, "data", "processed", "enriched_data.csv")
    raw_env = DummyVecEnv([lambda: PortfolioEnv(data_path=data_path)])
    
    # Load the Training Normalization Stats (Critical!)
    # The agent learned to see "Normalized Data". If we feed it raw data, it will panic.
    env = VecNormalize.load(stats_path, raw_env)
    env.training = False # Do NOT update stats during test
    env.norm_reward = False # We want real dollar rewards
    
    # 3. Load Agent
    model = PPO.load(model_path)
    
    # 4. Run Loop
    obs = env.reset()
    done = False
    
    # Logs
    portfolio_values = []
    spy_values = []
    cash_weights = []
    regimes = []
    dates = []
    
    # Access the internal DataFrame to get Dates and SPY price for benchmarking
    # The environment inside DummyVecEnv is accessed via envs[0]
    internal_env = env.envs[0] 
    data_len = len(internal_env.df)
    
    # Align start index
    start_step = internal_env.current_step
    initial_spy_price = internal_env.df.iloc[start_step]['SPY_Close']
    initial_balance = internal_env.portfolio_value
    
    print("Running Prediction Loop...")
    
    while not done:
        # Predict (Deterministic = True means NO randomness)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Log Data
        info_dict = info[0] # VectorEnv returns a list
        portfolio_values.append(info_dict['portfolio_value'])
        cash_weights.append(info_dict['cash_weight'])
        regimes.append(info_dict['regime'])
        
        # Benchmark Logic (SPY Buy & Hold)
        current_step = internal_env.current_step
        if current_step < data_len:
            current_spy = internal_env.df.iloc[current_step]['SPY_Close']
            spy_performance = (current_spy / initial_spy_price) * initial_balance
            spy_values.append(spy_performance)
            dates.append(internal_env.df.index[current_step])
        
        # Stop if we run out of data
        if current_step >= data_len - 1:
            break

    # 5. Plotting
    print("Generating Charts...")
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Performance
    ax1.plot(portfolio_values, label='Hybrid AAA Agent', color='blue', linewidth=2)
    ax1.plot(spy_values, label='SPY Buy & Hold', color='gray', linestyle='--', alpha=0.7)
    ax1.set_title(f"Backtest: Agent vs Market ({len(portfolio_values)} Days)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cash & Regimes
    # We color the background based on Regime
    ax2.plot(cash_weights, label='Cash Allocation %', color='green')
    ax2.set_ylabel("Cash Weight (0-1)")
    ax2.set_ylim(0, 1.1)
    
    # Shade Bear Regimes (Regime 1 or 2)
    # Using a simple fill approach
    regimes = np.array(regimes)
    bear_zones = (regimes >= 1)
    ax2.fill_between(range(len(regimes)), 0, 1, where=bear_zones, color='red', alpha=0.1, label='Bear/Volatile Regime')
    
    ax2.set_title("Agent Behavior: Cash Allocation vs Market Regimes")
    ax2.set_xlabel("Trading Days")
    ax2.legend()
    
    plt.tight_layout()
    output_path = os.path.join(project_root, "backtest_results.png")
    plt.savefig(output_path)
    print(f"Success! Chart saved to '{output_path}'")
    
    # Final Stats
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance * 100
    spy_return = (spy_values[-1] - initial_balance) / initial_balance * 100
    print("-" * 30)
    print(f"Final Agent Return: {total_return:.2f}%")
    print(f"Final SPY Return:   {spy_return:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    run_backtest()