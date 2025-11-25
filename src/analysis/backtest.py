import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import glob

# --- ABSOLUTE FIX: Ensure the project root is always in the path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------------------------------------------------------

def run_backtest(model_name="hybrid_aaa_ppo"):
    print(f"--- Starting Backtest for {model_name} ---")
    print(f"Project Root Detected: {project_root}")
    
    # 1. Define paths to critical files
    model_path = os.path.join(project_root, "models", "final", model_name)
    stats_path = os.path.join(project_root, "models", "final", f"{model_name}_vecnorm.pkl")
    data_path = os.path.join(project_root, "data", "processed", "enriched_data.csv")

    # Error Handling for file existence
    if not os.path.exists(stats_path):
        print(f"CRITICAL ERROR: Stats file not found at '{stats_path}'")
        return
        
    if not os.path.exists(data_path):
        print(f"CRITICAL ERROR: Data file not found at '{data_path}'")
        return

    # Note: PPO.load handles the .zip extension automatically
    if not os.path.exists(f"{model_path}.zip"):
        print(f"CRITICAL ERROR: Model file not found at '{model_path}.zip'")
        return

    print(f"Model Found: {model_path}.zip")
    print(f"Stats Found: {stats_path}")
    print(f"Data Found:  {data_path}")
    
    # 2. Setup Environment
    print("Loading Environment...")
    raw_env = DummyVecEnv([lambda: PortfolioEnv(data_path=data_path)])
    
    # Load Stats
    print(f"Loading Stats...")
    try:
        env = VecNormalize.load(stats_path, raw_env)
    except Exception as e:
        print(f"Failed to load normalization stats: {e}")
        return
        
    env.training = False 
    env.norm_reward = False 
    
    # 3. Load Agent
    print(f"Loading Agent from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 4. Run Loop
    obs = env.reset()
    done = False
    
    portfolio_values = []
    spy_values = []
    cash_weights = []
    regimes = []
    
    internal_env = env.envs[0] 
    data_len = len(internal_env.df)
    
    start_step = internal_env.current_step
    initial_spy_price = internal_env.df.iloc[start_step]['SPY_Close']
    initial_balance = internal_env.portfolio_value
    
    print("Running Prediction Loop...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        info_dict = info[0] 
        portfolio_values.append(info_dict['portfolio_value'])
        cash_weights.append(info_dict['cash_weight'])
        regimes.append(info_dict['regime'])
        
        current_step = internal_env.current_step
        if current_step < data_len:
            current_spy = internal_env.df.iloc[current_step]['SPY_Close']
            spy_performance = (current_spy / initial_spy_price) * initial_balance
            spy_values.append(spy_performance)
        
        if current_step >= data_len - 1:
            break

    # 5. Plotting
    print("Generating Charts...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(portfolio_values, label='Hybrid AAA Agent', color='blue', linewidth=2)
    ax1.plot(spy_values, label='SPY Buy & Hold', color='gray', linestyle='--', alpha=0.7)
    ax1.set_title(f"Backtest: Agent vs Market ({len(portfolio_values)} Days)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(cash_weights, label='Cash Allocation %', color='green')
    ax2.set_ylabel("Cash Weight (0-1)")
    ax2.set_ylim(0, 1.1)
    
    regimes = np.array(regimes)
    bear_zones = (regimes >= 1)
    ax2.fill_between(range(len(regimes)), 0, 1, where=bear_zones, color='red', alpha=0.1, label='Bear/Volatile Regime')
    
    ax2.set_title("Agent Behavior: Cash Allocation vs Market Regimes")
    ax2.set_xlabel("Trading Days")
    ax2.legend()
    
    output_path = os.path.join(project_root, "backtest_results.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Success! Chart saved to '{output_path}'")
    
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance * 100
    spy_return = (spy_values[-1] - initial_balance) / initial_balance * 100
    print("-" * 30)
    print(f"Final Agent Return: {total_return:.2f}%")
    print(f"Final SPY Return:   {spy_return:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    run_backtest()