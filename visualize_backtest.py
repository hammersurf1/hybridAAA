import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from data_loader import MarketDataProcessor
from portfolio_env import PortfolioOptEnv

def visualize_performance(model_path="ppo_portfolio_agent.zip"):
    print("--- Starting Backtest Visualization ---")

    # 1. Load Data
    processor = MarketDataProcessor(start_date="2015-01-01", end_date="2024-01-01")
    processor.fetch_data()
    processor.add_technical_features()
    
    feature_data = processor.get_data_for_gym()
    raw_returns_df = processor.get_asset_returns()
    return_data = raw_returns_df.values
    
    # 2. Prepare Test Set
    lookback_window = 30
    split_idx = int(len(feature_data) * 0.8)
    start_idx = max(0, split_idx - lookback_window)
    
    test_features = feature_data[start_idx:]
    test_returns = return_data[start_idx:]
    test_dates = raw_returns_df.index[split_idx:]
    
    print(f"Backtesting over {len(test_dates)} days of unseen data.")

    # 3. Initialize Model and Environment
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return

    test_env = PortfolioOptEnv(
        feature_data=test_features, 
        return_data=test_returns,
        initial_balance=10000,
        lookback_window=lookback_window
    )
    
    # 4. Run Backtest & TRACK WEIGHTS
    obs, _ = test_env.reset()
    done = False
    
    agent_values = [test_env.initial_balance]
    agent_weights = [] # Store daily weights
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        agent_values.append(info['portfolio_value'])
        agent_weights.append(info['weights'])
        
    # 5. Data Alignment
    simulation_values = agent_values[1:]
    weights_array = np.array(agent_weights) # Shape: (Days, Assets + Cash)
    
    aligned_len = min(len(test_dates), len(simulation_values))
    
    plotting_dates = test_dates[:aligned_len]
    plotting_agent_values = simulation_values[:aligned_len]
    plotting_weights = weights_array[:aligned_len]
    
    full_agent_values = [test_env.initial_balance] + plotting_agent_values
    final_agent_value = full_agent_values[-1]
    
    # 6. Benchmark
    test_returns_df = raw_returns_df.loc[plotting_dates]
    num_assets = test_returns_df.shape[1]
    benchmark_weights = np.ones(num_assets) / num_assets
    benchmark_daily_return = (test_returns_df * benchmark_weights).sum(axis=1)
    benchmark_values = (1 + benchmark_daily_return).cumprod() * test_env.initial_balance
    benchmark_values = [test_env.initial_balance] + benchmark_values.tolist()
    final_benchmark_value = benchmark_values[-1]

    # 7. Metrics
    def calculate_metrics(values):
        returns = pd.Series(values).pct_change().dropna()
        cumulative_return = (returns + 1).prod() - 1
        annualized_return = cumulative_return * (252 / len(returns))
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        cumulative_values = pd.Series(values)
        peak = cumulative_values.expanding(min_periods=1).max()
        drawdown = (cumulative_values / peak) - 1
        max_drawdown = drawdown.min()
        
        return {
            'Cum. Return': f"{cumulative_return * 100:.2f}%",
            'Ann. Return': f"{annualized_return * 100:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown * 100:.2f}%",
        }

    agent_metrics = calculate_metrics(full_agent_values)
    benchmark_metrics = calculate_metrics(benchmark_values)

    print("\n--- Final Performance Report ---")
    print(f"{'Metric':<20}{'RL Agent':<20}{'Benchmark':<20}")
    print("-" * 60)
    for metric in agent_metrics.keys():
        print(f"{metric:<20}{agent_metrics[metric]:<20}{benchmark_metrics[metric]:<20}")

    # 8. Dual Plot: Equity Curve + Weights Stackplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Equity Curve
    ax1.plot(plotting_dates, plotting_agent_values, label='RL Agent', color='dodgerblue', linewidth=2)
    ax1.plot(plotting_dates, benchmark_values[1:], label='Benchmark', color='orange', linestyle='--')
    ax1.set_title('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Portfolio Weights (Stacked)
    # Get ticker names + Cash
    asset_labels = processor.tickers + ['Cash']
    
    # Stackplot requires transposing the array to (Features, Time)
    ax2.stackplot(plotting_dates, plotting_weights.T, labels=asset_labels, alpha=0.8)
    ax2.set_title('Portfolio Allocation (Weights)')
    ax2.set_ylabel('Weight (0-1)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_performance()