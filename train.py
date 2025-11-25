import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from data_loader import MarketDataProcessor
from portfolio_env import PortfolioOptEnv

def train_agent():
    print("--- Starting Training Pipeline ---")
    
    # 1. Prepare the Data
    # -------------------
    # We fetch a long history to ensure we have different market regimes (Bull & Bear)
    processor = MarketDataProcessor(start_date="2015-01-01", end_date="2024-01-01")
    processor.fetch_data()
    processor.add_technical_features()
    
    # Align features and returns
    # The environment needs both: Features to 'see', Returns to 'calculate profit'
    feature_data = processor.get_data_for_gym()
    return_data = processor.get_asset_returns().values
    
    # Verify Alignment
    assert len(feature_data) == len(return_data), "Data mismatch! Features and Returns must be same length."
    
    # 2. Split Data (Train vs Test)
    # -----------------------------
    # 80% Training, 20% Testing
    split_idx = int(len(feature_data) * 0.8)
    
    train_features = feature_data[:split_idx]
    train_returns = return_data[:split_idx]
    
    test_features = feature_data[split_idx:]
    test_returns = return_data[split_idx:]
    
    print(f"Training Samples: {len(train_features)}")
    print(f"Testing Samples:  {len(test_features)}")

    # 3. Initialize Environment
    # -------------------------
    # We wrap the env in DummyVecEnv because PPO expects a vectorized environment
    # (even if it's just one env, it expects the array structure)
    env = DummyVecEnv([lambda: PortfolioOptEnv(
        feature_data=train_features, 
        return_data=train_returns,
        initial_balance=10000
    )])

    # 4. Setup the Agent (PPO)
    # ------------------------
    # MlpPolicy: Multi-Layer Perceptron (Standard Neural Network)
    # learning_rate: How fast it updates its beliefs (3e-4 is standard for PPO)
    # ent_coef: Entropy Coefficient. 0.01 encourages exploration (trying random things)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003, 
        ent_coef=0.01,
        batch_size=64
    )

    # 5. Train
    # --------
    # 100,000 timesteps is a good starting point for this complexity.
    print("\n--- Training Agent (This may take a few minutes) ---")
    model.learn(total_timesteps=100000)
    print("--- Training Complete ---")

    # 6. Save the Brain
    # -----------------
    model.save("ppo_portfolio_agent")
    print("Model saved as 'ppo_portfolio_agent.zip'")

    # 7. Backtest on Unseen Data (Test Set)
    # -------------------------------------
    print("\n--- Starting Backtest on Test Set (2022-2024) ---")
    test_env = PortfolioOptEnv(
        feature_data=test_features, 
        return_data=test_returns,
        initial_balance=10000
    )
    
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        # The model predicts the action based on observation
        # deterministic=True means "use your best guess", don't explore anymore
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        
        # Optional: Print monthly updates or major rebalances
        if test_env.current_step % 30 == 0:
            test_env.render()
            
    # Final Result
    final_val = info['portfolio_value']
    print(f"\nFinal Portfolio Value: ${final_val:.2f}")
    roi = ((final_val - 10000) / 10000) * 100
    print(f"Return on Investment: {roi:.2f}%")

if __name__ == "__main__":
    train_agent()