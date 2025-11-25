import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from src.envs.portfolio_env import PortfolioEnv

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])[0]
        if "portfolio_value" in infos:
            self.logger.record("custom/portfolio_value", infos["portfolio_value"])
        if "return" in infos:
            self.logger.record("custom/daily_return", infos["return"])
        if "regime" in infos:
            self.logger.record("custom/regime", infos["regime"])
        return True

class HierarchicalManager:
    """
    The Master Orchestrator for the Hybrid AAA HRL System.
    
    Architecture:
    1. High-Level Controller (HLC): PPO Agent -> Outputs Strategic Allocation (Target Weights).
    2. Low-Level Controller (LLC): DDPG/Execution Agent -> Outputs Tactical Orders (Buy/Sell).
    
    For a $10k account, the LLC can be a 'Pass-Through' (Direct Execution) or a cost-minimizer.
    This Manager coordinates the data flow between them.
    """
    def __init__(self, model_name="hybrid_aaa_v1", tensorboard_log="logs/hrl_system"):
        self.model_name = model_name
        self.model_path = f"models/final/{model_name}"
        self.tensorboard_log = tensorboard_log
        
        # Directories
        os.makedirs("models/final", exist_ok=True)
        os.makedirs("models/checkpoints", exist_ok=True)
        os.makedirs(tensorboard_log, exist_ok=True)
        
        # Components
        self.hlc_model = None
        self.llc_model = None # Placeholder for DDPG execution agent
        
        # Environment Containers
        self.train_env = None
        self.test_env = None

    def setup_hlc(self, learning_rate=3e-4):
        """Initializes the Strategic Policy (PPO)."""
        print(f"Initializing HLC (Strategic Layer): {self.model_name}_HLC")
        
        # 1. Create HLC Environment (Enriched Data)
        # norm_reward=True is critical for DSR optimization stability
        env = DummyVecEnv([lambda: PortfolioEnv(data_path="data/processed/enriched_data.csv")])
        self.train_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # 2. Define PPO Model
        self.hlc_model = PPO(
            "MlpPolicy",
            self.train_env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=self.tensorboard_log,
            device="auto"
        )

    def train_hlc(self, total_timesteps=100000):
        """
        Phase 1: Train the Strategic Allocator on Enriched Data.
        """
        if self.hlc_model is None:
            self.setup_hlc()
            
        print("Starting HRL Phase 1: Training High-Level Controller...")
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path="models/checkpoints/",
            name_prefix=f"{self.model_name}_HLC"
        )
        
        self.hlc_model.learn(
            total_timesteps=total_timesteps, 
            callback=[checkpoint_callback, TensorboardCallback()]
        )
        
        # Save HLC
        hlc_path = f"{self.model_path}_HLC"
        self.hlc_model.save(hlc_path)
        self.train_env.save(f"{hlc_path}_vecnorm.pkl")
        print(f"HLC Training Complete. Saved to {hlc_path}")

    def execution_logic(self, target_weights, current_weights):
        """
        The Tactical Layer (LLC) Logic.
        Currently implements a 'Direct Execution' baseline.
        TODO: Replace this with a trained DDPG agent for optimal execution.
        """
        # For now, we assume we move directly to target (Monolithic behavior)
        # A DDPG agent would break this into smaller chunks over the day to reduce slippage.
        execution_action = target_weights 
        return execution_action

    def run_hierarchical_loop(self):
        """
        Runs the Full HRL Loop (Strategy -> Execution).
        Validates the Hybrid AAA Architecture.
        """
        print(f"Running HRL System Validation: {self.model_name}")
        
        # 1. Setup Test Environment
        raw_env = DummyVecEnv([lambda: PortfolioEnv(data_path="data/processed/enriched_data.csv")])
        hlc_path = f"{self.model_path}_HLC"
        
        # Load Normalization Stats
        env = VecNormalize.load(f"{hlc_path}_vecnorm.pkl", raw_env)
        env.training = False 
        env.norm_reward = False
        
        # Load Brain
        self.hlc_model = PPO.load(hlc_path)
        
        obs = env.reset()
        done = False
        portfolio_values = []
        
        while not done:
            # --- LEVEL 1: STRATEGY (HLC) ---
            # Input: Market State (Regime + Features)
            # Output: Target Portfolio Allocation
            target_weights, _ = self.hlc_model.predict(obs, deterministic=True)
            
            # --- LEVEL 2: TACTICS (LLC) ---
            # Input: Target vs Current Weights
            # Output: Execution Orders
            # (Currently using Direct Execution baseline until DDPG is built)
            final_action = self.execution_logic(target_weights, current_weights=None)
            
            # --- EXECUTION ---
            obs, reward, done, info = env.step(final_action)
            
            # Logging
            current_value = info[0]['portfolio_value']
            portfolio_values.append(current_value)
            
            if len(portfolio_values) % 100 == 0:
                print(f"HRL Step {len(portfolio_values)}: Portfolio Value ${current_value:.2f}")

        print(f"HRL Validation Complete. Final Value: ${portfolio_values[-1]:.2f}")
        return portfolio_values

if __name__ == "__main__":
    # Initialize the Manager
    system = HierarchicalManager()
    
    # Train the Strategic Layer (HLC)
    system.train_hlc(total_timesteps=50000)
    
    # Validate the full hierarchy
    # system.run_hierarchical_loop()