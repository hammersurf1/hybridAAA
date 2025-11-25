import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from src.envs.portfolio_env import PortfolioEnv

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])[0]
        
        # Log standard metrics
        if "portfolio_value" in infos:
            self.logger.record("custom/portfolio_value", infos["portfolio_value"])
        if "return" in infos:
            self.logger.record("custom/daily_return", infos["return"])
        if "regime" in infos:
            self.logger.record("custom/regime", infos["regime"])
            
        # Log X-Ray metrics
        if "cash_weight" in infos:
            self.logger.record("custom/cash_weight", infos["cash_weight"])
        if "turnover" in infos:
            self.logger.record("custom/turnover", infos["turnover"])
            
        return True

class HLCAgent:
    def __init__(self, model_name="hybrid_aaa_ppo", tensorboard_log="logs/ppo_hlc"):
        self.model_name = model_name
        self.model_path = f"models/final/{model_name}"
        self.tensorboard_log = tensorboard_log
        
        os.makedirs("models/final", exist_ok=True)
        os.makedirs("models/checkpoints", exist_ok=True)
        os.makedirs(tensorboard_log, exist_ok=True)

    def train(self, total_timesteps=100000):
        print(f"Initializing PPO Training: {self.model_name}")
        
        env = DummyVecEnv([lambda: PortfolioEnv(data_path="data/processed/enriched_data.csv")])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,          # REDUCED: Less random shaking, more stability
            tensorboard_log=self.tensorboard_log,
            device="auto"
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path="models/checkpoints/",
            name_prefix=self.model_name
        )
        custom_logger = TensorboardCallback()
        
        print("Starting Training Loop...")
        model.learn(
            total_timesteps=total_timesteps, 
            callback=[checkpoint_callback, custom_logger]
        )
        
        model.save(self.model_path)
        env.save(f"{self.model_path}_vecnorm.pkl")
        print(f"Training Complete. Model saved to {self.model_path}")

    def test(self):
        # ... (Test logic remains the same)
        pass

if __name__ == "__main__":
    agent = HLCAgent()
    agent.train(total_timesteps=100000)