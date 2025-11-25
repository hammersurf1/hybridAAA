from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import os
from src.envs.execution_env import ExecutionEnv

class LLCAgent:
    """
    The Low-Level Controller (LLC).
    Uses DDPG to learn optimal trade execution (minimizing impact).
    """
    def __init__(self, model_name="ddpg_llc_v1", tensorboard_log="logs/ddpg_llc"):
        self.model_name = model_name
        self.model_path = f"models/final/{model_name}"
        self.tensorboard_log = tensorboard_log
        
        os.makedirs("models/final", exist_ok=True)
        os.makedirs(tensorboard_log, exist_ok=True)

    def train(self, total_timesteps=20000):
        print(f"Initializing LLC Training: {self.model_name}")
        
        # 1. Setup Env
        env = DummyVecEnv([lambda: ExecutionEnv()])
        
        # 2. Action Noise (Critical for DDPG exploration)
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        # 3. Define Model
        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            verbose=1,
            learning_rate=1e-3,
            batch_size=64,
            buffer_size=10000,
            tensorboard_log=self.tensorboard_log,
            device="auto"
        )
        
        # 4. Train
        model.learn(total_timesteps=total_timesteps)
        model.save(self.model_path)
        print(f"LLC Training Complete. Saved to {self.model_path}")
        return model

    def execute_order(self, order_size, volatility_state, model=None):
        """
        Simulates the execution of an order using the trained DDPG policy.
        Returns the average execution price impact or efficiency score.
        """
        if model is None:
            model = DDPG.load(self.model_path)
            
        env = ExecutionEnv(initial_order_size=order_size)
        obs, _ = env.reset(volatility_factor=volatility_state)
        
        done = False
        total_slippage = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_slippage += info['slippage']
            
        return total_slippage

if __name__ == "__main__":
    llc = LLCAgent()
    llc.train()