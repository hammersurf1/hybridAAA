import os
import sys

# Ensure the root directory is in the python path
sys.path.append(os.getcwd())

from src.data.pipeline import DataIngestion
from src.features.ae_gru import FeatureEngineer
from src.features.regime import RegimeClassifier
from src.data.processing import preprocess_data
from src.agents.hlc_ppo import HLCAgent

def main():
    print("=== Hybrid AAA Trading System: Master Sequence ===")
    
    # --- Step 1: Data Ingestion ---
    if not os.path.exists("data/raw/hybrid_aaa_raw_data.csv"):
        print("\n[Step 1] Raw Data missing. Starting Download Pipeline...")
        pipeline = DataIngestion()
        pipeline.run_pipeline()
    else:
        print("\n[Step 1] Raw Data found. Skipping download.")

    # --- Step 2: Feature Engineering (Vision & Context) ---
    # 2A. AE-GRU (Vision)
    if not os.path.exists("models/extractors/ae_gru.pt"):
        print("\n[Step 2A] AE-GRU Model missing. Training Feature Extractor...")
        # Note: We use the raw data to train the feature extractor
        fe = FeatureEngineer(data_path="data/raw/hybrid_aaa_raw_data.csv")
        fe.train()
    else:
        print("\n[Step 2A] AE-GRU Model found. Skipping training.")

    # 2B. Regime Classifier (Context)
    if not os.path.exists("models/extractors/regime_xgb.json"):
        print("\n[Step 2B] Regime Classifier missing. Training XGBoost...")
        rc = RegimeClassifier(data_path="data/raw/hybrid_aaa_raw_data.csv")
        rc.train()
    else:
        print("\n[Step 2B] Regime Classifier found. Skipping training.")

    # --- Step 3: Data Enrichment (Optimization) ---
    if not os.path.exists("data/processed/enriched_data.csv"):
        print("\n[Step 3] Enriched Data missing. Running Preprocessing (This may take a minute)...")
        preprocess_data()
    else:
        print("\n[Step 3] Enriched Data found. Skipping preprocessing.")

    # --- Step 4: The Brain (PPO Agent) ---
    print("\n[Step 4] Initializing PPO Agent Training...")
    print("Tensorboard logs will be saved to: logs/ppo_hlc")
    
    agent = HLCAgent()
    # Train for enough timesteps to see convergence
    agent.train(total_timesteps=100000)

if __name__ == "__main__":
    main()