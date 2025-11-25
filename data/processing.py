import pandas as pd
import numpy as np
import os
from tqdm import tqdm  # pip install tqdm
from src.features.ae_gru import FeatureEngineer
from src.features.regime import RegimeClassifier

def preprocess_data(raw_data_path="data/raw/hybrid_aaa_raw_data.csv", 
                   output_path="data/processed/enriched_data.csv"):
    """
    Runs the heavy Feature Extraction (AE-GRU) and Regime Classification (XGBoost)
    once, offline, and saves the result. This speeds up RL training significantly.
    """
    
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data not found at {raw_data_path}. Run pipeline.py first.")
        return

    print(f"Starting Preprocessing: {raw_data_path} -> {output_path}")
    
    # 1. Load Data
    df = pd.read_csv(raw_data_path, index_col=0)
    timestamps = df.index
    
    # 2. Initialize Models
    # Note: These assume models are trained and saved in 'models/extractors/'
    # If not, the classes usually try to load default paths or you must run training scripts first.
    print("Loading Feature Extractors...")
    fe = FeatureEngineer(raw_data_path, seq_len=60, hidden_dim=32)
    rc = RegimeClassifier(raw_data_path)
    
    # 3. Containers for new features
    latent_features = []
    regime_labels = []
    valid_indices = []
    
    # 4. Iterate (Sliding Window)
    # We need 60 days of history for AE-GRU, so we must start at index 60
    seq_len = 60
    
    print(f"Generating features for {len(df) - seq_len} timesteps...")
    
    # Using tqdm for a progress bar
    for i in tqdm(range(seq_len, len(df))):
        # --- AE-GRU Inference ---
        # Window: [i-60 : i]
        # We need to grab the specific columns used by AE-GRU (Asset Closes/Vols)
        # Based on pipeline.py, the first 8 columns are asset data (SPY_Close, SPY_Vol, etc.)
        window = df.iloc[i-seq_len : i, 0:8].values
        
        # Extract Z_t (Shape: 32,)
        z_t = fe.extract_features(window)
        latent_features.append(z_t)
        
        # --- Regime Inference ---
        # Current row for Macro context
        row = df.iloc[i]
        
        # Construct input dict for XGBoost
        # Note: Ensure these fallback values match your regime.py training logic
        regime_input = {
            'feature_vix': row.get('VIX_Close', 20.0),
            'feature_yield': row.get('TNX_Close', 4.0),
            'feature_trend': 1.0, # Placeholder if not computed
            'feature_rsi': 50.0   # Placeholder if not computed
        }
        
        regime_id = rc.predict_regime(regime_input)
        regime_labels.append(regime_id)
        
        # Keep track of the timestamp for this row
        valid_indices.append(timestamps[i])

    # 5. Compile Final DataFrame
    # Create columns for Latent Features: Z_0, Z_1, ... Z_31
    latent_cols = [f"Z_{k}" for k in range(32)]
    df_latent = pd.DataFrame(latent_features, columns=latent_cols, index=valid_indices)
    
    df_regime = pd.DataFrame(regime_labels, columns=['Regime'], index=valid_indices)
    
    # Align original data (Returns, Sentiment, Prices) with the valid indices
    df_aligned = df.loc[valid_indices]
    
    # Concatenate: [Original Data | Latent Features | Regime]
    final_df = pd.concat([df_aligned, df_latent, df_regime], axis=1)
    
    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path)
    print("-" * 50)
    print(f"Preprocessing Complete.")
    print(f"Enriched Data Shape: {final_df.shape}")
    print(f"Saved to: {output_path}")
    print("-" * 50)

if __name__ == "__main__":
    preprocess_data()