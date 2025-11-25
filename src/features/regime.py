import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class RegimeClassifier:
    """
    Classifies the market into distinct regimes (e.g., Calm Bull, Volatile Bear)
    using a Hybrid Unsupervised/Supervised approach.
    
    1. Unsupervised (GMM): Clusters historical returns/volatility to find 'Ground Truth' regimes.
    2. Supervised (XGBoost): Learns to predict these regimes using VIX, Yields, and Trends.
    """
    def __init__(self, data_path, model_path="models/extractors/regime_xgb.json"):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.gmm = None
        self.n_regimes = 3  # 0: Calm Bull, 1: High Vol/Bear, 2: Sideways
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def load_and_prep_data(self):
        print("Loading data for Regime Classification...")
        df = pd.read_csv(self.data_path, index_col=0)
        
        # 1. Feature Engineering (The Inputs for XGBoost)
        # We need to create the 'Macro' and 'Technical' features
        
        # Macro Features (Assumes ^VIX and ^TNX were fetched in pipeline)
        # Note: If VIX/TNX names differ in your CSV, adjust here. 
        # Pipeline usually saves them as 'VIX_Close', 'TNX_Close'
        if 'VIX_Close' in df.columns:
            df['feature_vix'] = df['VIX_Close']
        else:
            # Fallback if VIX missing: Calculate rolling vol of SPY
            df['feature_vix'] = df['SPY_Close'].pct_change().rolling(21).std() * 100

        if 'TNX_Close' in df.columns:
            df['feature_yield'] = df['TNX_Close']
        else:
            df['feature_yield'] = 0 # Placeholder if missing

        # Trend Features (SPY)
        df['spy_pct'] = df['SPY_Close'].pct_change()
        df['spy_vol_21'] = df['spy_pct'].rolling(window=21).std()
        
        # Simple Moving Averages ratio (Trend Strength)
        df['sma_50'] = df['SPY_Close'].rolling(window=50).mean()
        df['sma_200'] = df['SPY_Close'].rolling(window=200).mean()
        df['feature_trend'] = df['sma_50'] / df['sma_200']
        
        # RSI (Momentum)
        delta = df['SPY_Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['feature_rsi'] = 100 - (100 / (1 + rs))

        df.dropna(inplace=True)
        return df

    def create_labels(self, df):
        """
        Uses Gaussian Mixture Models to cluster data into regimes based on 
        Forward Returns and Volatility. This creates the 'Ground Truth'.
        """
        print("Clustering data to discover regimes...")
        
        # We look ahead to define what the regime *was*
        # We want to predict: "Is the market ABOUT to be volatile?"
        X_subset = df[['spy_pct', 'spy_vol_21']].copy()
        
        # Scale for GMM
        X_scaled = (X_subset - X_subset.mean()) / X_subset.std()
        
        # Fit GMM
        self.gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
        self.gmm.fit(X_scaled)
        
        # Get labels
        labels = self.gmm.predict(X_scaled)
        df['regime_label'] = labels
        
        # Interpretation check:
        # Usually, the regime with high Volatility is "Bear/Crash"
        # We can analyze the means to map ID to meaningful name if needed
        # For now, the RL agent just needs to know they are *different* states.
        
        print(f"Regime Distribution: {np.bincount(labels)}")
        return df

    def train(self):
        df = self.load_and_prep_data()
        df = self.create_labels(df)
        
        # Define Features (X) and Target (y)
        # We use current features to predict the regime
        features = ['feature_vix', 'feature_yield', 'feature_trend', 'feature_rsi']
        X = df[features]
        y = df['regime_label']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Init XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
        
        print("Training XGBoost Classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Regime Classification Accuracy: {acc:.2f}")
        print(classification_report(y_test, preds))
        
        # Save
        self.model.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict_regime(self, current_data_dict):
        """
        Inference: Takes a dictionary of current values and returns Regime ID (0, 1, 2)
        """
        if self.model is None:
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            
        # Create a 1-row DataFrame matching training features
        # Expects: {'vix': 15.0, 'yield': 4.0, 'trend': 1.05, 'rsi': 55}
        input_data = pd.DataFrame([current_data_dict])
        
        # Ensure column order matches training
        # We might need to ensure scaling matches if we used scaling in XGBoost (XGB is usually robust to unscaled)
        # Mapping dict keys to feature names:
        features = ['feature_vix', 'feature_yield', 'feature_trend', 'feature_rsi']
        # You'll need to pass these specific keys from your Env
        
        regime_id = self.model.predict(input_data[features])[0]
        return int(regime_id)

if __name__ == "__main__":
    # Test Run
    # Ensure raw data exists first
    data_path = "data/raw/hybrid_aaa_raw_data.csv"
    if os.path.exists(data_path):
        rc = RegimeClassifier(data_path=data_path)
        rc.train()
    else:
        print("Run src/data/pipeline.py first!")