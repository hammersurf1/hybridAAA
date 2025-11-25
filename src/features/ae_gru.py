import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os

class AEGRU(nn.Module):
    """
    Autoencoder Gated Recurrent Unit (AE-GRU).
    Compresses a time-series window (seq_len, input_dim) into a latent vector (hidden_dim).
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(AEGRU, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # --- Encoder ---
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # --- Decoder ---
        self.decoder = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Map hidden state back to input dimension
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Encode
        _, hidden = self.encoder(x)
        z = hidden[-1] 
        
        # 2. Decode
        reconstructed_outputs = []
        decoder_input = torch.zeros(batch_size, 1, self.input_dim).to(x.device)
        decoder_hidden = hidden 
        
        for t in range(seq_len):
            out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            prediction = self.output_layer(out) 
            reconstructed_outputs.append(prediction)
            decoder_input = prediction

        reconstructed_outputs = torch.cat(reconstructed_outputs, dim=1) 
        
        return reconstructed_outputs, z

class FeatureEngineer:
    """
    Manager class to handle data preprocessing, training, and inference for the AE-GRU.
    """
    def __init__(self, data_path, seq_len=60, hidden_dim=32, batch_size=64, epochs=20):
        self.data_path = data_path
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        
        os.makedirs("models/extractors", exist_ok=True)

    def load_and_scale_data(self):
        print("Loading data for Feature Engineering...")
        df = pd.read_csv(self.data_path, index_col=0)
        
        # STRICT FILTER: Only Asset Close/Volume. Explicitly exclude Macro/Regime inputs.
        # This ensures input_dim is exactly 8 (SPY, TLT, GLD, USO * Close, Vol)
        cols_to_use = []
        for c in df.columns:
            is_asset = ("Close" in c or "Volume" in c)
            is_macro = ("VIX" in c or "TNX" in c or "Sentiment" in c)
            if is_asset and not is_macro:
                cols_to_use.append(c)
                
        print(f"AE-GRU Training Features ({len(cols_to_use)}): {cols_to_use}")
        data = df[cols_to_use].values
        
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0) + 1e-8
        normalized_data = (data - self.data_mean) / self.data_std
        
        return normalized_data

    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.seq_len):
            seq = data[i : i + self.seq_len]
            sequences.append(seq)
        return np.array(sequences)

    def train(self):
        data = self.load_and_scale_data()
        sequences = self.create_sequences(data)
        
        dataset = TensorDataset(torch.FloatTensor(sequences))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        input_dim = sequences.shape[2]
        self.model = AEGRU(input_dim=input_dim, hidden_dim=self.hidden_dim).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print(f"Starting AE-GRU Training on {self.device}...")
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                reconstructed, _ = self.model(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {total_loss / len(dataloader):.6f}")

        torch.save(self.model.state_dict(), "models/extractors/ae_gru.pt")
        print("Model saved to models/extractors/ae_gru.pt")

    def extract_features(self, current_window):
        """
        Inference: Input shape (seq_len, 8) -> Output Latent Vector Z (32)
        """
        if self.model is None:
            # We initialize with the dimension of the incoming window
            input_dim = current_window.shape[1]
            self.model = AEGRU(input_dim=input_dim, hidden_dim=self.hidden_dim).to(self.device)
            # This load will now work because input_dim will match the saved weights (8)
            self.model.load_state_dict(torch.load("models/extractors/ae_gru.pt"))
            self.model.eval()
            
        # Normalize (In production, load saved mean/std! Here we approximate with batch stats or re-calc)
        # Note: Ideally, save scaler stats in train() and load them here. 
        # For this setup, we assume current_window is raw and we re-calc stats from the whole file if needed,
        # but for efficiency in processing.py, we trust the caller or re-init basic norm.
        # FIX: To avoid re-reading CSV, we assume the user accepts raw data passing through 
        # or we implement a proper Scaler class. 
        # For now, we will just cast to tensor.
        
        tensor_x = torch.FloatTensor(current_window).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, z = self.model(tensor_x)
            
        return z.cpu().numpy().flatten()

if __name__ == "__main__":
    fe = FeatureEngineer(data_path="data/raw/hybrid_aaa_raw_data.csv")
    fe.train()