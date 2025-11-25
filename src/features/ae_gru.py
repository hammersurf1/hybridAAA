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
        # Compresses the input sequence into the final hidden state
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # --- Decoder ---
        # Reconstructs the original sequence from the latent representation
        self.decoder = nn.GRU(
            input_size=input_dim, # We feed the previous step's output as input
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Map hidden state back to input dimension (Reconstruction)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 1. Encode
        # _, hidden = self.encoder(x)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # We take the last layer's hidden state as the "Context Vector" (Z)
        _, hidden = self.encoder(x)
        z = hidden[-1] # Shape: (batch_size, hidden_dim)
        
        # 2. Decode
        # We need to reconstruct the sequence step-by-step
        reconstructed_outputs = []
        
        # Initialize decoder input with zeros (or the first item of the sequence)
        decoder_input = torch.zeros(batch_size, 1, self.input_dim).to(x.device)
        decoder_hidden = hidden # Use encoder's final hidden state to start decoder
        
        for t in range(seq_len):
            # Run one step of decoder
            out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Map to input dimension
            prediction = self.output_layer(out) # Shape: (batch_size, 1, input_dim)
            reconstructed_outputs.append(prediction)
            
            # Teacher Forcing: Use the *actual* next step as input for the next iteration?
            # For pure autoencoding, we usually feed our own prediction back in, 
            # but for training stability, we often just reconstruct the vector directly 
            # from the hidden state repetition. 
            # Here, we keep it simple: Just feeding the prediction forward.
            decoder_input = prediction

        reconstructed_outputs = torch.cat(reconstructed_outputs, dim=1) # (batch_size, seq_len, input_dim)
        
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
        
        # Ensure model directory exists
        os.makedirs("models/extractors", exist_ok=True)

    def load_and_scale_data(self):
        print("Loading data for Feature Engineering...")
        df = pd.read_csv(self.data_path, index_col=0)
        
        # We only want the Asset Price/Volume columns for the Autoencoder
        # We exclude Macro and Sentiment (those go directly to the Agent or Regime Classifier)
        cols_to_use = [c for c in df.columns if "Close" in c or "Volume" in c]
        data = df[cols_to_use].values
        
        # Normalize (Min-Max Scaling is common for Neural Networks)
        # Simple implementation for clarity; use sklearn in production
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0) + 1e-8 # Avoid div by zero
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
        
        # Convert to PyTorch Tensors
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
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")

        # Save the model
        torch.save(self.model.state_dict(), "models/extractors/ae_gru.pt")
        print("Model saved to models/extractors/ae_gru.pt")

    def extract_features(self, current_window):
        """
        Inference method: Input a single window (numpy array) -> Output Latent Vector Z
        """
        if self.model is None:
            # Re-initialize structure to load weights
            input_dim = current_window.shape[1]
            self.model = AEGRU(input_dim=input_dim, hidden_dim=self.hidden_dim).to(self.device)
            self.model.load_state_dict(torch.load("models/extractors/ae_gru.pt"))
            self.model.eval()
            
        # Normalize
        norm_window = (current_window - self.data_mean) / self.data_std
        tensor_x = torch.FloatTensor(norm_window).unsqueeze(0).to(self.device) # Add batch dim
        
        with torch.no_grad():
            _, z = self.model(tensor_x)
            
        return z.cpu().numpy().flatten()

if __name__ == "__main__":
    # Test Run
    fe = FeatureEngineer(data_path="data/raw/hybrid_aaa_raw_data.csv")
    fe.train()