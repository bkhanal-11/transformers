import torch
import torch.nn as nn
import math

class FullyConnected(nn.Module):
    def __init__(self, embedding_dim, fully_connected_dim):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, fully_connected_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fully_connected_dim, embedding_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        
        return self.fc2(x)

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        self.scale = torch.nn.Parameter(torch.ones(self.hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        
        return self.scale * normalized + self.bias

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = self.get_positional_encoding(max_seq_len)

    def get_positional_encoding(self, max_seq_len):
        # Generate positional encoding matrix
        pe = torch.zeros(max_seq_len, self.d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        return pe

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        x = x * math.sqrt(self.d_model)
        x = x + pe
        
        return x

def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids -- (n, m) matrix
    
    Returns:
        mask -- (n, 1, m) binary tensor
    """
    # Create a matrix mask for the padding cells
    seq = 1 - torch.eq(decoder_token_ids, 0).float()
  
    # Add extra dimensions to add the padding to the attention logits
    return seq.unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(sequence_length):
    """
    Returns a lower triangular matrix filled with ones
    
    Arguments:
        sequence_length -- matrix size
    
    Returns:
        mask -- (size, size) tensor
    """
    # Create a lower triangular matrix filled with ones
    mask = torch.tril(torch.ones(sequence_length, sequence_length))
    
    return mask
