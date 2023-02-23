import torch
import torch.nn as nn
import numpy as np
import math

def FullyConnected(embedding_dim, fully_connected_dim):
    return nn.Sequential(
        nn.Linear(embedding_dim, fully_connected_dim),
        nn.ReLU(),
        nn.Linear(fully_connected_dim, embedding_dim)
    )


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        """
        Args:
            max_seq_len: length of input sequence
            d_model: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model

        PE = torch.zeros(max_seq_len, self.d_model)
        for pos in range(max_seq_len):
            for i in range(0, self.d_model, 2):
                PE[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.d_model)))
                PE[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.d_model)))
        PE = PE.unsqueeze(0)
        self.register_buffer('PE', PE)


    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.PE[:,:seq_len], requires_grad=False)
        
        return x

def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids -- (n, m) matrix
    
    Returns:
        mask -- (n, 1, m) binary tensor
    """
    seq = 1 - torch.eq(decoder_token_ids, 0).float()
  
    # add extra dimensions to add the padding to the attention logits.
    return seq.unsqueeze(1)

def create_look_ahead_mask(sequence_length):
    """
    Returns a lower triangular matrix filled with ones
    
    Arguments:
        sequence_length -- matrix size
    
    Returns:
        mask -- (size, size) tensor
    """
    mask = torch.tril(torch.ones(sequence_length, sequence_length))
    
    return mask
