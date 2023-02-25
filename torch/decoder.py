import torch
import torch.nn as nn
import math

from utils import PositionalEncoding, FullyConnected
from mha import MultiHeadAttention

class DecoderLayer(nn.Module):
    """
    The decoder layer is composed by an masked multi-head self-attention mechanism,
    followed by a multi-head attention mechanism to the output of the encoder and a 
    simple, position-wise fully connected feed-forward network. This architecture 
    includes residual connections around all of the three sub-layers, followed by 
    layer normalization.
    """
    def __init__(self, num_heads, d_model, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self.masked_mha = MultiHeadAttention(num_heads=num_heads,
                                             d_model=d_model,
                                             dropout_rate=dropout_rate)

        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      d_model=d_model,
                                      dropout_rate=dropout_rate)

        self.ffn = FullyConnected(embedding_dim=d_model,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.layernorm2 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.layernorm3 = nn.LayerNorm(d_model, eps=layernorm_eps)

        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for the Decoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            encoder_output -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
            look_ahead_mask -- Mask to ensure that the decoder does not attend to subsequent positions
            padding_mask -- Boolean mask to ensure that the padding is not treated as part of the input
        Returns:
            decoder_layer_out -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            attn_weights_block1 -- Tensor of shape (batch_size, num_heads, target_seq_len, target_seq_len)
            attn_weights_block2 -- Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        # masked multi-head self-attention
        mha_output1, attn_weights_block1 = self.masked_mha(x, x, x, look_ahead_mask) # (batch_size, target_seq_len, fully_connected_dim)

        # skip connection
        skip_x_attention1 = self.layernorm1(x + mha_output1)

        # multi-head attention on encoder output
        mha_output2, attn_weights_block2 = self.mha(skip_x_attention1, encoder_output, encoder_output, padding_mask) # (batch_size, target_seq_len, fully_connected_dim)

        # skip connection
        skip_x_attention2 = self.layernorm2(skip_x_attention1 + mha_output2)

        # pass the output of the multi-head attention layer through a ffn
        ffn_output = self.ffn(skip_x_attention2)

        # apply dropout layer to ffn output during training
        ffn_output = self.dropout_ffn(ffn_output)

        # apply layer normalization on sum of the output from multi-head attention (skip connection) and ffn output to get the output of the decoder layer
        decoder_layer_out = self.layernorm3(skip_x_attention2 + ffn_output)

        return decoder_layer_out, attn_weights_block1, attn_weights_block2

class Decoder(nn.Module):
    """
    The Decoder consists of N layers of DecoderLayer.
    """
    def __init__(self, num_layers, num_heads, d_model, fully_connected_dim, target_vocab_size,
                 maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding)
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(num_heads=num_heads,
                         d_model=d_model,
                         fully_connected_dim=fully_connected_dim,
                         dropout_rate=dropout_rate,
                         layernorm_eps=layernorm_eps)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for the Decoder.
        
        Arguments:
            x -- Tensor of shape (batch_size, target_seq_len)
            enc_output -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
            look_ahead_mask -- Boolean mask for the target sequence
            padding_mask -- Boolean mask for the input sequence
            
        Returns:
            decoder_output -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            attention_weights -- Dictionary of attention weights for each decoder layer
        """
        x = self.embedding(x) # (batch_size, target_seq_len, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model).float())
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        attention_weights = {}
        
        for i in range(self.num_layers):
            x, attn_wt_1,  attn_wt_2 = self.decoder_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer_{i+1}'] = attn_wt_1
            # attention_weights[f'decoder_layer_{i+1}'] = attn_wt_2
        
        return x, attention_weights
