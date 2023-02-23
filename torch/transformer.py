import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from utils import PositionalEmbedding

class Transformer(nn.Module):
    """
    Complete transformer with an Encoder and a Decoder.
    """
    def __init__(self, num_layers, num_heads, d_model, embedding_dim, value_dim, fully_connected_dim, input_vocab_size, 
                 max_positional_encoding, dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.positional_encoding = PositionalEmbedding(max_positional_encoding, d_model)
        
        self.dropout_encoding_input = nn.Dropout(dropout_rate)
        self.dropout_decoding_input = nn.Dropout(dropout_rate)

        self.encoder = Encoder(num_layers=num_layers,
                               num_heads=num_heads,
                               d_model=d_model,
                               embedding_dim=embedding_dim,
                               value_dim=value_dim,
                               fully_connected_dim=fully_connected_dim,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers, 
                               num_heads=num_heads,
                               d_model=d_model,
                               embedding_dim=embedding_dim,
                               value_dim=value_dim,
                               fully_connected_dim=fully_connected_dim,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.final_layer = nn.Linear(d_model, input_vocab_size)
    
    def forward(self, X, y, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):
        """
        Forward pass for the entire Transformer
        Arguments:
            X -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
                              An array of the indexes of the words in the input sentence
            y -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
                              An array of the indexes of the words in the output sentence
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            enc_padding_mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
            look_ahead_mask -- Boolean mask for the target_input
            dec_padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            final_output -- Describe me
            attention_weights - Dictionary of tensors containing all the attention weights for the decoder
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        
        """
        x = self.embedding(X)
        x = self.positional_encoding(x)
        x = self.dropout_encoding_input(x)
        # call self.encoder with the appropriate arguments to get the encoder output
        enc_output = self.encoder(x, enc_padding_mask)  # (batch_size, inp_seq_len, fully_connected_dim)

        y = self.embedding(y)
        y = y + self.positional_encoding
        y = self.dropout_decoding_input(y)
        
        # call self.decoder with the appropriate arguments to get the decoder output
        # dec_output.shape == (batch_size, tar_seq_len, fully_connected_dim)
        dec_output, attention_weights = self.decoder(y, enc_output, look_ahead_mask, dec_padding_mask)
        
        # pass decoder output through a linear layer and softmax
        final_output =  torch.matmul(self.embedding, dec_output.transpose(-2, -1))

        final_output = self.final_layer(final_output) # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

if __name__ == "__main__":
    num_layers, num_heads, d_model, embedding_dim, fully_connected_dim = 6, 8, 512, 64, 2048
    input_vocab_size, max_positioning_embedding = 29, 11
    batch_size = 3

    transformer = Transformer(num_layers=num_layers, 
                               num_heads=num_heads, 
                               d_model=d_model, 
                               embedding_dim=embedding_dim,
                               value_dim=embedding_dim,
                               fully_connected_dim=fully_connected_dim, 
                               input_vocab_size=input_vocab_size, 
                               max_positional_encoding=max_positioning_embedding)

    x = torch.LongTensor(batch_size, max_positioning_embedding, input_vocab_size)
    y = torch.LongTensor(batch_size, max_positioning_embedding, input_vocab_size)

    pred, _ = transformer(x, y)
    print(pred.shape)

    print(transformer)