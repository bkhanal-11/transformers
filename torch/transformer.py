import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_encoder_layers, num_heads, d_model, dff, input_vocab_size, max_seq_len, dropout_rate)

        self.decoder = Decoder(num_decoder_layers, num_heads, d_model, dff, target_vocab_size, max_seq_len, dropout_rate)

        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, look_ahead_mask=None, dec_padding_mask=None, enc_padding_mask=None):
        enc_output = self.encoder(inp, enc_padding_mask)

        dec_output, attention_weights = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
