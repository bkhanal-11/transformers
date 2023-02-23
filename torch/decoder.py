import torch.nn as nn
from utils import FullyConnected
from mha import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, embedding_dim, value_dim, fully_connected_dim,
                  dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self.mha1 = MultiHeadAttention(num_heads=num_heads,
                                      d_model=d_model,
                                      key_dim=embedding_dim,
                                      value_dim=value_dim,
                                      dropout=dropout_rate)

        self.mha2 = MultiHeadAttention(num_heads=num_heads,
                                      d_model=d_model,
                                      key_dim=embedding_dim,
                                      value_dim=value_dim,
                                      dropout=dropout_rate)

        self.ffn = FullyConnected(embedding_dim=d_model,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.layernorm2 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.layernorm3 = nn.LayerNorm(d_model, eps=layernorm_eps)

        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, fully_connected_dim)

        # BLOCK 1
        # calculate self-attention and return attention scores as attn_weights_block1.
        mult_attn_out1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)

        # apply layer normalization (layernorm1) to the sum of the attention output and the input
        Q1 = self.layernorm1(mult_attn_out1 + x)

        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output.
        mult_attn_out2, attn_weights_block2 = self.mha2(Q1, enc_output, enc_output, padding_mask, return_attention_scores=True)

        # apply layer normalization (layernorm2) to the sum of the attention output and the output of the first block
        mult_attn_out2 = self.layernorm2(mult_attn_out2 + Q1)  # (batch_size, target_seq_len, fully_connected_dim)

        # BLOCK 3
        # pass the output of the second block through a ffn
        ffn_output = self.ffn(mult_attn_out2)

        # apply a dropout layer to the ffn output
        ffn_output = self.dropout_ffn(ffn_output)

        # apply layer normalization (layernorm3) to the sum of the ffn output and the output of the second block
        out3 = self.layernorm3(ffn_output + mult_attn_out2)

        return out3, attn_weights_block1, attn_weights_block2

class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, embedding_dim, value_dim, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.dec_layers = nn.ModuleList([DecoderLayer(num_heads=num_heads,
                                                      d_model=d_model,
                                                      embedding_dim=self.embedding_dim,
                                                      value_dim=value_dim,
                                                      fully_connected_dim=fully_connected_dim,
                                                      dropout_rate=dropout_rate,
                                                      layernorm_eps=layernorm_eps) 
                                         for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attention_weights = {}

        # use a for loop to pass x through a stack of decoder layers and update attention_weights
        for i in range(self.num_layers):
            # pass x and the encoder output through a stack of decoder layers and save the attention weights of block 1 and 2
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

            # update attention_weights dictionary with the attention weights of block 1 and block 2
            attention_weights[f"decoder_layer{i + 1}_block1_self_att"] = block1
            attention_weights[f"decoder_layer{i + 1}_block2_decenc_att"] = block2

        return x, attention_weights
