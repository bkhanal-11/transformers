import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, values, mask=None):
        # Calculate dot products between the query and the key
        matmul_qk = torch.matmul(queries, keys.transpose(-2, -1))

        dk = keys.size()[-1]
        scaled_attention_logits = matmul_qk / (dk ** 0.5)

        if mask is not None:
            scaled_attention_logits += (1. - mask) * -1e9 
        
        # Apply softmax function to obtain attention weights
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # Apply dropout to the attention weights
        attention_weights_dropout = self.dropout(attention_weights)

        output = torch.matmul(attention_weights_dropout, values)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.attention = ScaledDotProductAttention(dropout_rate)

        self.num_heads = num_heads
        self.d_head = d_model // num_heads # key_dim = value_dim = d_head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Apply linear transformations to the inputs
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        # Reshape the inputs for appropriate attention calculation
        query = query.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)  # (batch_size, num_heads, query_seq_len, d_head)
        key = key.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)  # (batch_size, num_heads, key_seq_len, d_head)
        value = value.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)  # (batch_size, num_heads, value_seq_len, d_head)

        context, attention_weights = self.attention(query, key, value, mask)

        # Concatenate the output of the heads and apply a linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # (batch_size, query_seq_len, d_model)
        output = self.W_o(context)  # (batch_size, query_seq_len, d_model)

        return output, attention_weights
