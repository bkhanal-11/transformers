import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Apply linear transformations to the inputs
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)  # (batch_size, num_heads, query_seq_len, d_head)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)  # (batch_size, num_heads, key_seq_len, d_head)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)  # (batch_size, num_heads, value_seq_len, d_head)

        # Calculate dot products between the query and the key
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)  # (batch_size, num_heads, query_seq_len, key_seq_len)

        if mask is not None:
            scores += (1. - mask) * -1e9 

        # Apply softmax function to obtain attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, query_seq_len, key_seq_len)

        # Apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights)

        # Multiply the attention weights by the value
        context = torch.matmul(attention_weights, value)  # (batch_size, num_heads, query_seq_len, d_head)

        # Concatenate the output of the heads and apply a linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # (batch_size, query_seq_len, d_model)
        output = self.output_linear(context)  # (batch_size, query_seq_len, d_model)

        return output, attention_weights
