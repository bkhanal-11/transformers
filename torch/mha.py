import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, mask=None):
        matmul_qk = torch.matmul(queries, keys.transpose(-2, -1))

        dk = keys.size()[-1]
        scaled_attention_logits = matmul_qk / (dk ** 0.5)

        if mask is not None:
            scaled_attention_logits += (1. - mask) * -1e9 

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        attention_weights_dropout = self.dropout(attention_weights)

        output = torch.matmul(attention_weights_dropout, values)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, key_dim, value_dim, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = ScaledDotProductAttention(dropout)
        self.heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.d_model = d_model

        self.W_q = nn.Linear(self.d_model , self.key_dim ,bias=False)
        self.W_k = nn.Linear(self.d_model , self.key_dim ,bias=False)
        self.W_v = nn.Linear(self.d_model , self.value_dim ,bias=False)
        self.W_o = nn.Linear(self.heads * self.value_dim ,self.d_model) 

    def _reshape_tensor(self, x, heads, flag):
        if flag:
            x = x.view(x.size(0), x.size(1), heads, -1).transpose(1, 2)
        else:
            x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.heads * self.value_dim)
        
        return x

    def forward(self, queries, keys, values, mask=None, return_attention_scores=False):
        q_reshaped = self._reshape_tensor(self.W_q(queries), self.heads, True)
        k_reshaped = self._reshape_tensor(self.W_k(keys), self.heads, True)
        v_reshaped = self._reshape_tensor(self.W_v(values), self.heads, True)

        o_reshaped, attention_score = self.attention(q_reshaped, k_reshaped, v_reshaped, mask)

        output = self._reshape_tensor(o_reshaped, self.heads, False)
        
        if return_attention_scores:
            return self.W_o(output), attention_score

        return self.W_o(output)