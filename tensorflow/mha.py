import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, dropout_prob, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = Dropout(dropout_prob)
 
    def call(self, queries, keys, values, mask=None):
        matmul_qk = tf.matmul(queries, keys.T)

        dk = keys.shape[0]
        scaled_attention_logits = tf.divide(matmul_qk, np.sqrt(dk))

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (1. - mask) * -1e9 

        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.keras.activations.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, values)  # (..., seq_len_q, depth_v)

        return output, attention_weights

# Implementing the Multi-Head Attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, h, d_k, d_v, d_model, dropout_prob, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = ScaledDotProductAttention(dropout_prob)  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_q = Dense(d_k)
        self.W_k = Dense(d_k)
        self.W_v = Dense(d_v)
        self.W_o = Dense(d_model)
 
    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.d_k))
        return x
 
    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries, keys, values to be able to compute all heads in parallel
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
 
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped, _ = self.attention(q_reshaped, k_reshaped, v_reshaped, mask, self.dropout_prob)
 
        # Rearrange back the output into concatenated form
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)

        return self.W_o(output)
    