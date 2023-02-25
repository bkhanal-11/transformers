import tensorflow as tf
from tensorflow.keras.layers import Dropout

import numpy as np

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, dropout,  **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = Dropout(dropout)
 
    def call(self, queries, keys, values, mask=None):
        matmul_qk = tf.matmul(queries, keys, transpose_b=True)

        dk = tf.cast(tf.shape(keys)[-1], tf.float32)
        scaled_attention_logits = tf.divide(matmul_qk, np.sqrt(dk))

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (1. - mask) * -1e9 

        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.keras.activations.softmax(scaled_attention_logits, axis=-1)
        attention_weights_dropout = self.dropout(attention_weights)

        output = tf.matmul(attention_weights_dropout, values)  # (..., seq_len_q, depth_v)

        return output, attention_weights

# Implementing the Multi-Head Attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, key_dim, value_dim, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = ScaledDotProductAttention(dropout)  # Scaled dot product attention
        self.heads = num_heads  # Number of attention heads to use
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.d_model = d_model

        initializer = tf.keras.initializers.GlorotUniform(seed=3)
        self.W_q = tf.Variable(initializer(shape=(num_heads, d_model, key_dim)), trainable=True)
        self.W_k = tf.Variable(initializer(shape=(num_heads, d_model, key_dim)), trainable=True)
        self.W_v = tf.Variable(initializer(shape=(num_heads, d_model, value_dim)), trainable=True)
        self.W_o = tf.Variable(initializer(shape=(num_heads * value_dim, d_model)), trainable=True)
 
    def _reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads,     , -1)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, key_dim)
            x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[1]*tf.shape(x)[3]))
        
        return x
 
    def call(self, queries, keys, values, mask=None, return_attention_scores=False):
        # Rearrange the queries, keys, values to be able to compute all heads in parallel
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
        q_reshaped = self._reshape_tensor(tf.experimental.numpy.dot(queries, self.W_q), self.heads, True)
 
        k_reshaped = self._reshape_tensor(tf.experimental.numpy.dot(keys, self.W_k), self.heads, True)
        
        v_reshaped = self._reshape_tensor(tf.experimental.numpy.dot(values, self.W_v), self.heads, True)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped, attention_score = self.attention(q_reshaped, k_reshaped, v_reshaped, mask)
 
        # Rearrange back the output into concatenated form
        # Resulting tensor shape: (batch_size, input_seq_length, value_dim)
        output = self._reshape_tensor(o_reshaped, self.heads, False)
        
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)

        if return_attention_scores:
            return tf.experimental.numpy.dot(output, self.W_o), attention_score

        return tf.experimental.numpy.dot(output, self.W_o)
    