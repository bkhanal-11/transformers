import tensorflow as tf

from encoder import Encoder
from decoder import Decoder

class Transformer(tf.keras.Model):
    """
    Complete transformer with an Encoder and a Decoder.
    """
    def __init__(self, num_layers, embedding_dim, num_heads, d_model, fully_connected_dim, input_vocab_size, 
                 max_positional_encoding_input, target_vocab_size=None,
               max_positional_encoding_target=None, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               d_model=d_model,
                               fully_connected_dim=fully_connected_dim,
                               input_vocab_size=input_vocab_size,
                               maximum_position_encoding=max_positional_encoding_input,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers, 
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               d_model=d_model,
                               fully_connected_dim=fully_connected_dim,
                               target_vocab_size=input_vocab_size, 
                               maximum_position_encoding=max_positional_encoding_input,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.final_layer = tf.keras.layers.Dense(input_vocab_size, activation='softmax')
    
    def call(self, input_sentence, output_sentence, training, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):
        """
        Forward pass for the entire Transformer
        Arguments:
            input_sentence -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
                              An array of the indexes of the words in the input sentence
            output_sentence -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
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
        # call self.encoder with the appropriate arguments to get the encoder output
        enc_output = self.encoder(input_sentence, training, enc_padding_mask)  # (batch_size, inp_seq_len, fully_connected_dim)
        
        # call self.decoder with the appropriate arguments to get the decoder output
        # dec_output.shape == (batch_size, tar_seq_len, fully_connected_dim)
        dec_output, attention_weights = self.decoder(output_sentence, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        # pass decoder output through a linear layer and softmax
        final_output = self.final_layer(dec_output) # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

num_layers, num_heads, d_model, embedding_dim, fully_connected_dim = 6, 8, 512, 64, 2048
input_vocab_size, max_positioning_embedding = 29, 11
batch_size = 3


transformer = Transformer(num_layers=num_layers, 
                          num_heads=num_heads, 
                          d_model=d_model, 
                          embedding_dim=embedding_dim,
                        fully_connected_dim=fully_connected_dim, 
                        input_vocab_size=input_vocab_size, 
                        max_positional_encoding_input=max_positioning_embedding)

input_shape = (None, max_positioning_embedding, input_vocab_size)


x = tf.random.uniform((batch_size, max_positioning_embedding, input_vocab_size))
y =  tf.random.uniform((batch_size, max_positioning_embedding, input_vocab_size))

pred = transformer(x, y, training=True)
print(pred.shape)

transformer.summary()