import torch
import unittest
from transformer import Transformer
from utils import create_look_ahead_mask, create_padding_mask

class TestTransformer(unittest.TestCase):
    
    def setUp(self):
        self.num_layer = 6
        self.input_vocab_size=100
        self.target_vocab_size=100
        self.max_seq_len=100
        self.batch_size = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transformer = Transformer(
            num_encoder_layers=self.num_layer,
            num_decoder_layers=self.num_layer,
            d_model=512,
            num_heads=8,
            dff=2048,
            input_vocab_size=self.input_vocab_size,
            target_vocab_size=self.target_vocab_size,
            max_seq_len=self.max_seq_len,
            dropout_rate=0.1
        )
        
    def test_forward_pass(self):
        x = torch.randint(0, self.input_vocab_size, (self.batch_size, self.max_seq_len)).to(self.device)
        y = torch.randint(0, self.target_vocab_size, (self.batch_size, self.max_seq_len)).to(self.device)
        
        enc_padding_mask = create_padding_mask(x).to(self.device)
        dec_padding_mask = create_padding_mask(y).to(self.device)
        look_ahead_mask = create_look_ahead_mask(self.max_seq_len).to(self.device)
        
        output, _ = self.transformer(x, y, enc_padding_mask, look_ahead_mask, dec_padding_mask)
        
        self.assertEqual(output.shape, (self.batch_size, self.max_seq_len, self.target_vocab_size))
        
if __name__ == "__main__":
    unittest.main()
