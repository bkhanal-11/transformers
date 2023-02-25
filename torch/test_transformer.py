import torch
import unittest
from transformer import Transformer

class TestTransformer(unittest.TestCase):
    
    def setUp(self):
        self.transformer = Transformer(
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_model=512,
            num_heads=8,
            dff=2048,
            input_vocab_size=100,
            target_vocab_size=100,
            max_seq_len=100,
            dropout_rate=0.1
        )
        
    def test_forward_pass(self):
        x = torch.randint(0, 100, (2, 30))
        y = torch.randint(0, 100, (2, 20))
        # enc_padding_mask = torch.zeros(2, 1, 30)
        # look_ahead_mask = torch.ones(2, 20, 20)
        # dec_padding_mask = torch.zeros(2, 1, 20)
        
        output, attention_weights = self.transformer(x, y)
        
        self.assertEqual(output.shape, torch.Size([2, 20, 100]))
        
if __name__ == "__main__":
    unittest.main()
