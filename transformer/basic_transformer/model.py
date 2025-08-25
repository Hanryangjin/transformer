import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(d_model, src_vocab_size, max_seq_length, num_heads, d_ff, num_layers, self.device, dropout=0.1)
        self.decoder = Decoder(d_model, tgt_vocab_size, max_seq_length, num_heads, d_ff, num_layers, self.device, dropout=0.1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):    
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        return output
    