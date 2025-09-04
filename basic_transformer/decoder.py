import torch
import torch.nn as nn
from attention import MultiHeadAttention, FeedForward, _Embedding

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, dec_input, enc_output, src_mask=None, tgt_mask=None):
        _x = dec_input
        x = self.self_attention(dec_input, dec_input, dec_input, tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(_x + x)
        
        if enc_output is not None:
            _x = x
            x = self.cross_attention(x, enc_output, enc_output, src_mask)
            x = self.dropout2(x)
            x = self.norm2(_x + x)
        
        _x = x
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x = self.norm3(_x + x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, vocab_size, max_len, num_heads, d_ff, num_layers, device, dropout=0.1):
        super().__init__()
        self.embedding = _Embedding(d_model, vocab_size, max_len, device, dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model=d_model,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        output = self.linear(x)
        return output