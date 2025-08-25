import torch
import torch.nn as nn
from code_transformer.attention import MultiHeadAttention, FeedForward, _Embedding

class BertLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, src_mask=None):
        _x = x
        x = self.attention(x, x, x, src_mask)
        x = self.dropout1(x)
        x = self.norm1(_x + x)
        
        _x = x
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        
        return x

class BertEncoder(nn.Module):
    def __init__(self, d_model, vocab_size, max_len, num_heads, d_ff, num_layers, device, dropout=0.1):
        super().__init__()
        self.embedding = _Embedding(d_model, vocab_size, max_len, device, dropout)
        self.layers = nn.ModuleList([
            BertLayer(d_model=d_model,
                     num_heads=num_heads,
                     d_ff=d_ff,
                     dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        
        return x 