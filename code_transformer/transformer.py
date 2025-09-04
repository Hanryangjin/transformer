import torch
import torch.nn as nn
from transformer.basic_transformer.attention import MultiHeadAttention
from transformer.basic_transformer.attention import _Embedding
from transformer.basic_transformer.attention import FeedForward
import math

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        residual = x
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        
        # Feed-forward
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention
        residual = x
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        
        # Cross-attention
        residual = x
        x = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        
        # Feed-forward
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(residual + x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_len=512, dropout=0.1, device='cuda'):
        super().__init__()
        self.embedding = _Embedding(d_model, vocab_size, max_len, device, dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 임베딩
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        
        # 인코더
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # 디코더
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # 출력층
        output = self.final_layer(dec_output)
        
        return output 