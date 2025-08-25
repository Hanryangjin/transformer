import torch
import torch.nn as nn
from code_transformer.attention import _Embedding, FeedForward
from bert_code.bert_encoder import BertEncoder
from bart_code.mask_infilling import MaskInfilling

class BartEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_len=512, dropout=0.1, device='cuda'):
        super().__init__()
        self.embedding = _Embedding(d_model, vocab_size, max_len, device, dropout)
        self.encoder = BertEncoder(d_model, vocab_size, max_len, num_heads, d_ff, num_layers, device, dropout)
        
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        sequence_output = self.encoder(embeddings, attention_mask)
        return sequence_output

class BartDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_len=512, dropout=0.1, device='cuda'):
        super().__init__()
        self.embedding = _Embedding(d_model, vocab_size, max_len, device, dropout)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, num_heads, dropout=dropout),
                'norm1': nn.LayerNorm(d_model),
                'cross_attn': nn.MultiheadAttention(d_model, num_heads, dropout=dropout),
                'norm2': nn.LayerNorm(d_model),
                'ffn': FeedForward(d_model, d_ff, dropout),
                'norm3': nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, encoder_output, attention_mask=None, decoder_mask=None):
        embeddings = self.embedding(input_ids)
        x = embeddings
        
        # 어텐션 마스크 변환
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_mask = (1.0 - attention_mask) * -10000.0  # 마스킹된 부분을 -inf로 설정
        
        if decoder_mask is not None:
            # 디코더 마스크 생성 (상삼각 행렬)
            seq_len = input_ids.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(input_ids.device)
            
            # 패딩 마스크와 결합
            decoder_mask = decoder_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            decoder_mask = decoder_mask.expand(-1, -1, seq_len, -1)  # [batch_size, 1, seq_len, seq_len]
            decoder_mask = decoder_mask.masked_fill(causal_mask, float('-inf'))
        
        for layer in self.layers:
            # Self-attention
            residual = x
            x = layer['norm1'](x)
            x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
            x, _ = layer['self_attn'](x, x, x, attn_mask=decoder_mask)
            x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
            x = residual + x
            
            # Cross-attention
            residual = x
            x = layer['norm2'](x)
            x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
            encoder_output_t = encoder_output.transpose(0, 1)  # [seq_len, batch_size, d_model]
            x, _ = layer['cross_attn'](x, encoder_output_t, encoder_output_t, attn_mask=attention_mask)
            x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
            x = residual + x
            
            # Feed-forward
            residual = x
            x = layer['norm3'](x)
            x = layer['ffn'](x)
            x = residual + x
        
        return x

class BartModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_len=512, dropout=0.1, device='cuda', lambda_param=3.0):
        super().__init__()
        self.encoder = BartEncoder(vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout, device)
        self.decoder = BartDecoder(vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout, device)
        self.mask_infilling = MaskInfilling(None, lambda_param)
        
    def set_tokenizer(self, tokenizer):
        self.mask_infilling.tokenizer = tokenizer
        
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_mask=None):
        encoder_output = self.encoder(input_ids, attention_mask)
        decoder_output = self.decoder(decoder_input_ids, encoder_output, attention_mask, decoder_mask)
        
        return encoder_output, decoder_output
    
    def generate_masked_input(self, text):
        return self.mask_infilling.get_masking_info(text) 