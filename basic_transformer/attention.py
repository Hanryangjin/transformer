import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        """
        print("scores", scores.shape, "mask", mask.shape)  # [디버그용]
        """
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear layer
        output = self.w_o(context)
        
        return output
    
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_model = k.size()

        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_model)  # scaled dot product

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)
        v = score @ v

        return v, score
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class _Embedding(nn.Module):
    def __init__(self, d_model, vocab_size, max_len, device, dropout=0.1):
        super(_Embedding, self).__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(dropout)
        self.class_count = 0
        
    def forward(self, x):
        # 입력 텐서를 long 타입으로 변환
        if x.dtype != torch.long:
            x = x.long()
        
        """디버깅용"""
        if self.class_count == 0:
            print(f"\n최대 인덱스: {x.max().item()}")
            print(f"임베딩 사이즈: {self.tok_embedding.num_embeddings}")
            self.class_count += 1
        
        # 토큰 임베딩: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        tok_embedding = self.tok_embedding(x)
        
        # 위치 인코딩: [batch_size, seq_len, d_model]
        positional_encoding = self.positional_encoding(x)
        
        # 두 임베딩을 더하고 드롭아웃 적용
        return self.dropout(tok_embedding + positional_encoding)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # 입력 텐서의 크기에 맞게 위치 인코딩을 반환
        # x의 크기는 [batch_size, seq_len] 또는 [batch_size, seq_len, d_model]
        if len(x.size()) == 2:
            batch_size, seq_len = x.size()
            # [batch_size, seq_len, d_model] 형태로 확장
            return self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # 이미 [batch_size, seq_len, d_model] 형태인 경우
            batch_size, seq_len, _ = x.size()
            return self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)