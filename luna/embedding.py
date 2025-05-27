import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, device) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, d_model, requires_grad=False)
        position = torch.arange(0, max_length, dtype=torch.float, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, step=2, device=device).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]
    
"""
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
"""