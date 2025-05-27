import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# 
class DotProductAttention(nn.Module):
    r"""
    Args: dim, mask
        dim (int): dimension of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoders outputs.
    """
    def __init__(self, dim: int, scale: bool = True) -> None:
        super(DotProductAttention, self).__init__()
        if scale:
            self.sqrt_dim = np.sqrt(dim)
        else:
            self.sqrt_dim = 1

    def forward(self, 
                query: torch.FloatTensor, 
                key: torch.FloatTensor,
                value: torch.FloatTensor,
                mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # attention matrix 계산
        score = torch.matmul(query, key.transpose(2, 3)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)

        # product = attn @ v
        if len(query.size()) == 3:
            context = torch.bmm(attn, value)
        else:
            context = torch.matmul(attn, value)

        return context, attn


class MultiHeadAttention(nn.Module):
    r"""
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        dim (int): The dimension of model (default: 512)
        num_attention_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_attention_heads, v_len): tensor containing the attention (alignment) from the encoders outputs.
    """
    def __init__(self, dim: int = 512,
                 num_attention_heads: int = 8
    ) -> None:
        super(MultiHeadAttention, self).__init__()

        assert dim % num_attention_heads == 0, "d_model % num_attention_heads 가 0이 아님"

        self.d_head = int(dim / num_attention_heads)
        self.num_attention_heads = num_attention_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        
        self.scaled_dot_attn = DotProductAttention(dim, scale=True)

    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                value: torch.FloatTensor,
                mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_attention_heads, self.d_head).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_attention_heads, self.d_head).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_attention_heads, self.d_head).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_attention_heads * self.d_head)

        return context, attn


class PackNUnpackAttention(nn.Module):
    def __init__(self, dim, num_attention_heads: int = 8) -> None:
        super(PackNUnpackAttention, self).__init__()
        self.pack_attention = MultiHeadAttention(dim, num_attention_heads)
        self.unpack_attention = MultiHeadAttention(dim, num_attention_heads)

    def forward(
            self,
            query: torch.FloatTensor,
            key: torch.FloatTensor,
            value: torch.FloatTensor,
            p: torch.FloatTensor,
            attention_padding_mask: torch.BoolTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        packed_context, _ = self.pack_attention(p, key, value, attention_padding_mask)
        unpacked_context, _ = self.unpack_attention(query, packed_context, packed_context)
        return unpacked_context, packed_context