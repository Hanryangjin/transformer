import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# 
class DotProductAttention(nn.Module):
    r"""
    Args: dim, scale
        dim (int): attention 차원
        scale (bool, optional): attenion에 스케일 적용 여부

    Inputs: query, key, value, mask
        - query (torch.FloatTensor): Query값 - 입력 X에 대한 완전연결층 출력
        - key (torch.FloatTensor): Key값 - 입력 X에 대한 완전연결층 출력
        - value (torch.FloatTensor): Value값 - 입력 X에 대한 완전연결층 출력
        - mask (Optional[torch.FloatTensor], optional): Mask 적용 여부

    Returns: context, attn
        - context: attention 연산 결과
        - attn: attention matrix
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

        # query 차원에 따라 attn @ v 계산
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
        dim (int): attention 차원
        num_attention_heads (int): attention head 수

    Inputs: query, key, value, mask
        - query (torch.FloatTensor[batch, q_len, d_model]): Query값 - 입력 X에 대한 완전연결층 출력
        - key (torch.FloatTensor[batch, k_len, d_model]): Key값 - 입력 X에 대한 완전연결층 출력
        - value (torch.FloatTensor[batch, v_len, d_model]): Value값 - 입력 X에 대한 완전연결층 출력
        - mask (Optional[torch.FloatTensor], optional): Mask 적용 여부

    Returns: output, attn
        - **output** (batch, output_len, dimensions): attention 연산 결과
        - **attn** (batch * num_attention_heads, v_len): attention matrix
    """
    def __init__(self, dim: int, num_attention_heads: int) -> None:
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