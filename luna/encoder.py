import torch
import torch.nn as nn

from transformer.luna.attention import PackNUnpackAttention
from transformer.luna.feed_forward import PositionwiseFeedForwardNetwork


class LunaTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_attention_heads: int = 8,
            d_ff: int = 2048,
            dropout_p: float = 0.3,
    ) -> None:
        super(LunaTransformerEncoderLayer, self).__init__()
        self.luna_attention = PackNUnpackAttention(d_model, num_attention_heads)
        self.feed_forward = PositionwiseFeedForwardNetwork(d_model, d_ff, dropout_p)
        self.packed_context_layer_norm = nn.LayerNorm(d_model)
        self.unpacked_context_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

    def forward(
            self,
            inputs: torch.FloatTensor,
            p: torch.FloatTensor,
            attention_padding_mask: torch.FloatTensor = None,
    ):
        unpacked_context, packed_context = self.luna_attention(
            query=inputs,
            key=inputs,
            value=inputs,
            p=p,
            attention_padding_mask=attention_padding_mask,
        )

        packed_context = self.packed_context_layer_norm(packed_context + p)
        unpacked_context = self.unpacked_context_layer_norm(unpacked_context + inputs)

        outputs = self.feed_forward(unpacked_context)
        outputs = self.feed_forward_layer_norm(outputs + unpacked_context)

        return outputs, packed_context