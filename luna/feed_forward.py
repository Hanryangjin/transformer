import torch
import torch.nn as nn

class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout_p: float = 0.1) -> None:
        super(PositionwiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(inputs)