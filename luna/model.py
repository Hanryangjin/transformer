import math
import torch
import torch.nn as nn

from transformer.luna.embedding import PositionalEncoding
from transformer.luna.encoder import LunaTransformerEncoderLayer
from transformer.luna.mask import get_attn_pad_mask

class LunaTransformerEncoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_layers: int = 6,
            num_attention_heads: int = 8,
            d_ff: int = 2048,
            dropout_p: float = 0.1,
            project_embedding_length: int = 32,
            max_length: int = 1024,
    ):
        super(LunaTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.projected_embedding_length = project_embedding_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # projected embeddings 초기화
        self.projected_embeddings = nn.Parameter(torch.Tensor(project_embedding_length, self.d_model))
        self.projected_positions = PositionalEncoding(
            d_model=self.d_model,
            max_length=project_embedding_length,
            device=self.device
        )
        nn.init.normal_(self.projected_embeddings, mean=0.0, std=self.d_model ** -0.5)
        
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout_p)
        self.input_positions = PositionalEncoding(
            d_model=d_model,
            max_length=max_length,
            device=self.device
        )

        self.input_norm = nn.LayerNorm(d_model)
        self.embed_scale = math.sqrt(self.d_model)
        self.layers = nn.ModuleList([
            LunaTransformerEncoderLayer(
                d_model=d_model,
                num_attention_heads=num_attention_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
        ])

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        batch_size, seq_length = inputs.size()

        attention_padding_mask = get_attn_pad_mask(inputs, input_lengths, self.projected_embedding_length)

        embedded = self.input_embedding(inputs)

        # input embedding 및 projected embedding(Packed Matrix) 스케일링
        embedded *= self.embed_scale
        projected_embedded = self.projected_embeddings * self.embed_scale

        # input embedding 및 projected embedding 위치 인코딩
        embedded += self.input_positions(embedded.size(1))
        projected_embedded += self.projected_positions(self.projected_embedding_length).squeeze(0)

        seq_length, dim = projected_embedded.size()
        projected_embedded = projected_embedded.unsqueeze(0).expand(batch_size, seq_length, dim)

        outputs = self.dropout(embedded)
        p = self.dropout(projected_embedded)

        for layer in self.layers:
            outputs, p = layer(outputs, p, attention_padding_mask)

        return outputs

class EditOperationPredictor(nn.Module):
    def __init__(self, d_model, num_operations=4):
        super().__init__()
        self.operation_embedding = nn.Linear(d_model, d_model)
        self.operation_classifier = nn.Linear(d_model, num_operations)
        
    def forward(self, x):
        x = self.operation_embedding(x)
        return self.operation_classifier(x)

class TokenGenerator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.token_embedding = nn.Linear(d_model, d_model)
        self.token_classifier = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.token_embedding(x)
        return self.token_classifier(x)

class EditBasedLunaModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_attention_heads=8):
        super().__init__()
        self.encoder = LunaTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            d_ff=2048,
            dropout_p=0.1,
            project_embedding_length=32,
            max_length=1024
        )
        
        # Edit Operation 예측기
        self.operation_predictor = EditOperationPredictor(d_model)
        
        # 토큰 생성기
        self.token_generator = TokenGenerator(d_model, vocab_size)
        
        # Operation 임베딩
        self.operation_embedding = nn.Embedding(4, d_model)  # KEEP, DELETE, INSERT, REPLACE
        
    def forward(self, input_ids, input_lengths):
        # 인코더를 통과
        encoder_outputs = self.encoder(input_ids, input_lengths)
        
        # Edit Operation 예측
        operation_logits = self.operation_predictor(encoder_outputs)
        
        # 토큰 생성
        token_logits = self.token_generator(encoder_outputs)
        
        return {
            'operation_logits': operation_logits,
            'token_logits': token_logits
        }
    
    def predict(self, input_ids, input_lengths):
        """
        추론 시 사용하는 메서드
        """
        # 인코더를 통과
        encoder_outputs = self.encoder(input_ids, input_lengths)
        
        # Edit Operation 예측
        operation_logits = self.operation_predictor(encoder_outputs)
        predicted_operations = torch.argmax(operation_logits, dim=-1)
        
        # 토큰 생성
        token_logits = self.token_generator(encoder_outputs)
        predicted_tokens = torch.argmax(token_logits, dim=-1)
        
        return {
            'operations': predicted_operations,
            'tokens': predicted_tokens
        }