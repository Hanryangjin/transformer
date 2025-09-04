import math
import torch
import torch.nn as nn

from transformer.luna.embedding import PositionalEncoding
from transformer.luna.encoder import LunaTransformerEncoderLayer
from transformer.luna.mask import get_attn_pad_mask, get_attn_subsequent_mask
from transformer.code_transformer.transformer import DecoderBlock
from transformer.basic_transformer.attention import _Embedding

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
        """
        nn.init.normal_(self.projected_embeddings, mean=0.0, std=self.d_model ** -0.5)
        # 더 안정적인 초기화
        nn.init.xavier_uniform_(self.projected_embeddings)
        """
        ### 변경1. FP16 최적화
        nn.init.kaiming_uniform_(self.projected_embeddings, a=math.sqrt(5))

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
        self.final_layer_norm = nn.LayerNorm(self.d_model)

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

        ### 변경2. 최종 출력에 정규화 레이어 추가
        return self.final_layer_norm(outputs)

class LunaTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_layers: int = 6,
            num_attention_heads: int = 8,
            d_ff: int = 2048,
            dropout_p: float = 0.1,
            project_embedding_length: int = 32,
            max_length: int = 128,
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.PAD_TOKEN_ID = 0

        self.embedding = _Embedding(d_model, vocab_size, max_length, self.device, dropout_p)
        self.encoder_layers = LunaTransformerEncoder(vocab_size, d_model, num_layers, num_attention_heads, d_ff, dropout_p, project_embedding_length, max_length)
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_attention_heads, d_ff, dropout_p)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(d_model, vocab_size)
        
    """$수정필. 디코더에 mask 삽입"""
    def forward(self, src, src_lengths, tgt):
        B, S = src.size()
        _, T = tgt.size()

        # 임베딩
        tgt_emb = self.embedding(tgt)
        
        # 인코더
        enc_input = src
        enc_output = self.encoder_layers(enc_input, src_lengths)

        # mask
        enc_pad_mask = (src == self.pad_token_id)
        tgt_mask = self._subsequent_mask(T).to(src.device)

        # 디코더
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, enc_pad_mask, tgt_mask)

        # 출력층
        output = self.final_layer(dec_output)

        # [Debug]이상징후 감지 조건
        try:
            enc_mean = enc_output.mean().item()
            enc_var = enc_output.var().item()
            dec_mean = dec_output.mean().item()
            dec_max = dec_output.max().item()

            abnormal = (
                not math.isfinite(enc_mean) or
                not math.isfinite(enc_var) or
                not math.isfinite(dec_mean) or
                not math.isfinite(dec_max) or
                abs(enc_mean) > 1e2 or
                enc_var > 100.0 or
                dec_max > 50.0 or
                not torch.isfinite(output).all()
            )

            if abnormal:
                print(f"\n[⚠️ 이상 감지] Encoder/Decoder 출력 이상!")
                print(f"Encoder 출력 평균: {enc_mean}, 분산: {enc_var}")
                print(f"Decoder 출력 평균: {dec_mean}, 최대: {dec_max}")
                print(f"Output logits 예시: {output[0][:5]}")
        except Exception as e:
            print(f"[경고] 이상 감지 중 오류 발생: {e}")
        
        return output 

# $수정필. OP(operation)에 대응되는 Head
class EditOperationPredictor(nn.Module):
    def __init__(self, d_model, num_operations=4):
        super().__init__()
        self.operation_embedding = nn.Linear(d_model, d_model)
        self.operation_classifier = nn.Linear(d_model, num_operations)
        
    def forward(self, x):
        x = self.operation_embedding(x)
        return self.operation_classifier(x)

# $수정필. vocab_size(어휘 크기)에 대응되는 Head
class TokenGenerator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.token_embedding = nn.Linear(d_model, d_model)
        self.token_classifier = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.token_embedding(x)
        return self.token_classifier(x)
    
# EditOperationPredictor과 TokenGenerator에 더해
# vocab을 PeplaceHead와 AppendHead로 나눈 Head 분리 클래스
# $수정필. $분리권장. OP_SIZE 확인 후 입력. 
'''
    KEEP/DEL/REP/APP/SPACE_* 등 추가할 operation 확인 후 OP_SIZE 확정
'''
TAGS = {"KEEP":0, "DEL":1, "REP":2, "APP":3}
OP_SIZE = len(TAGS)

class LunaTagger(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.2):
        super().__init__()
        self.encoder = LunaTransformerEncoder(
            vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
            num_attention_heads=num_heads, d_ff=d_ff, dropout_p=dropout,
            project_embedding_length=32, max_length=128
        )  # Pack&Unpack 포함
        self.op_head   = nn.Linear(d_model, OP_SIZE)      # KEEP/DEL/REP/APP/SPACE_*
        self.rep_head  = nn.Linear(d_model, vocab_size)   # REPLACE용
        self.app_head  = nn.Linear(d_model, vocab_size)   # APPEND용

    def forward(self, input_ids, input_lengths):
        h = self.encoder(input_ids, input_lengths)        # (B,T,D)
        op_logits  = self.op_head(h)                      # (B,T,|OP|)
        rep_logits = self.rep_head(h)                     # (B,T,V)
        app_logits = self.app_head(h)                     # (B,T,V)
        return op_logits, rep_logits, app_logits

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