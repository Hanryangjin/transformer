import math
import torch
import torch.nn as nn

from transformer.luna.embedding import PositionalEncoding
from transformer.luna.encoder import LunaTransformerEncoderLayer
from transformer.luna.mask import get_attn_pad_mask, get_attn_subsequent_mask
from transformer.code_transformer.transformer import DecoderBlock, EncoderBlock
from transformer.basic_transformer.attention import _Embedding

class LunaTransformerEncoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int = 512,
            num_layers: int = 6,
            num_attention_heads: int = 8,
            d_ff: int = 2048,
            dropout_p: float = 0.1,
            project_embedding_length: int = 32,
            max_length: int = 128,
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

        # FP16 최적화
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
        #self.encoder_layers = LunaTransformerEncoder(vocab_size, d_model, num_layers, num_attention_heads, d_ff, dropout_p, project_embedding_length, max_length)
        #"""
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_attention_heads, d_ff, dropout_p)
            for _ in range(num_layers)
        ])
        #"""
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_attention_heads, d_ff, dropout_p)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, src_lengths, tgt):
        B, S = src.size()
        _, T = tgt.size()

        # 임베딩
        tgt_emb = self.embedding(tgt)
        
        # 인코더
        enc_input = src
        #enc_output = self.encoder_layers(enc_input, src_lengths)
        #"""
        enc_output = self.embedding(enc_input)
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)
        #"""
        # mask
        enc_pad_mask = get_attn_pad_mask(src, src_lengths, expand_length=T)
        enc_pad_mask  = (1.0 - enc_pad_mask.float()).unsqueeze(1)

        tgt_mask = get_attn_subsequent_mask(tgt)
        tgt_mask  = (1.0 - tgt_mask.float()).unsqueeze(1)
        
        """
        # [디버그용 출력]
        print("tgt_mask", tgt_mask.shape, tgt_mask.dtype,
            tgt_mask.min().item(), tgt_mask.max().item())
        print("src_mask", enc_pad_mask.shape, enc_pad_mask.dtype,
            enc_pad_mask.min().item(), enc_pad_mask.max().item())
        """

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