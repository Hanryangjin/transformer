import torch
import torch.nn as nn
from code_transformer.attention import _Embedding
from bert_code.bert_encoder import BertEncoder

class BertModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_len=512, dropout=0.1, device='cuda'):
        super().__init__()
        self.embedding = _Embedding(d_model, vocab_size, max_len, device, dropout)
        self.encoder = BertEncoder(d_model, vocab_size, max_len, num_heads, d_ff, num_layers, device, dropout)
        self.pooler = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()
        
    def forward(self, input_ids, attention_mask=None):
        # 임베딩 레이어
        embeddings = self.embedding(input_ids)
        
        # 인코더 레이어
        sequence_output = self.encoder(embeddings, attention_mask)
        
        # [CLS] 토큰의 임베딩을 사용하여 문장 표현 생성
        pooled_output = self.pooler(sequence_output[:, 0])
        pooled_output = self.activation(pooled_output)
        
        return sequence_output, pooled_output

class BertForMaskedLM(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.mlm_head = nn.Linear(bert_model.embedding.tok_embedding.weight.size(1),
                                bert_model.embedding.tok_embedding.weight.size(0))
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, attention_mask)
        prediction_scores = self.mlm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))
            
        return prediction_scores, loss 