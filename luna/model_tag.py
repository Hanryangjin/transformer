# model_tagger.py  (새 파일로 분리 권장)
import torch, torch.nn as nn
from transformer.luna.model import LunaTransformerEncoder  # :contentReference[oaicite:8]{index=8}

TAGS = {"KEEP":0, "DEL":1, "REP":2, "APP":3}
OP_SIZE = len(TAGS)

class LunaTagger(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.2):
        super().__init__()
        self.encoder = LunaTransformerEncoder(
            vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
            num_attention_heads=num_heads, d_ff=d_ff, dropout_p=dropout,
            project_embedding_length=32, max_length=1024
        )  # Pack&Unpack 포함 :contentReference[oaicite:9]{index=9}
        self.op_head   = nn.Linear(d_model, OP_SIZE)    # (B,T,|OP|)
        self.rep_head  = nn.Linear(d_model, vocab_size) # (B,T,V)
        self.app_head  = nn.Linear(d_model, vocab_size) # (B,T,V)

    def forward(self, input_ids, input_lengths):
        h = self.encoder(input_ids, input_lengths)        # (B,T,D)
        return self.op_head(h), self.rep_head(h), self.app_head(h)

# loss_tagger.py
import torch, torch.nn as nn

TAGS = {"KEEP":0, "DEL":1, "REP":2, "APP":3}

class TaggerLoss(nn.Module):
    def __init__(self, pad_id=0, bos_id=2, eos_id=3, w_op=1.0, w_rep=1.0, w_app=1.0):
        super().__init__()
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.ce = nn.CrossEntropyLoss(reduction='none')

        self.w_op, self.w_rep, self.w_app = w_op, w_rep, w_app

    def forward(self, op_logits, rep_logits, app_logits, op_ids, rep_ids, app_ids):
        B, T, _ = op_logits.size()

        # 공통 ignore mask (PAD/BOS/EOS 제외)
        ignore = torch.zeros(B, T, dtype=torch.bool, device=op_logits.device)
        # 필요시 op_ids 기반 더 정교화 가능

        # op loss
        op_loss_all = self.ce(op_logits.view(B*T, -1), op_ids.view(-1))
        op_loss = (op_loss_all.view(B,T)[~ignore]).mean()

        # rep loss (REP 위치만)
        rep_mask = (op_ids == TAGS["REP"]) & (~ignore)
        if rep_mask.any():
            rep_loss_all = self.ce(rep_logits.view(B*T, -1), rep_ids.view(-1)).view(B,T)
            rep_loss = (rep_loss_all[rep_mask]).mean()
        else:
            rep_loss = torch.zeros((), device=op_logits.device)

        # app loss (APP 위치만) – 초기는 0으로 떨어질 수 있음
        app_mask = (op_ids == TAGS["APP"]) & (~ignore)
        if app_mask.any():
            app_loss_all = self.ce(app_logits.view(B*T, -1), app_ids.view(-1)).view(B,T)
            app_loss = (app_loss_all[app_mask]).mean()
        else:
            app_loss = torch.zeros((), device=op_logits.device)

        total = self.w_op*op_loss + self.w_rep*rep_loss + self.w_app*app_loss
        return total, {"op": op_loss.item(), "rep": rep_loss.item(), "app": app_loss.item()}
