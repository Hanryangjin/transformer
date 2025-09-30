# -*- coding: utf-8 -*-
"""
KoBERT(monologg/kobert) 기반 GEC 태깅 멀티태스크 학습 스크립트 (GECToR 스타일 수정 적용)
- 학습 데이터: ./transformer/TrainData/kogector_extended.json
- 적용 변경:
  1) 추론 임계치: MIN_ERROR_PROB / ADDITIONAL_KEEP_LOGIT
  2) 반복(Iterative) 검증 디코딩: N_ITER
  3) 비-KEEP 가중 label-smoothed CE
  4) Cold epochs: 초기 몇 epoch 백본 freeze
  5) tqdm / 예시 5개 / PRF0.5 / ckpt 저장( state_dict )
"""

import os
import json
import math
import random
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from difflib import SequenceMatcher
from tqdm.auto import tqdm

# ---------------------------
# 기본 설정
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Korean-GEC-Tagger")

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 사용자 설정 ==========
MODEL_PATH = "monologg/kobert"
TRAIN_JSON_PATH = "./transformer/TrainData/kogector_extended.json"
VAL_JSON_PATH   = ""    # 비워두면 TRAIN에서 자동 분할(90/10)

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
MAX_LEN = 128
GRAD_ACCUM_STEPS = 1
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0  # 선택: grad clipping

# 멀티태스크 손실 가중치
LOSS_WEIGHT_TOKEN = 1.0
LOSS_WEIGHT_BOUND = 0.5
LOSS_WEIGHT_PART  = 0.75  # (현재 loss 미포함)

# 평가/출력 관련
NUM_SHOW_SAMPLES = 5   # 매 에포크마다 예시 출력 개수

# ====== GECToR 스타일 하이퍼 ======
# 1) 토큰 임계치/바이어스
MIN_ERROR_PROB = 0.30          # (1 - P(KEEP)) < MIN_ERROR_PROB 이면 KEEP 강제
ADDITIONAL_KEEP_LOGIT = 0.20   # KEEP 로그잇에 가산 바이어스(>0이면 보수적)
# 2) 반복 추론 횟수 (검증 시)
N_ITER = 3                     # 1회차는 배치 출력, 2~N회는 단문 재호출
# 3) 비-KEEP 가중치 (손실)
TOKEN_NON_KEEP_WEIGHT = 3.0
BOUND_NON_KEEP_WEIGHT = 2.0
# 4) Cold epochs
COLD_EPOCHS = 1                # 처음 1 epoch 백본 freeze

# ----------------------------------------------------
# Label-smoothed loss (class weight 지원)
# ----------------------------------------------------
def label_smoothed_loss(pred, target, epsilon=0.1, ignore_index=-100, class_weight=None):
    V = pred.size(-1)
    log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
    mask = (target != ignore_index)
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    target_clamped = target.clone()
    target_clamped[~mask] = 0
    one_hot = torch.nn.functional.one_hot(target_clamped, num_classes=V).float()
    smoothed = (1 - epsilon) * one_hot + (epsilon / V)
    if class_weight is not None:
        # class_weight: [V]
        smoothed = smoothed * class_weight.unsqueeze(0)
        # 정규화(선택): smoothed 합이 1 근사 유지하도록
        denom = smoothed.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        smoothed = smoothed / denom
    loss_per_tok = -(smoothed * log_probs).sum(dim=-1)
    loss = loss_per_tok[mask].mean()
    return loss

# ----------------------------------------------------
# 데이터 유틸
# ----------------------------------------------------
def load_examples(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            items = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if not line: continue
                items.append(json.loads(line))
    return items

# -100: 무시되는 라벨 값
IGNORE_LABEL = -100

def collect_label_sets(examples: List[Dict[str, Any]]):
    token_label_set, boundary_label_set, particle_label_set = set(), set(), set()
    for ex in examples:
        for lb in ex.get("token_labels", []):
            if lb != str(IGNORE_LABEL):
                token_label_set.add(lb)
        for lb in ex.get("boundary_labels", []):
            if lb != str(IGNORE_LABEL):
                boundary_label_set.add(lb)
        for lb in ex.get("particle_labels", []):
            if lb != str(IGNORE_LABEL):
                particle_label_set.add(lb)

    token_label_set.update(["KEEP", "DELETE"])
    boundary_label_set.update(["KEEP"])
    if len(particle_label_set) == 0:
        particle_label_set.add("KEEP")

    token_labels = sorted(list(token_label_set))
    boundary_labels = sorted(list(boundary_label_set))
    particle_labels = sorted(list(particle_label_set))

    token2id = {lb: i for i, lb in enumerate(token_labels)}
    bound2id = {lb: i for i, lb in enumerate(boundary_labels)}
    part2id  = {lb: i for i, lb in enumerate(particle_labels)}

    meta = {
        "token_labels": token_labels,
        "boundary_labels": boundary_labels,
        "particle_labels": particle_labels,
        "token2id": token2id,
        "bound2id": bound2id,
        "part2id": part2id,
        "id2token": {i: lb for lb, i in token2id.items()},
        "id2bound": {i: lb for lb, i in bound2id.items()},
        "id2part":  {i: lb for lb, i in part2id.items()},
    }
    return meta

class GECTagDataset(Dataset):
    def __init__(self, examples, tokenizer, label_meta, max_len=256):
        self.examples = examples
        self.tok = tokenizer
        self.max_len = max_len
        self.label_meta = label_meta

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        pieces = ex["pieces"]
        if len(pieces) > self.max_len - 2:
            pieces = pieces[: self.max_len - 2]

        # [CLS] + pieces + [SEP]
        pieces_with_special = [self.tok.cls_token] + pieces + [self.tok.sep_token]
        input_ids = self.tok.convert_tokens_to_ids(pieces_with_special)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        def align_labels(raw_labels, label2id):
            aligned = [IGNORE_LABEL]
            for lb in raw_labels[: len(pieces)]:
                if lb == str(IGNORE_LABEL):
                    aligned.append(IGNORE_LABEL)
                else:
                    mapped = label2id.get(lb, label2id.get("KEEP", 0))
                    aligned.append(mapped)
            aligned.append(IGNORE_LABEL)
            return aligned

        token_labels = align_labels(ex.get("token_labels", []), self.label_meta["token2id"])
        boundary_labels = align_labels(ex.get("boundary_labels", []), self.label_meta["bound2id"])
        particle_labels = align_labels(ex.get("particle_labels", []), self.label_meta["part2id"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "token_labels": torch.tensor(token_labels, dtype=torch.long),
            "boundary_labels": torch.tensor(boundary_labels, dtype=torch.long),
            "particle_labels": torch.tensor(particle_labels, dtype=torch.long),
            "raw_pieces": pieces,       # 디코딩용
            "meta": ex.get("meta", {}), # src/tgt
        }

def pad_sequence(items: List[torch.Tensor], pad_val: int) -> torch.Tensor:
    max_len = max(t.size(0) for t in items)
    out = []
    for t in items:
        if t.size(0) < max_len:
            pad = torch.full((max_len - t.size(0),), pad_val, dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        out.append(t)
    return torch.stack(out, dim=0)

def collate_fn(batch):
    input_ids = pad_sequence([b["input_ids"] for b in batch], 0)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], 0)
    token_type_ids = pad_sequence([b["token_type_ids"] for b in batch], 0)
    token_labels = pad_sequence([b["token_labels"] for b in batch], IGNORE_LABEL)
    boundary_labels = pad_sequence([b["boundary_labels"] for b in batch], IGNORE_LABEL)
    particle_labels = pad_sequence([b["particle_labels"] for b in batch], IGNORE_LABEL)

    raw_pieces = [b["raw_pieces"] for b in batch]
    metas = [b["meta"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "token_labels": token_labels,
        "boundary_labels": boundary_labels,
        "particle_labels": particle_labels,
        "raw_pieces": raw_pieces,
        "metas": metas,
    }

# ----------------------------------------------------
# KoBERT SPM 호환성 점검(선택)
# ----------------------------------------------------
def check_spm_coverage(tokenizer, samples, top_n=2000):
    unk_id = tokenizer.unk_token_id
    oov, total = 0, 0
    boundary_mismatch = 0
    for ex in samples[:top_n]:
        pcs = ex["pieces"]
        ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + pcs + [tokenizer.sep_token])
        oov += sum(1 for t in ids if t == unk_id)
        total += len(ids)
        spm_like = sum(1 for p in pcs if p.startswith("▁"))
        if spm_like == 0:
            boundary_mismatch += 1
    oov_rate = (oov / max(1, total)) * 100
    if oov_rate > 5.0 or boundary_mismatch > 0:
        print(f"[경고] KoBERT SPM 불일치 가능성: OOV {oov_rate:.2f}%, 경계불일치 샘플 {boundary_mismatch}")
    else:
        print(f"[OK] KoBERT SPM 커버리지 양호: OOV {oov_rate:.2f}%")

# ----------------------------------------------------
# 모델
# ----------------------------------------------------
class MultiHeadGECTagger(nn.Module):
    def __init__(self, model_name_or_path: str, label_meta: Dict[str, Any]):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        hidden = self.config.hidden_size

        self.num_token = len(label_meta["token_labels"])
        self.num_bound = len(label_meta["boundary_labels"])
        self.num_part  = len(label_meta["particle_labels"])

        self.dropout = nn.Dropout(getattr(self.config, "hidden_dropout_prob", 0.1))
        self.head_token = nn.Linear(hidden, self.num_token)
        self.head_bound = nn.Linear(hidden, self.num_bound)
        self.head_part  = nn.Linear(hidden, self.num_part)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                token_labels=None, boundary_labels=None, particle_labels=None,
                loss_f_token=None, loss_f_bound=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        seq_out = self.dropout(outputs.last_hidden_state)

        logits_token = self.head_token(seq_out)
        logits_bound = self.head_bound(seq_out)
        logits_part  = self.head_part(seq_out)

        loss = None
        # 조사(part)는 loss 미포함(원 코드 유지)
        if (token_labels is not None) or (boundary_labels is not None):
            loss = 0.0
            if token_labels is not None:
                if loss_f_token is None:
                    loss_f_token = lambda pred, tgt: label_smoothed_loss(
                        pred, tgt, epsilon=0.1, ignore_index=IGNORE_LABEL
                    )
                loss += LOSS_WEIGHT_TOKEN * loss_f_token(
                    logits_token.view(-1, logits_token.size(-1)),
                    token_labels.view(-1)
                )
            if boundary_labels is not None:
                if loss_f_bound is None:
                    loss_f_bound = lambda pred, tgt: label_smoothed_loss(
                        pred, tgt, epsilon=0.1, ignore_index=IGNORE_LABEL
                    )
                loss += LOSS_WEIGHT_BOUND * loss_f_bound(
                    logits_bound.view(-1, logits_bound.size(-1)),
                    boundary_labels.view(-1)
                )

        return {
            "loss": loss,
            "logits_token": logits_token,
            "logits_bound": logits_bound,
            "logits_part": logits_part,
        }

# ----------------------------------------------------
# 디코더/평가 유틸
# ----------------------------------------------------
def _strip_leading_bar(token: str) -> Tuple[str, bool]:
    if token.startswith("▁"):
        return token[1:], True
    return token, False

def apply_tags_to_sentence(pieces: List[str],
                           token_label_strs: List[str],
                           boundary_label_strs: List[str],
                           particle_label_strs: List[str]) -> str:
    out_tokens: List[str] = []
    spaces_after: List[bool] = []

    i = 0
    while i < len(pieces):
        tok = pieces[i]
        base_tok, sp_boundary = _strip_leading_bar(tok)

        t_lab = token_label_strs[i] if i < len(token_label_strs) else "KEEP"
        p_lab = particle_label_strs[i] if i < len(particle_label_strs) else "KEEP"

        effective_label = p_lab if (p_lab != str(IGNORE_LABEL) and p_lab != "-100" and p_lab != "KEEP") else t_lab

        if effective_label.startswith("REPLACE_"):
            new_tok = effective_label.split("REPLACE_", 1)[1]
            out_tokens.append(new_tok)
            spaces_after.append(sp_boundary)
        elif effective_label.startswith("INSERT_"):
            out_tokens.append(base_tok)
            spaces_after.append(sp_boundary)
            ins_tok = effective_label.split("INSERT_", 1)[1]
            out_tokens.append(ins_tok)
            spaces_after.append(False)
        elif effective_label == "DELETE":
            pass
        else:
            out_tokens.append(base_tok)
            spaces_after.append(sp_boundary)
        i += 1

    for i in range(len(out_tokens) - 1):
        if i+1 < len(boundary_label_strs):
            b = boundary_label_strs[i+1]
            if b == "SPACE_INS":
                spaces_after[i] = True
            elif b == "SPACE_DEL":
                spaces_after[i] = False

    s = []
    for i, tok in enumerate(out_tokens):
        if i == 0:
            s.append(tok)
        else:
            if spaces_after[i-1]:
                s.append(" ")
            s.append(tok)
    return "".join(s)

def word_edits(src: str, dst: str) -> List[Tuple[str, Tuple[int,int], Tuple[int,int]]]:
    src_tok = src.split()
    dst_tok = dst.split()
    sm = SequenceMatcher(a=src_tok, b=dst_tok)
    edits = []
    for tag, i0, i1, j0, j1 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            edits.append(("replace", (i0, i1), (j0, j1)))
        elif tag == "delete":
            edits.append(("delete", (i0, i1), (j0,  j0)))
        elif tag == "insert":
            edits.append(("insert", (i0, i0), (j0, j1)))
    return edits

def prf05_from_edits(src: str, hyp: str, tgt: str) -> Tuple[float,float,float]:
    gold = word_edits(src, tgt)
    pred = word_edits(src, hyp)

    def normalize(edits, dst_text):
        dst_tok = dst_text.split()
        norm = []
        for op, (i0,i1), (j0,j1) in edits:
            dst_segment = " ".join(dst_tok[j0:j1])
            norm.append((op, i1-i0, j1-j0, dst_segment))
        return set(norm)

    G = normalize(gold, tgt)
    P = normalize(pred, hyp)

    tp = len(G & P)
    fp = len(P - G)
    fn = len(G - P)

    prec = tp / (tp + fp) if (tp+fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp+fn) > 0 else 0.0
    beta2 = 0.5 * 0.5
    if prec + rec == 0:
        f05 = 0.0
    else:
        f05 = (1 + beta2) * prec * rec / (beta2 * prec + rec)
    return prec, rec, f05

@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = labels.ne(IGNORE_LABEL)
    correct = (preds.eq(labels) & mask).sum().item()
    total = mask.sum().item()
    return (correct / total) if total > 0 else 0.0

# ----------------------------------------------------
# 임계치/바이어스 적용 태그 디코딩 (1회)
# ----------------------------------------------------
@torch.no_grad()
def decode_with_thresholds_from_logits(
    logits_token: torch.Tensor, logits_bound: torch.Tensor, logits_part: torch.Tensor,
    pieces: List[str], id2token: Dict[int,str], id2bound: Dict[int,str], id2part: Dict[int,str],
    keep_id: int
) -> Tuple[str, List[str], List[str], List[str]]:
    """
    배치 내 특정 문장에 대한 logits → (hyp sentence, token_strs, bound_strs, part_strs)
    logits_*: [L, C] (CLS/SEP 포함된 상태라면 caller에서 잘라서 넘겨야 함)
    """
    # KEEP 바이어스(로그잇에 가산) → softmax
    biased_token_logits = logits_token.clone()
    biased_token_logits[..., keep_id] += ADDITIONAL_KEEP_LOGIT
    probs = torch.softmax(biased_token_logits, dim=-1)  # [L, Ct]

    # 임계치 적용: error_prob = 1 - P(KEEP)
    keep_prob = probs[..., keep_id]
    error_prob = 1.0 - keep_prob
    pred_token = probs.argmax(-1)  # [L]
    pred_token = torch.where(error_prob >= MIN_ERROR_PROB, pred_token, torch.full_like(pred_token, keep_id))

    token_strs = [id2token[int(x)] for x in pred_token]
    bound_strs = [id2bound[int(x)] for x in logits_bound.argmax(-1)]
    part_strs  = [id2part[int(x)]  for x in logits_part.argmax(-1)]

    # 최종 문장 조립
    hyp = apply_tags_to_sentence(pieces, token_strs, bound_strs, part_strs)
    return hyp, token_strs, bound_strs, part_strs

# ----------------------------------------------------
# 단문 forward + 디코딩(토크나이처 pieces 입력)
# ----------------------------------------------------
@torch.no_grad()
def forward_on_pieces(model, tok, pieces: List[str], id_maps, keep_id: int) -> str:
    # [CLS] + pieces + [SEP]
    input_ids = tok.convert_tokens_to_ids([tok.cls_token] + pieces + [tok.sep_token])
    attn = [1] * len(input_ids)
    type_ids = [0] * len(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    attn = torch.tensor(attn, dtype=torch.long, device=DEVICE).unsqueeze(0)
    type_ids = torch.tensor(type_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    out = model(input_ids=input_ids, attention_mask=attn, token_type_ids=type_ids)
    # CLS/SEP 제외
    lt = out["logits_token"][0, 1:-1, :]
    lb = out["logits_bound"][0, 1:-1, :]
    lp = out["logits_part"][0, 1:-1, :]

    hyp, *_ = decode_with_thresholds_from_logits(
        lt, lb, lp, pieces,
        id_maps["id2token"], id_maps["id2bound"], id_maps["id2part"], keep_id
    )
    return hyp

# ----------------------------------------------------
# 학습/평가
# ----------------------------------------------------
def print_label_distribution(items, name):
    t_counter = Counter()
    b_counter = Counter()
    p_counter = Counter()
    for ex in items:
        t_counter.update(ex.get("token_labels", []))
        b_counter.update(ex.get("boundary_labels", []))
        p_counter.update(ex.get("particle_labels", []))
    print(f"[{name}] token_labels 분포: {dict(t_counter)}")
    print(f"[{name}] boundary_labels 분포: {dict(b_counter)}")
    print(f"[{name}] particle_labels 분포: {dict(p_counter)}")

def build_class_weight_vector(labels: List[str], pivot_keep: str, non_keep_weight: float, device) -> torch.Tensor:
    V = len(labels)
    w = torch.ones(V, dtype=torch.float, device=device)
    if pivot_keep in labels:
        keep_id = labels.index(pivot_keep)
        for i in range(V):
            if i != keep_id:
                w[i] = non_keep_weight
    return w

def freeze_backbone(m: nn.Module, freeze: bool):
    for p in m.bert.parameters():
        p.requires_grad = not freeze

def train_and_eval():
    train_items = load_examples(TRAIN_JSON_PATH)
    val_items = load_examples(VAL_JSON_PATH) if VAL_JSON_PATH and os.path.exists(VAL_JSON_PATH) else []

    if not train_items:
        raise FileNotFoundError(f"학습 데이터를 찾을 수 없습니다: {TRAIN_JSON_PATH}")

    label_meta = collect_label_sets(train_items + val_items)

    # ===== 라벨 분포 진단 =====
    print_label_distribution(train_items, "train")
    print_label_distribution(val_items, "valid")

    # KoBERT 토크나이저 로드
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    add_map = {}
    if tok.special_tokens_map.get("unk_token") is None: add_map["unk_token"] = "[UNK]"
    if tok.special_tokens_map.get("sep_token") is None: add_map["sep_token"] = "[SEP]"
    if tok.special_tokens_map.get("cls_token") is None: add_map["cls_token"] = "[CLS]"
    if tok.special_tokens_map.get("pad_token") is None: add_map["pad_token"] = "[PAD]"
    if add_map: tok.add_special_tokens(add_map)

    # 검증 데이터 없으면 train에서 일부 분리
    if not val_items:
        random.shuffle(train_items)
        split = int(0.9 * len(train_items))
        train_items, val_items = train_items[:split], train_items[split:]

    # (선택) SPM 커버리지 점검
    check_spm_coverage(tok, train_items)

    train_ds = GECTagDataset(train_items, tok, label_meta, MAX_LEN)
    val_ds   = GECTagDataset(val_items, tok, label_meta, MAX_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = MultiHeadGECTagger(MODEL_PATH, label_meta).to(DEVICE)
    if hasattr(tok, "__len__") and len(tok) != model.config.vocab_size:
        model.bert.resize_token_embeddings(len(tok))

    # ===== 비-KEEP 가중치 벡터 =====
    token_class_weight = build_class_weight_vector(
        label_meta["token_labels"], "KEEP", TOKEN_NON_KEEP_WEIGHT, DEVICE
    )
    bound_class_weight = build_class_weight_vector(
        label_meta["boundary_labels"], "KEEP", BOUND_NON_KEEP_WEIGHT, DEVICE
    )

    # 손실 래퍼 (고정 파라미터 주입)
    def loss_f_token_fn(pred, tgt):
        return label_smoothed_loss(pred, tgt, epsilon=0.1, ignore_index=IGNORE_LABEL, class_weight=token_class_weight)
    def loss_f_bound_fn(pred, tgt):
        return label_smoothed_loss(pred, tgt, epsilon=0.1, ignore_index=IGNORE_LABEL, class_weight=bound_class_weight)

    # 옵티마이저/스케줄러
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    num_train_steps = EPOCHS * math.ceil(len(train_dl) / GRAD_ACCUM_STEPS)
    num_warmup = int(num_train_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_train_steps)

    id2token = label_meta["id2token"]
    id2bound = label_meta["id2bound"]
    id2part  = label_meta["id2part"]

    keep_id = label_meta["token2id"]["KEEP"]
    id_maps = {"id2token": id2token, "id2bound": id2bound, "id2part": id2part}

    best_score = -1.0

    for epoch in range(1, EPOCHS + 1):
        # Cold epochs: 백본 freeze
        freeze_backbone(model, freeze=(epoch <= COLD_EPOCHS))

        # ----------------- Train -----------------
        model.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(train_dl, 1), total=len(train_dl), desc=f"Epoch {epoch} [train]")
        for step, batch in pbar:
            for k in ("input_ids","attention_mask","token_type_ids",
                      "token_labels","boundary_labels","particle_labels"):
                batch[k] = batch[k].to(DEVICE)

            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],
                        token_labels=batch["token_labels"],
                        boundary_labels=batch["boundary_labels"],
                        particle_labels=batch["particle_labels"],
                        loss_f_token=loss_f_token_fn,
                        loss_f_bound=loss_f_bound_fn)

            loss = out["loss"] / GRAD_ACCUM_STEPS
            loss.backward()

            if CLIP_NORM is not None and CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

            total_loss += loss.item()
            if step % GRAD_ACCUM_STEPS == 0:
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            pbar.set_postfix(loss=f"{total_loss/step:.4f}")

        # ----------------- Valid -----------------
        model.eval()
        val_loss = 0.0
        acc_token = acc_bound = acc_part = 0.0
        pr_sum = rc_sum = f05_sum = 0.0
        n_batches = 0

        vbar = tqdm(enumerate(val_dl, 1), total=len(val_dl), desc=f"Epoch {epoch} [valid]")
        all_samples_buffer = []

        with torch.no_grad():
            for step, batch in vbar:
                for k in ("input_ids","attention_mask","token_type_ids",
                          "token_labels","boundary_labels","particle_labels"):
                    batch[k] = batch[k].to(DEVICE)

                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            token_type_ids=batch["token_type_ids"],
                            token_labels=batch["token_labels"],
                            boundary_labels=batch["boundary_labels"],
                            particle_labels=batch["particle_labels"])

                if out["loss"] is not None:
                    val_loss += out["loss"].item()

                acc_token += masked_accuracy(out["logits_token"], batch["token_labels"])
                acc_bound += masked_accuracy(out["logits_bound"], batch["boundary_labels"])
                acc_part  += masked_accuracy(out["logits_part"],  batch["particle_labels"])
                n_batches += 1

                # --------- 1회차: 배치 logits 기반 디코딩(임계치/바이어스 적용) ----------
                lt = out["logits_token"].cpu()
                lb = out["logits_bound"].cpu()
                lp = out["logits_part"].cpu()

                # 문장 단위 루프
                for b_idx, pieces in enumerate(batch["raw_pieces"]):
                    L = len(pieces)
                    # CLS/SEP 제외한 구간의 logits만 잘라서 1회차 디코딩
                    hyp, _, _, _ = decode_with_thresholds_from_logits(
                        lt[b_idx][1:1+L, :], lb[b_idx][1:1+L, :], lp[b_idx][1:1+L, :],
                        pieces, id2token, id2bound, id2part, keep_id
                    )
                    src = batch["metas"][b_idx].get("src", "")
                    tgt = batch["metas"][b_idx].get("tgt", "")

                    # --------- 반복(Iterative) 디코딩 2~N회 ----------
                    prev_hyp = hyp
                    for it in range(2, N_ITER + 1):
                        # hyp를 토크나이즈 → 단문 forward → 다시 디코드
                        hyp_pieces = tok.tokenize(prev_hyp)
                        new_hyp = forward_on_pieces(model, tok, hyp_pieces, id_maps, keep_id)
                        if new_hyp == prev_hyp:
                            break  # 변화 없으면 조기 종료
                        prev_hyp = new_hyp
                    hyp = prev_hyp

                    # ------- 문장 단위 PRF 누적 -------
                    p, r, f05 = prf05_from_edits(src, hyp, tgt)
                    pr_sum += p; rc_sum += r; f05_sum += f05

                    if len(all_samples_buffer) < NUM_SHOW_SAMPLES * 3:
                        all_samples_buffer.append((src, hyp, tgt))

                vbar.set_postfix(val_loss=f"{val_loss/max(1,n_batches):.4f}")

        # 평균 지표
        val_loss /= max(1, n_batches)
        acc_token /= max(1, n_batches)
        acc_bound /= max(1, n_batches)
        acc_part  /= max(1, n_batches)

        num_sent = len(val_ds)
        P = pr_sum / max(1, num_sent)
        R = rc_sum / max(1, num_sent)
        F05 = f05_sum / max(1, num_sent)

        logger.info(
            f"[Epoch {epoch}] "
            f"train_loss={total_loss/len(train_dl):.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"acc_token={acc_token:.4f} acc_bound={acc_bound:.4f} acc_part={acc_part:.4f} | "
            f"P={P:.4f} R={R:.4f} F0.5={F05:.4f}"
        )

        # ---- 샘플 5개 출력 ----
        print("\n=== 예시 문장 (무작위 5개) ===")
        random.shuffle(all_samples_buffer)
        for i, (src, hyp, tgt) in enumerate(all_samples_buffer[:NUM_SHOW_SAMPLES], 1):
            print(f"샘플 {i}:")
            print(f"> 입력: {src}")
            print(f"> 예측: {hyp}")
            print(f"> 정답: {tgt}\n")

        # ---- 베스트 저장 (F0.5 기준) ----
        score = F05
        if score > best_score:
            best_score = score
            save_dir = f"./ckpt_epoch{epoch}_F05_{best_score:.4f}"
            os.makedirs(save_dir, exist_ok=True)

            torch.save(
                {
                    "state_dict": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                    "label_meta": label_meta,
                    "backbone_name": MODEL_PATH,
                },
                os.path.join(save_dir, "model.ckpt")
            )
            try:
                tok.save_pretrained(save_dir)
            except TypeError:
                tok.save_vocabulary(save_dir)

            with open(os.path.join(save_dir, "label_meta.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "token_labels": label_meta["token_labels"],
                    "boundary_labels": label_meta["boundary_labels"],
                    "particle_labels": label_meta["particle_labels"]
                }, f, ensure_ascii=False, indent=2)

def load_checkpoint(ckpt_dir: str, device: torch.device = DEVICE) -> tuple:
    tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=False)
    payload = torch.load(os.path.join(ckpt_dir, "model.ckpt"), map_location=device)
    label_meta = payload["label_meta"]
    backbone_name = payload.get("backbone_name", "monologg/kobert")
    model = MultiHeadGECTagger(backbone_name, label_meta).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, tok, label_meta

if __name__ == "__main__":
    train_and_eval()
