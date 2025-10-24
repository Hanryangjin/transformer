# -*- coding: utf-8 -*-
"""
KoBERT(monologg/kobert) 기반 GEC 태깅 (GECToR 스타일 개선 적용)
- 데이터: ./transformer/TrainData/kogector_extended.json
- 변경 요약:
  (1) 조사 head 비활성화
  (2) 경계 head: 원본 인덱스 기준 결정 → 토큰 편집 시 재투영(mapping)
  (3) 임계치/바이어스는 토큰 head에만 적용
  (4) REPLACE_*/INSERT_* 라벨을 축약 액션(KEEP/DELETE/REPLACE/INSERT)으로 학습
      실제 치환/삽입 토큰은 후처리에서 MLM top-k로 선택(BERT MLM 헤드 사용)
  (5) cold epochs, class weight 등 기존 안정화 유지
  (6) OOV > 5%면 중단
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
    AutoModelForMaskedLM,
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
TRAIN_JSON_PATH = "./transformer/out_kobert2.jsonl"#"./transformer/TrainData/kogector_extended.jsonl"
VAL_JSON_PATH   = ""    # 비워두면 TRAIN에서 자동 분할(90/10)

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
MAX_LEN = 128
GRAD_ACCUM_STEPS = 1
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0  # grad clipping

# 멀티태스크 손실 가중치
LOSS_WEIGHT_TOKEN = 1.0
LOSS_WEIGHT_BOUND = 0.5
LOSS_WEIGHT_PART  = 0.0  # (요청 #1) 조사 head 완전 비활성화

# 평가/출력 관련
NUM_SHOW_SAMPLES = 5   # 매 에포크마다 예시 출력 개수
TOPK_MLM = 5           # 치환/삽입 후보 top-k

# ====== GECToR 스타일 하이퍼 ======
# 1) 토큰 헤드 임계치/바이어스 (경계/조사에는 적용하지 않음; 요청 #3)
MIN_ERROR_PROB = 0.30          # (1 - P(KEEP)) < MIN_ERROR_PROB 이면 KEEP 강제
ADDITIONAL_KEEP_LOGIT = 0.20   # KEEP 로그잇 가산 바이어스(>0이면 보수적)
# 2) 반복 추론 횟수 (검증 시): 문장 교정 1~N회
N_ITER = 3
# 3) 비-KEEP 가중 (불균형 완화; 경계는 낮게 시작)
TOKEN_NON_KEEP_WEIGHT = 3.0
BOUND_NON_KEEP_WEIGHT = 1.0
# 4) Cold epochs
COLD_EPOCHS = 1                # 처음 1 epoch 백본 freeze

# ----------------------------------------------------
# Label-smoothed loss (class weight 지원)
# ----------------------------------------------------
def label_smoothed_loss(pred, target, epsilon=0.02, ignore_index=-100, class_weight=None):
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
        smoothed = smoothed * class_weight.unsqueeze(0)
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

# ---- (요청 #4) 축약 액션 공간 매핑 ----
# 데이터의 token_labels가 REPLACE_x / INSERT_x 등을 포함해도,
# 학습에서는 아래 4클래스만 사용: KEEP / DELETE / REPLACE / INSERT
def normalize_token_label(lb: str) -> str:
    if lb == str(IGNORE_LABEL):
        return lb
    if lb.startswith("REPLACE_"):
        return "REPLACE"
    if lb.startswith("INSERT_"):
        return "INSERT"
    if lb in ("KEEP", "DELETE"):
        return lb
    # 기타는 보수적으로 KEEP
    return "KEEP"

def collect_label_sets(examples: List[Dict[str, Any]]):
    token_label_set, boundary_label_set, particle_label_set = set(), set(), set()
    for ex in examples:
        # --- token: 축약 액션 공간 반영 ---
        for lb in ex.get("token_labels", []):
            nl = normalize_token_label(lb)
            if nl != str(IGNORE_LABEL):
                token_label_set.add(nl)
        # --- boundary: 그대로 수집(KEEP/SPACE_INS/SPACE_DEL 등) ---
        for lb in ex.get("boundary_labels", []):
            if lb != str(IGNORE_LABEL):
                boundary_label_set.add(lb)
        # --- particle: 완전 비활성화(요청 #1). 그래도 집합 생성만 형태상 유지 ---
        # (실제 학습/디코딩에서는 사용하지 않음)
        for lb in ex.get("particle_labels", []):
            pass

    # 최소 보장
    token_label_set.update(["KEEP", "DELETE", "REPLACE", "INSERT"])
    boundary_label_set.update(["KEEP"])

    token_labels = sorted(list(token_label_set))
    boundary_labels = sorted(list(boundary_label_set))
    # particle은 비활성 → dummy
    particle_labels = ["KEEP"]

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

        def align_token_labels(raw):
            aligned = [IGNORE_LABEL]
            for lb in raw[: len(pieces)]:
                nl = normalize_token_label(lb)
                if nl == str(IGNORE_LABEL):
                    aligned.append(IGNORE_LABEL)
                else:
                    aligned.append(self.label_meta["token2id"].get(nl, self.label_meta["token2id"]["KEEP"]))
            aligned.append(IGNORE_LABEL)
            return aligned

        def align_labels(raw_labels, label2id):
            aligned = [IGNORE_LABEL]
            for lb in raw_labels[: len(pieces)]:
                if lb == str(IGNORE_LABEL):
                    aligned.append(IGNORE_LABEL)
                else:
                    aligned.append(label2id.get(lb, label2id.get("KEEP", 0)))
            aligned.append(IGNORE_LABEL)
            return aligned

        token_labels = align_token_labels(ex.get("token_labels", []))
        boundary_labels = align_labels(ex.get("boundary_labels", []), self.label_meta["bound2id"])

        # 조사 라벨은 비활성 → dummy
        particle_labels = [IGNORE_LABEL] * (len(pieces) + 2)

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
# KoBERT SPM 호환성 점검 + 중단(요청 #6)
# ----------------------------------------------------
def check_spm_coverage_or_die(tokenizer, samples, top_n=2000, oov_threshold_pct=5.0):
    unk_id = tokenizer.unk_token_id
    oov, total = 0, 0
    boundary_mismatch = 0
    n = min(len(samples), top_n)
    for ex in samples[:n]:
        pcs = ex["pieces"]
        ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + pcs + [tokenizer.sep_token])
        oov += sum(1 for t in ids if t == unk_id)
        total += len(ids)
        spm_like = sum(1 for p in pcs if p.startswith("▁"))
        if spm_like == 0:
            boundary_mismatch += 1
    oov_rate = (oov / max(1, total)) * 100
    msg = f"[SPM] OOV {oov_rate:.2f}% (threshold {oov_threshold_pct}%), boundary_mismatch {boundary_mismatch}"
    print(msg)
    if oov_rate > oov_threshold_pct:
        raise RuntimeError(f"OOV rate {oov_rate:.2f}% exceeds threshold {oov_threshold_pct}% — stop training.")

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
        # 조사 head 비활성 → 생성만 해도 되지만 여기서는 아예 사용하지 않음
        self.num_part  = 1

        self.dropout = nn.Dropout(getattr(self.config, "hidden_dropout_prob", 0.1))
        self.head_token = nn.Linear(hidden, self.num_token)
        self.head_bound = nn.Linear(hidden, self.num_bound)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                token_labels=None, boundary_labels=None,
                loss_f_token=None, loss_f_bound=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        seq_out = self.dropout(outputs.last_hidden_state)

        logits_token = self.head_token(seq_out)
        logits_bound = self.head_bound(seq_out)

        loss = None
        if (token_labels is not None) or (boundary_labels is not None):
            loss = 0.0
            if token_labels is not None:
                if loss_f_token is None:
                    loss_f_token = lambda pred, tgt: label_smoothed_loss(
                        pred, tgt, epsilon=0.02, ignore_index=IGNORE_LABEL
                    )
                loss += LOSS_WEIGHT_TOKEN * loss_f_token(
                    logits_token.view(-1, logits_token.size(-1)),
                    token_labels.view(-1)
                )
            if boundary_labels is not None:
                if loss_f_bound is None:
                    loss_f_bound = lambda pred, tgt: label_smoothed_loss(
                        pred, tgt, epsilon=0.0, ignore_index=IGNORE_LABEL
                    )
                loss += LOSS_WEIGHT_BOUND * loss_f_bound(
                    logits_bound.view(-1, logits_bound.size(-1)),
                    boundary_labels.view(-1)
                )
        return {
            "loss": loss,
            "logits_token": logits_token,
            "logits_bound": logits_bound,
        }

# ----------------------------------------------------
# 디코더/평가 유틸
# ----------------------------------------------------
def _strip_leading_bar(token: str) -> Tuple[str, bool]:
    if token.startswith("▁"):
        return token[1:], True  # (surface, space_before=True)
    return token, False

def build_space_before_from_spm(pieces: List[str]) -> List[bool]:
    # 각 piece의 기본 공백 여부: ▁ 존재 여부
    return [_strip_leading_bar(p)[1] for p in pieces]

def apply_boundary_overrides(space_before: List[bool], boundary_label_strs: List[str]) -> List[bool]:
    # boundary_labels[i]는 "토큰 i 앞"의 공백 결정을 의미한다고 정의
    out = space_before[:]
    for i in range(1, len(out)):
        if i < len(boundary_label_strs):
            b = boundary_label_strs[i]
            if b == "SPACE_INS":
                out[i] = True
            elif b == "SPACE_DEL":
                out[i] = False
            # KEEP은 그대로
    return out

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
# (요청 #3) 토큰 head에만 임계치/바이어스 적용한 1회 디코딩
# ----------------------------------------------------
@torch.no_grad()
def token_boundary_decode_onepass(
    lt: torch.Tensor, lb: torch.Tensor,
    pieces: List[str],
    id2token: Dict[int,str], id2bound: Dict[int,str],
    keep_id: int
) -> Tuple[List[str], List[str]]:
    """
    lt, lb: [L, Ct], [L, Cb]  (CLS/SEP 제외된 길이 L 입력)
    반환: token_action_strs[L], boundary_label_strs[L]
    """
    # --- 토큰: KEEP bias + 임계치 적용 ---
    biased_token_logits = lt.clone()
    biased_token_logits[..., keep_id] += ADDITIONAL_KEEP_LOGIT
    probs = torch.softmax(biased_token_logits, dim=-1)
    keep_prob = probs[..., keep_id]
    error_prob = 1.0 - keep_prob
    pred_token = probs.argmax(-1)  # [L]
    pred_token = torch.where(error_prob >= MIN_ERROR_PROB, pred_token, torch.full_like(pred_token, keep_id))
    token_strs = [id2token[int(x)] for x in pred_token]

    # --- 경계: 임계치/바이어스 미적용(요청 #3) ---
    bound_ids = lb.argmax(-1)
    boundary_strs = [id2bound[int(x)] for x in bound_ids]

    return token_strs, boundary_strs

# ----------------------------------------------------
# MLM 기반 치환/삽입 후보 선택
# ----------------------------------------------------
@torch.no_grad()
def mlm_replace_or_insert(tok, mlm_model, text_before: List[str], pos_piece_index: int,
                          mode: str, topk: int = 5) -> str:
    """
    text_before: 원본 pieces 문자열(▁ 포함) → 실제 문장으로 변환 후 MLM 질의
    pos_piece_index: 교체/삽입 기준 piece 인덱스 (원 인덱스)
    mode: "REPLACE" or "INSERT"
    반환: 선택된 후보 piece (▁ 유무 포함된 subword 문자열)
    """
    # 문장화(SPM 기준): ▁ → 공백
    def pieces_to_text(pcs):
        s = []
        for i, p in enumerate(pcs):
            surface, space = _strip_leading_bar(p)
            if i > 0 and (space):
                s.append(" ")
            s.append(surface)
        return "".join(s)

    pcs = text_before[:]  # 원본 pieces 복사
    text = pieces_to_text(pcs)

    # 토크나이저가 [MASK]를 지원해야 함
    mask_token = tok.mask_token
    if mask_token is None:
        # 안전장치: [MASK]가 없다면 그냥 원 토큰 유지
        return pcs[pos_piece_index] if mode == "REPLACE" else "▁"

    # 마스크 입력 구성
    if mode == "REPLACE":
        # 해당 piece의 표면 위치를 대강 근사: 다시 토크나이즈로 정렬
        # 간단 경로: pieces→문장→tokenize→piece 기준으로 재구성은 비용이 큼.
        # 여기서는 문자열 레벨 근사 대신 SentencePiece 단위로 다시 조합해 [MASK]를 삽입한다.
        pcs_masked = pcs[:]
        pcs_masked[pos_piece_index] = "▁" + mask_token if pcs_masked[pos_piece_index].startswith("▁") else mask_token
    else:  # INSERT (after pos)
        insert_idx = pos_piece_index + 1
        pcs_masked = pcs[:insert_idx] + (["▁"+mask_token] if (insert_idx < len(pcs) and pcs[insert_idx].startswith("▁")) else [mask_token]) + pcs[insert_idx:]

    text_m = pieces_to_text(pcs_masked)

    # MLM 질의
    enc = tok(text_m, return_tensors="pt").to(DEVICE)
    out = mlm_model(**enc)
    logits = out.logits  # [B, T, V]
    # 마스크 토큰 위치 찾기
    mask_id = tok.mask_token_id
    mask_positions = (enc["input_ids"] == mask_id).nonzero(as_tuple=False)
    if mask_positions.size(0) == 0:
        # 마스크를 못 찾으면 보수적으로 빈 토큰
        return "▁"

    # 첫 번째 마스크만 사용
    _, mpos = mask_positions[0].tolist()
    mlm_logits = logits[0, mpos, :]  # [V]
    topk_ids = torch.topk(mlm_logits, k=min(topk, mlm_logits.size(0))).indices.tolist()
    # 상위 후보 중 하나를 선택(여기서는 1위)
    cand_id = topk_ids[0]
    cand_token = tok.convert_ids_to_tokens(cand_id)  # subword(▁ 포함 가능)

    # KoBERT의 SentencePiece 토큰을 그대로 반환
    # 단, 없는 경우 대비
    if not isinstance(cand_token, str):
        return "▁"
    return cand_token

# ----------------------------------------------------
# 편집 적용(요청 #2): 원본 인덱스 기반 경계 → 편집 시 재투영
# ----------------------------------------------------
def decode_apply_edits_with_mapping(
    pieces: List[str],
    token_action_strs: List[str],   # ["KEEP","DELETE","REPLACE","INSERT",...]
    boundary_label_strs: List[str], # 길이 L (마지막 -100 제외 형태로 들어옴)
    tok, mlm_model
) -> str:
    """
    1) SPM 기반 space_before(원본) 계산
    2) boundary override를 "토큰 i 앞" 기준으로 먼저 적용 → space_before_src
    3) 원본 인덱스를 따라 토큰 편집을 적용하면서,
       출력 토큰의 space_before를 재투영(mapping)
    """
    L = len(pieces)
    # 1) 원본 기반 space_before (각 토큰 앞 공백 여부)
    space_before_src = build_space_before_from_spm(pieces)
    # 2) 경계 override
    space_before_src = apply_boundary_overrides(space_before_src, boundary_label_strs)

    # 3) 편집 적용
    out_tokens: List[Tuple[str, bool]] = []  # (surface, space_before)
    i = 0
    while i < L:
        piece = pieces[i]
        surface_i, _ = _strip_leading_bar(piece)
        sb_i = space_before_src[i]  # 토큰 i 앞의 공백 여부(원본 기준, override 반영)

        act = token_action_strs[i] if i < len(token_action_strs) else "KEEP"
        if act == "KEEP":
            out_tokens.append((surface_i, sb_i))
        elif act == "DELETE":
            # 다음 토큰의 space_before(원본 기준)는 그대로 유지 → 별 처리 없음
            pass
        elif act == "REPLACE":
            # MLM 후보로 교체 subword 결정
            cand = mlm_replace_or_insert(tok, mlm_model, pieces, i, mode="REPLACE", topk=TOPK_MLM)
            cand_surface, cand_space = _strip_leading_bar(cand)
            # 교체의 space_before는 원본 토큰의 sb_i를 우선 유지 (문맥 일관성)
            out_tokens.append((cand_surface, sb_i))
        elif act == "INSERT":
            # 원본 토큰 먼저 내보내고, 바로 뒤에 삽입
            out_tokens.append((surface_i, sb_i))
            cand = mlm_replace_or_insert(tok, mlm_model, pieces, i, mode="INSERT", topk=TOPK_MLM)
            cand_surface, cand_space = _strip_leading_bar(cand)
            # 삽입 토큰 앞 공백: cand_space를 존중(▁여부)하되, 너무 붙으면 가독성↓

            out_tokens.append((cand_surface, cand_space))
        else:
            # 알 수 없는 액션은 KEEP
            out_tokens.append((surface_i, sb_i))
        i += 1

    # 4) 문자열 조립
    s = []
    for k, (surf, space_b) in enumerate(out_tokens):
        if k == 0:
            s.append(surf)
        else:
            if space_b:
                s.append(" ")
            s.append(surf)
    return "".join(s)

# ----------------------------------------------------
# 학습/평가
# ----------------------------------------------------
def print_label_distribution(items, name):
    t_counter = Counter()
    b_counter = Counter()
    for ex in items:
        # token: 원본을 직접 집계(참고용)
        for lb in ex.get("token_labels", []):
            t_counter[normalize_token_label(lb)] += 1
        for lb in ex.get("boundary_labels", []):
            b_counter[lb] += 1
    print(f"[{name}] token_labels(축약) 분포: {dict(t_counter)}")
    print(f"[{name}] boundary_labels 분포: {dict(b_counter)}")

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

    # KoBERT 토크나이저 + MLM 모델(후처리용) 로드
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()

    add_map = {}
    if tok.special_tokens_map.get("unk_token") is None: add_map["unk_token"] = "[UNK]"
    if tok.special_tokens_map.get("sep_token") is None: add_map["sep_token"] = "[SEP]"
    if tok.special_tokens_map.get("cls_token") is None: add_map["cls_token"] = "[CLS]"
    if tok.special_tokens_map.get("pad_token") is None: add_map["pad_token"] = "[PAD]"
    if tok.special_tokens_map.get("mask_token") is None: add_map["mask_token"] = "[MASK]"
    if add_map: tok.add_special_tokens(add_map)

    # 검증 데이터 없으면 train에서 일부 분리
    if not val_items:
        random.shuffle(train_items)
        split = int(0.9 * len(train_items))
        train_items, val_items = train_items[:split], train_items[split:]

    # SPM 커버리지 검사(요청 #6) — 임계 초과 시 즉시 중단
    check_spm_coverage_or_die(tok, train_items, oov_threshold_pct=5.0)

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

    def loss_f_token_fn(pred, tgt):
        return label_smoothed_loss(pred, tgt, epsilon=0.02, ignore_index=IGNORE_LABEL, class_weight=token_class_weight)
    def loss_f_bound_fn(pred, tgt):
        return label_smoothed_loss(pred, tgt, epsilon=0.0, ignore_index=IGNORE_LABEL, class_weight=bound_class_weight)

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

    keep_id = label_meta["token2id"]["KEEP"]

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
                      "token_labels","boundary_labels"):
                batch[k] = batch[k].to(DEVICE)

            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],
                        token_labels=batch["token_labels"],
                        boundary_labels=batch["boundary_labels"],
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
        acc_token = acc_bound = 0.0
        pr_sum = rc_sum = f05_sum = 0.0
        n_batches = 0

        vbar = tqdm(enumerate(val_dl, 1), total=len(val_dl), desc=f"Epoch {epoch} [valid]")
        all_samples_buffer = []

        with torch.no_grad():
            for step, batch in vbar:
                for k in ("input_ids","attention_mask","token_type_ids",
                          "token_labels","boundary_labels"):
                    batch[k] = batch[k].to(DEVICE)

                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            token_type_ids=batch["token_type_ids"],
                            token_labels=batch["token_labels"],
                            boundary_labels=batch["boundary_labels"])

                if out["loss"] is not None:
                    val_loss += out["loss"].item()

                acc_token += masked_accuracy(out["logits_token"], batch["token_labels"])
                acc_bound += masked_accuracy(out["logits_bound"], batch["boundary_labels"])
                n_batches += 1

                # --------- 1회차: 배치 logits 기반 디코딩 ----------
                lt = out["logits_token"].cpu()
                lb = out["logits_bound"].cpu()

                for b_idx, pieces in enumerate(batch["raw_pieces"]):
                    L = len(pieces)
                    # (요청 #3) 토큰 head에만 임계치/바이어스 적용
                    token_actions, boundary_strs = token_boundary_decode_onepass(
                        lt[b_idx][1:1+L, :], lb[b_idx][1:1+L, :],
                        pieces, id2token, id2bound, keep_id
                    )
                    # 편집 적용(+경계 재투영; 요청 #2)
                    hyp = decode_apply_edits_with_mapping(pieces, token_actions, boundary_strs, tok, mlm_model)

                    src = batch["metas"][b_idx].get("src", "")
                    tgt = batch["metas"][b_idx].get("tgt", "")

                    # --------- 반복(Iterative) 디코딩 ----------
                    prev_hyp = hyp
                    for it in range(2, N_ITER + 1):
                        hyp_pieces = tok.tokenize(prev_hyp)
                        # 단문 forward
                        enc = tok.convert_tokens_to_ids([tok.cls_token] + hyp_pieces + [tok.sep_token])
                        attn = [1]*len(enc); type_ids = [0]*len(enc)
                        enc = torch.tensor(enc, dtype=torch.long, device=DEVICE).unsqueeze(0)
                        attn = torch.tensor(attn, dtype=torch.long, device=DEVICE).unsqueeze(0)
                        type_ids = torch.tensor(type_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
                        out2 = model(input_ids=enc, attention_mask=attn, token_type_ids=type_ids)

                        lt2 = out2["logits_token"][0, 1:-1, :].cpu()
                        lb2 = out2["logits_bound"][0, 1:-1, :].cpu()

                        token_actions2, boundary_strs2 = token_boundary_decode_onepass(
                            lt2, lb2, hyp_pieces, id2token, id2bound, keep_id
                        )
                        new_hyp = decode_apply_edits_with_mapping(hyp_pieces, token_actions2, boundary_strs2, tok, mlm_model)
                        if new_hyp == prev_hyp:
                            break
                        prev_hyp = new_hyp
                    hyp = prev_hyp

                    p, r, f05 = prf05_from_edits(src, hyp, tgt)
                    pr_sum += p; rc_sum += r; f05_sum += f05

                    if len(all_samples_buffer) < NUM_SHOW_SAMPLES * 3:
                        all_samples_buffer.append((src, hyp, tgt))

                vbar.set_postfix(val_loss=f"{val_loss/max(1,n_batches):.4f}")

        # 평균 지표
        val_loss /= max(1, n_batches)
        acc_token /= max(1, n_batches)
        acc_bound /= max(1, n_batches)

        num_sent = len(val_ds)
        P = pr_sum / max(1, num_sent)
        R = rc_sum / max(1, num_sent)
        F05 = f05_sum / max(1, num_sent)

        logger.info(
            f"[Epoch {epoch}] "
            f"train_loss={total_loss/len(train_dl):.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"acc_token={acc_token:.4f} acc_bound={acc_bound:.4f} | "
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
