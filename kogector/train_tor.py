#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_multi_head_gec.py

- token_labels / space_labels / particle_labels 를 동시에 다루는 멀티헤드 학습 코드.
- train_one_head.py 구조를 확장한 버전.
- CASE 2~4(토큰+띄어쓰기+조사) 모두 이 파일로 처리 가능.

사용 예시:

# CASE 2: token + space 학습
python3 train_multi_head_gec.py \
  --train ./transformer/out_kobert2.jsonl \
  --valid None \
  --model monologg/kobert \
  --outdir ./runs/kobert_token_space \
  --use_space --lambda_space 1.0

# CASE 3: token + particle 학습
python3 train_multi_head_gec.py \
  --train ./transformer/out_kobert2.jsonl \
  --valid None \
  --model monologg/kobert \
  --outdir ./runs/kobert_token_particle \
  --use_particle --lambda_particle 1.0

# CASE 4: token + space + particle 학습
python3 train_multi_head_gec.py \
  --train ./transformer/out_kobert2.jsonl \
  --valid None \
  --model monologg/kobert \
  --outdir ./runs/kobert_all_heads \
  --use_space --use_particle \
  --lambda_space 1.0 --lambda_particle 1.0
"""

import os
import json
import random
import argparse
import time
from collections import Counter
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

IGNORE_INDEX = -100


# ==============================
# 유틸 / 라벨 메타
# ==============================
def normalize_token_label_for_display(lb: str) -> str:
    if not isinstance(lb, str):
        return "KEEP"
    s = lb.strip()
    if s.startswith("APPEND_"):
        return "INSERT"
    if s.startswith("INSERT_") or s.startswith("INSERT"):
        return "INSERT"
    if s.startswith("REPLACE_") or s.startswith("REPLACE"):
        return "REPLACE"
    if s in ("KEEP", "DELETE"):
        return s
    if s in ("REPLACE_UNK", "INSERT_UNK"):
        return s.replace("_UNK", "")
    return "KEEP"


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            items = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def _prefix_fix_raw_label(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.startswith("APPEND_"):
        s = s.replace("APPEND_", "INSERT_", 1)
    if s.startswith("REPLACE") and not s.startswith("REPLACE_"):
        s = s.replace("REPLACE", "REPLACE_", 1)
    if s.startswith("INSERT") and not s.startswith("INSERT_"):
        s = s.replace("INSERT", "INSERT_", 1)
    return s


def _collapse_composite_label(s: str) -> str:
    """
    복합 태그("REPLACE_...|INSERT_...")를 단일 의미 태그로 축약.
    우선순위: REPLACE_* > INSERT_*/APPEND_* > DELETE > KEEP
    """
    if not isinstance(s, str) or "|" not in s:
        return s
    parts = [p.strip() for p in s.split("|") if p.strip()]
    for p in parts:
        if p.startswith("REPLACE_") or p.startswith("REPLACE"):
            return _prefix_fix_raw_label(p)
    for p in parts:
        if p.startswith("APPEND_"):
            return _prefix_fix_raw_label(p)
        if p.startswith("INSERT_") or p.startswith("INSERT"):
            return _prefix_fix_raw_label(p)
    if "DELETE" in parts:
        return "DELETE"
    if "KEEP" in parts:
        return "KEEP"
    return _prefix_fix_raw_label(parts[0])


def print_label_distribution(items: List[Dict[str, Any]], name: str):
    raw = Counter()
    disp = Counter()
    for ex in items:
        for lb in ex.get("token_labels", []):
            raw[lb] += 1
            disp[normalize_token_label_for_display(lb)] += 1
    print(f"[{name}] token_labels(원형) 상위 20개: {raw.most_common(20)}")
    print(f"[{name}] token_labels(표시용 축약): {dict(disp)}")


def build_token_label_meta(items: List[Dict[str, Any]],
                           topn_replace: int = None,
                           topn_insert: int = None) -> Dict[str, Any]:
    cnt = Counter()
    for ex in items:
        for lb in ex["token_labels"]:
            if lb:
                cnt[lb] += 1

    if topn_replace is not None:
        keep = set([l for l, _ in cnt.most_common() if l.startswith("REPLACE_")][:topn_replace])
        for l in list(cnt.keys()):
            if l.startswith("REPLACE_") and l not in keep:
                cnt["REPLACE_UNK"] += cnt.pop(l)
    if topn_insert is not None:
        keep = set([l for l, _ in cnt.most_common() if l.startswith("INSERT_")][:topn_insert])
        for l in list(cnt.keys()):
            if l.startswith("INSERT_") and l not in keep:
                cnt["INSERT_UNK"] += cnt.pop(l)

    cnt["KEEP"] += 0
    cnt["DELETE"] += 0

    labels = sorted(cnt.keys())
    token2id = {lb: i for i, lb in enumerate(labels)}
    id2token = {i: lb for lb, i in token2id.items()}
    print(f"[meta-token] 라벨 크기={len(labels)} (예: 앞 30) {labels[:30]}")
    return {"labels": labels, "token2id": token2id, "id2token": id2token}


def build_simple_label_meta(items: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """
    space_labels, particle_labels 같이 종류가 적은 라벨들의 어휘를 만든다.
    """
    cnt = Counter()
    for ex in items:
        seq = ex.get(key)
        if not isinstance(seq, list):
            continue
        for lb in seq:
            if lb is not None:
                cnt[str(lb)] += 1
    if not cnt:
        return {"labels": [], "token2id": {}, "id2token": {}}
    labels = sorted(cnt.keys())
    token2id = {lb: i for i, lb in enumerate(labels)}
    id2token = {i: lb for lb, i in token2id.items()}
    print(f"[meta-{key}] 라벨 크기={len(labels)}: {labels}")
    return {"labels": labels, "token2id": token2id, "id2token": id2token}


# ==============================
# 띄어쓰기 평가용 헬퍼
# ==============================
def _space_boundaries(s: str):
    positions = []
    for idx, ch in enumerate(s):
        if ch != " ":
            positions.append(idx)
    n = len(positions)
    if n <= 1:
        return []
    boundaries = []
    for k in range(n - 1):
        i = positions[k]
        j = positions[k + 1]
        has_space = False
        for t in range(i + 1, j):
            if s[t] == " ":
                has_space = True
                break
        boundaries.append(has_space)
    return boundaries


def _space_accuracy(pred_texts, gold_texts) -> float:
    total = 0
    correct = 0
    for pred, gold in zip(pred_texts, gold_texts):
        if not pred or not gold:
            continue
        pred_chars = [c for c in pred if c != " "]
        gold_chars = [c for c in gold if c != " "]
        if pred_chars != gold_chars:
            continue
        b_pred = _space_boundaries(pred)
        b_gold = _space_boundaries(gold)
        if not b_pred or not b_gold:
            continue
        m = min(len(b_pred), len(b_gold))
        for i in range(m):
            total += 1
            if b_pred[i] == b_gold[i]:
                correct += 1
    return correct / max(1, total)


# ==============================
# Dataset / Collate
# ==============================
class MultiHeadDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        tokenizer,
        token2id: Dict[str, int],
        space2id: Dict[str, int],
        particle2id: Dict[str, int],
        max_len: int,
    ):
        self.items = items
        self.tokenizer = tokenizer
        self.t2i = token2id
        self.s2i = space2id
        self.p2i = particle2id
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        pieces = ex.get("pieces")
        if pieces is None:
            if isinstance(ex.get("src_token"), list) and ex["src_token"]:
                pieces = ex["src_token"]
            else:
                src = ex.get("meta", {}).get("src") or ex.get("src") or ""
                pieces = self.tokenizer.tokenize(src)

        token_labels = ex.get("token_labels") or []
        token_labels = [_collapse_composite_label(_prefix_fix_raw_label(lb)) for lb in token_labels]
        if len(pieces) != len(token_labels):
            raise ValueError(
                f"[MultiHeadDataset] token_labels 길이 불일치: pieces={len(pieces)}, token_labels={len(token_labels)}"
            )

        # space_labels
        space_labels = ex.get("space_labels")
        if isinstance(space_labels, list) and len(space_labels) == len(pieces):
            space_ids = [self.s2i.get(str(lb), IGNORE_INDEX) for lb in space_labels]
        else:
            space_ids = [IGNORE_INDEX] * len(pieces)

        # particle_labels
        particle_labels = ex.get("particle_labels")
        if isinstance(particle_labels, list) and len(particle_labels) == len(pieces):
            particle_ids = [self.p2i.get(str(lb), IGNORE_INDEX) for lb in particle_labels]
        else:
            particle_ids = [IGNORE_INDEX] * len(pieces)

        # 이미 sentencepiece 토큰이므로 재토크나이즈 없이 id 변환만
        input_ids = self.tokenizer.convert_tokens_to_ids(pieces)

        # truncation
        input_ids = input_ids[: self.max_len]
        token_labels = token_labels[: self.max_len]
        space_ids = space_ids[: self.max_len]
        particle_ids = particle_ids[: self.max_len]

        label_token_ids = [self.t2i.get(lb, self.t2i.get("KEEP")) for lb in token_labels]
        attn_mask = [1] * len(input_ids)

        meta = ex.get("meta", {})

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "label_token_ids": torch.tensor(label_token_ids, dtype=torch.long),
            "label_space_ids": torch.tensor(space_ids, dtype=torch.long),
            "label_particle_ids": torch.tensor(particle_ids, dtype=torch.long),
            "pieces": pieces[: self.max_len],
            "token_labels": token_labels,
            "meta": meta,
        }


def collate_multi(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(t, val):
        pad_len = max_len - len(t)
        if pad_len > 0:
            t = torch.cat([t, torch.full((pad_len,), val, dtype=t.dtype)], dim=0)
        return t

    input_ids = torch.stack([pad(b["input_ids"], pad_id) for b in batch])
    attention_mask = torch.stack([pad(b["attention_mask"], 0) for b in batch])
    label_token_ids = torch.stack([pad(b["label_token_ids"], IGNORE_INDEX) for b in batch])
    label_space_ids = torch.stack([pad(b["label_space_ids"], IGNORE_INDEX) for b in batch])
    label_particle_ids = torch.stack([pad(b["label_particle_ids"], IGNORE_INDEX) for b in batch])

    pieces = [b["pieces"] + ["[PAD]"] * (max_len - len(b["pieces"])) for b in batch]
    token_labels = [b["token_labels"] + ["KEEP"] * (max_len - len(b["token_labels"])) for b in batch]
    metas = [b.get("meta", {}) for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_token_ids": label_token_ids,
        "label_space_ids": label_space_ids,
        "label_particle_ids": label_particle_ids,
        "pieces": pieces,
        "token_labels": token_labels,
        "metas": metas,
    }


# ==============================
# 모델 (최대 3헤드)
# ==============================
class MultiHeadTagger(nn.Module):
    def __init__(self, model_name: str, num_token_labels: int, num_space_labels: int, num_particle_labels: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_token = nn.Linear(hidden, num_token_labels)
        self.head_space = nn.Linear(hidden, num_space_labels) if num_space_labels > 0 else None
        self.head_particle = nn.Linear(hidden, num_particle_labels) if num_particle_labels > 0 else None

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        seq = self.dropout(seq)
        logits_token = self.head_token(seq)
        logits_space = self.head_space(seq) if self.head_space is not None else None
        logits_particle = self.head_particle(seq) if self.head_particle is not None else None
        return logits_token, logits_space, logits_particle


# ==============================
# 디코딩 유틸
# ==============================
def _strip_leading(tok: str) -> Tuple[str, bool]:
    if tok.startswith("▁"):
        return tok[1:], True
    if tok.startswith("##"):
        return tok[2:], False
    return tok, False


def decode_apply_token_actions(pieces: List[str], actions: List[str]) -> str:
    """
    space head를 쓰지 않을 때(또는 CASE 1/3) 사용하는 기본 디코더.
    KoBERT의 ▁ 정보를 그대로 활용.
    """
    out: List[Tuple[str, bool]] = []
    for i, p in enumerate(pieces):
        act = actions[i] if i < len(actions) else "KEEP"
        surf, sp = _strip_leading(p)
        if act == "KEEP":
            out.append((surf, sp))
        elif act == "DELETE":
            continue
        elif act.startswith("REPLACE_"):
            tgt = act[len("REPLACE_") :]
            ts, _tsp = _strip_leading(tgt)
            out.append((ts, sp))
        elif act.startswith("INSERT_"):
            out.append((surf, sp))
            tgt = act[len("INSERT_") :]
            ts, _tsp = _strip_leading(tgt)
            out.append((ts, True))
        elif act in ("REPLACE_UNK", "INSERT_UNK"):
            out.append((surf, sp))
        else:
            out.append((surf, sp))

    s = []
    for k, (ts, sp) in enumerate(out):
        if k == 0:
            s.append(ts)
        else:
            s.append((" " if sp else "") + ts)
    return "".join(s)


def decode_with_space_labels(
    pieces: List[str],
    token_actions: List[str],
    space_actions: List[str],
    default_space_label: str = "SPACE_KEEP",
) -> str:
    """
    token_labels + space_labels를 함께 사용해 최종 문장을 복원.

    - 토큰 내부의 ▁ 여부는 모두 무시.
    - 각 토큰 앞 공백 여부는 space_actions로만 결정:
      * SPACE_KEEP   : 입력 토큰의 ▁ 여부 유지
      * SPACE_INSERT : 무조건 공백 삽입
      * SPACE_DELETE : 무조건 공백 없음
    - INSERT_x 로 삽입되는 토큰은 앞에 공백(True)을 주는 것으로 처리.
    """
    out: List[Tuple[str, bool]] = []

    for i, piece in enumerate(pieces):
        act = token_actions[i] if i < len(token_actions) else "KEEP"
        sp_lb = space_actions[i] if i < len(space_actions) else default_space_label

        base_sp = piece.startswith("▁")  # 원래 입력 토큰의 공백 플래그
        if sp_lb == "SPACE_KEEP":
            has_space = base_sp
        elif sp_lb == "SPACE_INSERT":
            has_space = True
        elif sp_lb == "SPACE_DELETE":
            has_space = False
        else:
            has_space = base_sp

        surf, _ = _strip_leading(piece)

        if act == "KEEP":
            out.append((surf, has_space))

        elif act == "DELETE":
            continue

        elif act.startswith("REPLACE_"):
            tgt = act[len("REPLACE_") :]
            ts, _ = _strip_leading(tgt)
            out.append((ts, has_space))

        elif act.startswith("INSERT_"):
            tgt = act[len("INSERT_") :]
            ts, _ = _strip_leading(tgt)
            out.append((surf, has_space))   # 원래 토큰
            out.append((ts, True))          # 새 토큰은 앞에 공백

        elif act in ("REPLACE_UNK", "INSERT_UNK"):
            out.append((surf, has_space))

        else:
            out.append((surf, has_space))

    text = []
    first = True
    for ts, sp in out:
        if first:
            text.append(ts)
            first = False
        else:
            text.append((" " if sp else "") + ts)
    return "".join(text)


def edits_from_actions(pieces: List[str], actions: List[str]) -> List[Tuple[int, str, str]]:
    E = []
    for i, act in enumerate(actions):
        if act.startswith("REPLACE_"):
            E.append((i, "R", act[len("REPLACE_") :]))
        elif act.startswith("INSERT_"):
            E.append((i, "I", act[len("INSERT_") :]))
        elif act == "DELETE":
            E.append((i, "D", ""))
    return E


def prf_from_editsets(pred_edits, gold_edits, beta: float = 0.5):
    pred_set = set(pred_edits)
    gold_set = set(gold_edits)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    P = tp / (tp + fp + 1e-8)
    R = tp / (tp + fn + 1e-8)
    b2 = beta * beta
    F = (1 + b2) * P * R / (b2 * P + R + 1e-8)
    return P, R, F


def _reconstruct_src_from_pieces(pieces: List[str]) -> str:
    s = []
    for i, p in enumerate(pieces):
        tok, sp = _strip_leading(p)
        if i == 0:
            s.append(tok)
        else:
            s.append((" " if sp else "") + tok)
    return "".join(s)


# ==============================
# 평가 (토큰 + 띄어쓰기 + 조사)
# ==============================
def evaluate(
    model,
    loader,
    id2tok,
    tokenizer,
    device,
    space2id,
    id2space,
    particle2id,
    id2particle,
    use_space: bool,
    use_particle: bool,
):
    model.eval()

    n_tok = n_tok_correct = 0
    n_edit = n_edit_correct = 0

    P_all = R_all = F_all = 0.0
    n_sent = 0

    all_pred_texts = []
    all_gold_texts = []

    n_part_all = n_part_all_correct = 0
    n_part_is = n_part_is_correct = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y_tok = batch["label_token_ids"].to(device)
            y_space = batch["label_space_ids"].to(device)
            y_part = batch["label_particle_ids"].to(device)

            logits_tok, logits_space, logits_part = model(input_ids, attn)
            pred_tok_ids = torch.argmax(logits_tok, dim=-1)

            mask = y_tok.ne(IGNORE_INDEX) & attn.ne(0)
            n_tok += mask.sum().item()
            n_tok_correct += (pred_tok_ids.eq(y_tok) & mask).sum().item()

            keep_id = None
            for k, v in id2tok.items():
                if v == "KEEP":
                    keep_id = k
                    break
            if keep_id is not None:
                edit_mask = mask & (~(y_tok == keep_id))
            else:
                edit_mask = mask
            n_edit += edit_mask.sum().item()
            n_edit_correct += ((pred_tok_ids.eq(y_tok)) & edit_mask).sum().item()

            B, Lmax = input_ids.size()
            for b in range(B):
                valid_len = mask[b].sum().item()
                if valid_len == 0:
                    continue

                pieces = batch["pieces"][b][:valid_len]
                meta = batch["metas"][b] if "metas" in batch else {}

                gold_actions = [id2tok[y_tok[b, i].item()] for i in range(valid_len)]
                pred_actions = [id2tok[pred_tok_ids[b, i].item()] for i in range(valid_len)]

                gold_ed = edits_from_actions(pieces, gold_actions)
                pred_ed = edits_from_actions(pieces, pred_actions)
                P, R, F = prf_from_editsets(pred_ed, gold_ed, beta=0.5)
                P_all += P
                R_all += R
                F_all += F
                n_sent += 1

                # Gold 텍스트
                gold_text = meta.get("tgt") or decode_apply_token_actions(pieces, gold_actions)

                # Pred 텍스트: use_space=True면 space_labels 반영
                if use_space and logits_space is not None:
                    space_row = logits_space[b, :valid_len, :]
                    pred_space_ids = torch.argmax(space_row, dim=-1).tolist()
                    pred_space_labels = [id2space[idx] for idx in pred_space_ids]
                    pred_text = decode_with_space_labels(pieces, pred_actions, pred_space_labels)
                else:
                    pred_text = decode_apply_token_actions(pieces, pred_actions)

                all_gold_texts.append(gold_text)
                all_pred_texts.append(pred_text)

                # 조사 분류 정확도
                if use_particle and logits_part is not None:
                    y_part_row = y_part[b, :valid_len]
                    pred_part_row = torch.argmax(logits_part[b, :valid_len, :], dim=-1)
                    for i in range(valid_len):
                        gold_id = y_part_row[i].item()
                        if gold_id == IGNORE_INDEX:
                            continue
                        pred_id = pred_part_row[i].item()
                        n_part_all += 1
                        if gold_id == pred_id:
                            n_part_all_correct += 1

                        gold_tag = id2particle[gold_id]
                        if gold_tag.startswith("PARTICLE_IS"):
                            if pred_id == gold_id:
                                n_part_is_correct += 1
                            n_part_is += 1

    acc_tok = n_tok_correct / max(1, n_tok)
    acc_edit = n_edit_correct / max(1, n_edit)
    P = P_all / max(1, n_sent)
    R = R_all / max(1, n_sent)
    F = F_all / max(1, n_sent)
    space_acc = _space_accuracy(all_pred_texts, all_gold_texts)

    if use_particle and n_part_all > 0:
        particle_acc_all = n_part_all_correct / n_part_all
    else:
        particle_acc_all = None

    if use_particle and n_part_is > 0:
        particle_acc_is = n_part_is_correct / n_part_is
    else:
        particle_acc_is = None

    return acc_tok, acc_edit, P, R, F, space_acc, particle_acc_all, particle_acc_is


def preview_samples(
    model,
    tokenizer,
    id2tok,
    device,
    items: List[Dict[str, Any]],
    title: str,
    use_space: bool,
    id2space: Dict[int, str],
    max_len: int = 256,
    k: int = 5,
):
    print(f"\n=== [{title}] 예시 문장 (무작위 {k}개) ===")
    if not items:
        print("(빈 데이터)")
        return
    model.eval()
    with torch.no_grad():
        for idx, ex in enumerate(random.sample(items, min(k, len(items))), start=1):
            pieces = ex.get("pieces") or ex.get("src_token") or []
            labels = ex.get("token_labels") or []
            meta = ex.get("meta", {})
            if not pieces or not labels:
                continue
            pieces = pieces[:max_len]
            labels = labels[:max_len]

            inp_text = meta.get("src") or _reconstruct_src_from_pieces(pieces)
            gold_actions = labels
            gold_text = meta.get("tgt") or decode_apply_token_actions(pieces, gold_actions)

            input_ids = tokenizer.convert_tokens_to_ids(pieces)
            attn = [1] * len(input_ids)
            input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
            attn_t = torch.tensor([attn], dtype=torch.long, device=device)
            logits_tok, logits_space, _ = model(input_t, attn_t)
            pred_ids = torch.argmax(logits_tok, dim=-1)[0].tolist()
            pred_actions = [id2tok[i] for i in pred_ids[: len(pieces)]]

            if use_space and logits_space is not None and logits_space.size(-1) > 0:
                space_row = logits_space[0, : len(pieces), :]
                pred_space_ids = torch.argmax(space_row, dim=-1).tolist()
                pred_space_labels = [id2space[idx] for idx in pred_space_ids]
                pred_text = decode_with_space_labels(pieces, pred_actions, pred_space_labels)
            else:
                pred_text = decode_apply_token_actions(pieces, pred_actions)

            print(f"샘플 {idx}:")
            print(f"> 입력 : {inp_text}")
            print(f"> 예측 : {pred_text}")
            print(f"> 정답 : {gold_text}\n")


# ==============================
# 학습 루프
# ==============================
def _eta_from_pbar(start: float, n_done: int, total: int) -> str:
    if n_done <= 0:
        return "--:--"
    elapsed = time.time() - start
    rate = elapsed / max(1, n_done)
    remain = rate * max(0, total - n_done)
    h = int(remain // 3600)
    m = int((remain % 3600) // 60)
    s = int(remain % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    # ----- 데이터 로드 & 전처 스크리닝 -----
    if args.train is None or not os.path.exists(args.train):
        raise FileNotFoundError(f"--train 경로가 없습니다: {args.train}")
    train_items_all = read_json_or_jsonl(args.train)

    valid_items = None
    if args.valid is not None and str(args.valid).lower() != "none":
        if not os.path.exists(args.valid):
            raise FileNotFoundError(f"--valid 경로가 없습니다: {args.valid}")
        valid_items = read_json_or_jsonl(args.valid)

    def _pre_screen(items):
        good = []
        skipped = 0
        for rec in items:
            pieces = rec.get("pieces")
            if pieces is None:
                if isinstance(rec.get("src_token"), list) and rec["src_token"]:
                    pieces = rec["src_token"]
                else:
                    src = rec.get("meta", {}).get("src") or rec.get("src") or ""
                    if not src:
                        skipped += 1
                        continue
                    pieces = tok.tokenize(src)
            token_labels = rec.get("token_labels") or []
            token_labels = [_collapse_composite_label(_prefix_fix_raw_label(x)) for x in token_labels]
            if len(pieces) != len(token_labels):
                skipped += 1
                continue
            rec["pieces"] = pieces
            rec["token_labels"] = token_labels
            good.append(rec)
        if skipped:
            print(f"[pre] 스킵 {skipped}개 (길이 불일치/결측)")
        return good

    train_items_all = _pre_screen(train_items_all)

    if valid_items is None:
        rand = random.Random(args.split_seed)
        idxs = list(range(len(train_items_all)))
        rand.shuffle(idxs)
        cut = max(1, int(len(idxs) * 0.1))
        valid_idx = set(idxs[:cut])
        train_items = [train_items_all[i] for i in idxs[cut:]]
        valid_items = [train_items_all[i] for i in idxs[:cut]]
        print(f"[split] valid=None → train {len(train_items)} / valid {len(valid_items)} (10%)")
    else:
        train_items = train_items_all
        valid_items = _pre_screen(valid_items)

    print_label_distribution(train_items, "train")
    print_label_distribution(valid_items, "valid")

    # ----- 라벨 메타 -----
    token_meta = build_token_label_meta(train_items, topn_replace=args.topn_replace, topn_insert=args.topn_insert)
    token2id = token_meta["token2id"]
    id2token = token_meta["id2token"]

    space_meta = build_simple_label_meta(train_items, "space_labels")
    space2id = space_meta["token2id"]
    id2space = space_meta["id2token"]

    particle_meta = build_simple_label_meta(train_items, "particle_labels")
    particle2id = particle_meta["token2id"]
    id2particle = particle_meta["id2token"]

    train_ds = MultiHeadDataset(train_items, tok, token2id, space2id, particle2id, args.max_len)
    valid_ds = MultiHeadDataset(valid_items, tok, token2id, space2id, particle2id, args.max_len)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    train_dl = DataLoader(
        train_ds,
        batch_size=args.bsz,
        shuffle=True,
        collate_fn=lambda b: collate_multi(b, pad_id),
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.bsz,
        shuffle=False,
        collate_fn=lambda b: collate_multi(b, pad_id),
    )

    num_space_labels = len(space_meta["labels"]) if args.use_space else 0
    num_particle_labels = len(particle_meta["labels"]) if args.use_particle else 0

    model = MultiHeadTagger(
        args.model,
        num_token_labels=len(token_meta["labels"]),
        num_space_labels=num_space_labels,
        num_particle_labels=num_particle_labels,
    ).to(device)

    # ----- loss 세팅 -----
    weight_tok = None
    if args.non_keep_weight > 0:
        w = torch.ones(len(token_meta["labels"]))
        keep_id = token2id.get("KEEP", None)
        if keep_id is not None:
            w[keep_id] = 1.0
            non_keep_ids = [i for i in range(len(token_meta["labels"])) if i != keep_id]
            w[non_keep_ids] = args.non_keep_weight
        weight_tok = w.to(device)
    criterion_tok = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=weight_tok)

    criterion_space = None
    if args.use_space and num_space_labels > 0:
        criterion_space = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    criterion_part = None
    if args.use_particle and num_particle_labels > 0:
        criterion_part = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    best_f = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        t0 = time.time()

        pbar = tqdm(train_dl, desc=f"Epoch {ep} [train]", ncols=120)
        pbar_start = time.time()

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y_tok = batch["label_token_ids"].to(device)
            y_space = batch["label_space_ids"].to(device)
            y_part = batch["label_particle_ids"].to(device)

            logits_tok, logits_space, logits_part = model(input_ids, attn)

            loss = 0.0
            loss_tok = criterion_tok(logits_tok.view(-1, logits_tok.size(-1)), y_tok.view(-1))
            loss = loss + loss_tok

            if args.use_space and criterion_space is not None and logits_space is not None:
                loss_space = criterion_space(
                    logits_space.view(-1, logits_space.size(-1)), y_space.view(-1)
                )
                loss = loss + args.lambda_space * loss_space
            else:
                loss_space = torch.tensor(0.0, device=device)

            if args.use_particle and criterion_part is not None and logits_part is not None:
                loss_part = criterion_part(
                    logits_part.view(-1, logits_part.size(-1)), y_part.view(-1)
                )
                loss = loss + args.lambda_particle * loss_part
            else:
                loss_part = torch.tensor(0.0, device=device)

            opt.zero_grad()
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            sched.step()

            loss_sum += loss.item()
            lr_now = opt.param_groups[0]["lr"]
            eta = _eta_from_pbar(pbar_start, pbar.n, pbar.total or 1)
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "tok": f"{loss_tok.item():.4f}",
                    "sp": f"{loss_space.item():.4f}",
                    "pt": f"{loss_part.item():.4f}",
                    "lr": f"{lr_now:.2e}",
                    "grad": f"{grad_norm:.2f}",
                    "eta": eta,
                }
            )

        train_loss = loss_sum / max(1, len(train_dl))

        # ----- 검증 -----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            pbar_v = tqdm(valid_dl, desc=f"Epoch {ep} [valid]", ncols=120)
            pbar_v_start = time.time()
            for batch in pbar_v:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                y_tok = batch["label_token_ids"].to(device)
                y_space = batch["label_space_ids"].to(device)
                y_part = batch["label_particle_ids"].to(device)

                logits_tok, logits_space, logits_part = model(input_ids, attn)
                loss_tok = criterion_tok(logits_tok.view(-1, logits_tok.size(-1)), y_tok.view(-1))
                loss = loss_tok

                if args.use_space and criterion_space is not None and logits_space is not None:
                    loss_space = criterion_space(
                        logits_space.view(-1, logits_space.size(-1)), y_space.view(-1)
                    )
                    loss = loss + args.lambda_space * loss_space
                else:
                    loss_space = torch.tensor(0.0, device=device)

                if args.use_particle and criterion_part is not None and logits_part is not None:
                    loss_part = criterion_part(
                        logits_part.view(-1, logits_part.size(-1)), y_part.view(-1)
                    )
                    loss = loss + args.lambda_particle * loss_part
                else:
                    loss_part = torch.tensor(0.0, device=device)

                val_loss_sum += loss.item()
                eta_v = _eta_from_pbar(pbar_v_start, pbar_v.n, pbar_v.total or 1)
                pbar_v.set_postfix(
                    {"val_loss": f"{loss.item():.4f}", "tok": f"{loss_tok.item():.4f}", "eta": eta_v}
                )

        val_loss = val_loss_sum / max(1, len(valid_dl))

        (
            acc_tok,
            acc_edit,
            P,
            R,
            F,
            space_acc,
            particle_acc_all,
            particle_acc_is,
        ) = evaluate(
            model,
            valid_dl,
            id2token,
            tok,
            device,
            space2id,
            id2space,
            particle2id,
            id2particle,
            use_space=args.use_space,
            use_particle=args.use_particle,
        )

        msg = (
            f"[Epoch {ep}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"acc_token={acc_tok:.4f} | acc_edit={acc_edit:.4f} | "
            f"P={P:.4f} R={R:.4f} F0.5={F:.4f} | space_acc={space_acc:.4f}"
        )
        if args.use_particle:
            if particle_acc_all is not None:
                msg += f" | particle_acc_all={particle_acc_all:.4f}"
            if particle_acc_is is not None:
                msg += f" | particle_acc_IS={particle_acc_is:.4f}"
        msg += f" | time={time.time()-t0:.1f}s"
        print(msg)

        # 프리뷰
        preview_samples(
            model,
            tok,
            id2token,
            device,
            train_items,
            title="train",
            use_space=args.use_space,
            id2space=id2space,
            max_len=args.max_len,
            k=args.preview_k,
        )
        preview_samples(
            model,
            tok,
            id2token,
            device,
            valid_items,
            title="valid",
            use_space=args.use_space,
            id2space=id2space,
            max_len=args.max_len,
            k=args.preview_k,
        )

        # best 저장 (F0.5 기준)
        if F > best_f:
            best_f = F
            torch.save(
                {
                    "model": model.state_dict(),
                    "token_meta": token_meta,
                    "space_meta": space_meta,
                    "particle_meta": particle_meta,
                    "args": vars(args),
                },
                os.path.join(args.outdir, "best.pt"),
            )
            print(f"  -> 새 best 저장 (F0.5={best_f:.4f})")

    torch.save(
        {
            "model": model.state_dict(),
            "token_meta": token_meta,
            "space_meta": space_meta,
            "particle_meta": particle_meta,
            "args": vars(args),
        },
        os.path.join(args.outdir, "last.pt"),
    )
    print("훈련 종료")


# ==============================
# CLI
# ==============================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="./transformer/out_kobert_space_particle.jsonl")
    p.add_argument("--valid", type=str, default=None)
    p.add_argument("--model", type=str, default="monologg/kobert")
    p.add_argument("--outdir", type=str, default="./runs/kobert_multi")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bsz", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--cpu", action="store_true")

    # 불균형 완화 (token head)
    p.add_argument("--non_keep_weight", type=float, default=3.0)

    # token 라벨 축소 옵션
    p.add_argument("--topn_replace", type=int, default=None)
    p.add_argument("--topn_insert", type=int, default=None)

    # space / particle 사용 여부 및 가중치(보상)
    p.add_argument("--use_space", action="store_true", help="space_labels를 사용해 두 번째 헤드로 학습")
    p.add_argument("--lambda_space", type=float, default=1.0, help="space head loss 가중치")

    p.add_argument("--use_particle", action="store_true", help="particle_labels를 사용해 세 번째 헤드로 학습")
    p.add_argument(
        "--lambda_particle",
        type=float,
        default=1.0,
        help="particle head loss 가중치 (조사 보상)",
    )

    # split & preview
    p.add_argument("--split_seed", type=int, default=13)
    p.add_argument("--preview_k", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
