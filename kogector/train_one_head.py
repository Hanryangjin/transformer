#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_one_head.py  (A안 + 방법1: 복합 태그 단일 의미로 정규화, tqdm 진행바 + ETA/LR/GradNorm)

- 입력: KoBERT 전처리 JSONL/JSON (각 샘플에 src_token, tgt_token, token_labels 포함)
- 라벨: KEEP / DELETE / REPLACE_<tok> / INSERT_<tok> / (선택) REPLACE_UNK / INSERT_UNK
- 복합 태그("KEEP|INSERT_.", "REPLACE_가|INSERT_.", ...)는
  정규화 규칙(우선순위: REPLACE > INSERT(APPEND) > DELETE > KEEP)으로 "하나"의 의미 태그로 축소.

사용 예:
PYTHONPATH=. python3 train_one_head.py \
  --train ./transformer/out_kobert2.jsonl \
  --valid None \
  --model monologg/kobert --outdir ./runs/kobert_a \
  --epochs 10 --bsz 64 --lr 3e-5
"""

import os, json, random, argparse, time, math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

IGNORE_INDEX = -100

# ==============================
# 0) 표시용 normalize (집계/로그용)
#    - 학습/메타/디코딩에서는 절대 사용하지 않음!!
# ==============================
def normalize_token_label_for_display(lb: str) -> str:
    """로그 집계용 축약: REPLACE_*→REPLACE, INSERT_* or APPEND_*→INSERT, 나머지 KEEP/DELETE.
       (언더스코어 누락/앞뒤 공백 등 안전 처리)"""
    if not isinstance(lb, str):
        return "KEEP"
    s = lb.strip()
    if s.startswith("APPEND_"):         # 과거 산출 호환(표시용으로 INSERT로 묶기)
        return "INSERT"
    if s.startswith("INSERT_") or s.startswith("INSERT"):
        return "INSERT"
    if s.startswith("REPLACE_") or s.startswith("REPLACE"):
        return "REPLACE"
    if s in ("KEEP", "DELETE"):
        return s
    if s in ("REPLACE_UNK", "INSERT_UNK"):
        return s.replace("_UNK", "")     # REPLACE, INSERT로 집계
    return "KEEP"


# ==============================
# 1) 데이터 로더/유틸
# ==============================
def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            items = json.load(f)
        else:
            for line in f:
                line=line.strip()
                if line:
                    items.append(json.loads(line))
    return items

def _prefix_fix_raw_label(s: str) -> str:
    """원형 라벨 보존. 전처리 잔재 보정만 수행: APPEND_→INSERT_, REPLACE/INSERT 언더스코어 보정."""
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
    방법1: 복합 태그를 단일 의미 태그로 축약.
    우선순위: REPLACE_* > INSERT_*/APPEND_* > DELETE > KEEP
    """
    if not isinstance(s, str) or "|" not in s:
        return s
    parts = [p.strip() for p in s.split("|") if p.strip()]
    # REPLACE 우선
    for p in parts:
        if p.startswith("REPLACE_") or p.startswith("REPLACE"):
            return _prefix_fix_raw_label(p)
    # INSERT/APPEND 다음
    for p in parts:
        if p.startswith("APPEND_"):
            return _prefix_fix_raw_label(p)  # 위에서 INSERT_로 바뀜
        if p.startswith("INSERT_") or p.startswith("INSERT"):
            return _prefix_fix_raw_label(p)
    # DELETE
    if "DELETE" in parts:
        return "DELETE"
    # KEEP
    if "KEEP" in parts:
        return "KEEP"
    # fallback: 첫 파트 보정
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

def build_label_meta(items: List[Dict[str, Any]],
                     topn_replace: int = None,
                     topn_insert: int = None) -> Dict[str, Any]:
    """원형 라벨로 어휘 구성. (선택) 상위 N만 유지하고 나머지는 *_UNK로 통합."""
    cnt = Counter()
    for ex in items:
        for lb in ex["token_labels"]:
            if lb: cnt[lb] += 1

    # 상위 N 축소(선택)
    if topn_replace is not None:
        keep = set([l for l,_ in cnt.most_common() if l.startswith("REPLACE_")][:topn_replace])
        for l in list(cnt.keys()):
            if l.startswith("REPLACE_") and l not in keep:
                cnt["REPLACE_UNK"] += cnt.pop(l)
    if topn_insert is not None:
        keep = set([l for l,_ in cnt.most_common() if l.startswith("INSERT_")][:topn_insert])
        for l in list(cnt.keys()):
            if l.startswith("INSERT_") and l not in keep:
                cnt["INSERT_UNK"] += cnt.pop(l)

    cnt["KEEP"] += 0
    cnt["DELETE"] += 0

    labels = sorted(cnt.keys())
    token2id = {lb:i for i,lb in enumerate(labels)}
    id2token = {i:lb for lb,i in token2id.items()}
    print(f"[meta] 라벨 크기={len(labels)} (예: 앞 30) {labels[:30]}")
    return {"labels": labels, "token2id": token2id, "id2token": id2token}

class TagDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], tokenizer, token2id: Dict[str,int], max_len: int):
        self.items = items
        self.tokenizer = tokenizer
        self.t2i = token2id
        self.max_len = max_len
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        ex = self.items[idx]
        pieces = ex.get("pieces")
        if pieces is None:
            if isinstance(ex.get("src_token"), list) and ex["src_token"]:
                pieces = ex["src_token"]
            else:
                # 최후의 수단 (가능하면 도달하지 않게 전처리/로드)
                src = ex.get("meta", {}).get("src") or ex.get("src") or ""
                pieces = self.tokenizer.tokenize(src)
        labels = ex["token_labels"]
        # 접두 잔재 보정 + 복합 태그 단일 의미로 축약 (방법1)
        labels = [_collapse_composite_label(_prefix_fix_raw_label(lb)) for lb in labels]

        if len(labels) != len(pieces):
            raise ValueError(f"길이 불일치: labels={len(labels)} pieces={len(pieces)}")

        # convert tokens -> ids (재분절 없이!)
        input_ids = self.tokenizer.convert_tokens_to_ids(pieces)
        # truncation
        input_ids = input_ids[:self.max_len]
        labels = labels[:self.max_len]

        label_ids = [self.t2i.get(lb, self.t2i.get("KEEP")) for lb in labels]
        attn_mask = [1]*len(input_ids)

        # meta 보존(프리뷰용)
        meta = ex.get("meta", {})
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "pieces": pieces[:self.max_len],
            "token_labels": labels,
            "meta": meta
        }

def collate_batch(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad(t, val):
        pad_len = max_len - len(t)
        if pad_len>0:
            t = torch.cat([t, torch.full((pad_len,), val, dtype=t.dtype)], dim=0)
        return t
    input_ids = torch.stack([pad(b["input_ids"], pad_id) for b in batch])
    attention_mask = torch.stack([pad(b["attention_mask"], 0) for b in batch])
    label_ids = torch.stack([pad(b["label_ids"], IGNORE_INDEX) for b in batch])
    pieces = [b["pieces"] + ["[PAD]"]*(max_len-len(b["pieces"])) for b in batch]
    token_labels = [b["token_labels"] + ["KEEP"]*(max_len-len(b["token_labels"])) for b in batch]
    metas = [b.get("meta", {}) for b in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "label_ids": label_ids, "pieces": pieces, "token_labels": token_labels,
            "metas": metas}

# ==============================
# 2) 모델 (한 헤드 분류)
# ==============================
class Tagger(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        seq = self.dropout(seq)
        logits = self.classifier(seq)
        return logits

# ==============================
# 3) 디코딩(태그 문자열 직접 적용) & 평가/프리뷰
# ==============================
def _strip_leading(tok: str) -> Tuple[str, bool]:
    """SentencePiece/WordPiece 경계 전처리: ('▁', '##') 처리 -> (표면형, 앞에 공백 여부)"""
    if tok.startswith("▁"):
        return tok[1:], True
    if tok.startswith("##"):
        return tok[2:], False
    return tok, False

def decode_apply_token_actions(pieces: List[str], actions: List[str]) -> str:
    """MLM 없이 태그 문자열을 직접 적용하여 표면 문자열 복원 (복합 태그는 로더에서 이미 단일화됨)"""
    out: List[Tuple[str,bool]] = []
    for i, p in enumerate(pieces):
        act = actions[i] if i < len(actions) else "KEEP"
        surf, sp = _strip_leading(p)
        if act == "KEEP":
            out.append((surf, sp))
        elif act == "DELETE":
            continue
        elif act.startswith("REPLACE_"):
            tgt = act[len("REPLACE_"):]
            ts, _tsp = _strip_leading(tgt)
            out.append((ts, sp))
        elif act.startswith("INSERT_"):
            # 현재 토큰 유지 + 뒤에 삽입
            out.append((surf, sp))
            tgt = act[len("INSERT_"):]
            ts, _tsp = _strip_leading(tgt)
            out.append((ts, True))
        elif act in ("REPLACE_UNK", "INSERT_UNK"):
            # 보수적 정책
            out.append((surf, sp))
        else:
            out.append((surf, sp))
    # 표면 조립
    s=[]
    for k,(ts,sp) in enumerate(out):
        if k==0: s.append(ts)
        else: s.append((" " if sp else "") + ts)
    return "".join(s)

def edits_from_actions(pieces: List[str], actions: List[str]) -> List[Tuple[int,str,str]]:
    """간단 편집 추출: (index, OP, arg) 리스트 (평가용)"""
    E=[]
    for i, act in enumerate(actions):
        if act.startswith("REPLACE_"):
            E.append((i, "R", act[len("REPLACE_"):]))
        elif act.startswith("INSERT_"):
            E.append((i, "I", act[len("INSERT_"):]))
        elif act=="DELETE":
            E.append((i, "D", ""))
    return E

def prf_from_editsets(pred_edits, gold_edits, beta: float = 0.5):
    """편집 집합 교집합 기반 P/R/Fβ (간이)."""
    pred_set = set(pred_edits)
    gold_set = set(gold_edits)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    P = tp / (tp+fp+1e-8)
    R = tp / (tp+fn+1e-8)
    b2 = beta*beta
    F = (1+b2)*P*R / (b2*P + R + 1e-8)
    return P, R, F

def evaluate(model, loader, id2tok, tokenizer, device):
    model.eval()
    n_tok, n_tok_correct = 0, 0
    n_edit, n_edit_correct = 0, 0

    P_all=R_all=F_all=0.0; n_sent=0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label_ids"].to(device)

            logits = model(input_ids, attn)
            pred_ids = torch.argmax(logits, dim=-1)

            # token acc & edit-token acc
            mask = labels.ne(IGNORE_INDEX) & attn.ne(0)
            n_tok += mask.sum().item()
            n_tok_correct += (pred_ids.eq(labels) & mask).sum().item()

            # edit-only acc
            keep_id = id2tok_inv.get("KEEP", None)
            if keep_id is not None:
                edit_mask = mask & (~(labels == keep_id))
            else:
                edit_mask = mask
            n_edit += edit_mask.sum().item()
            n_edit_correct += ((pred_ids.eq(labels)) & edit_mask).sum().item()

            # sentence-level P/R/F0.5
            for b in range(input_ids.size(0)):
                L = mask[b].sum().item()
                if L == 0:
                    continue
                pieces = batch["pieces"][b][:L]
                gold_actions = [id2tok[labels[b,i].item()] for i in range(L)]
                pred_actions = [id2tok[pred_ids[b,i].item()] for i in range(L)]

                pred_ed = edits_from_actions(pieces, pred_actions)
                gold_ed = edits_from_actions(pieces, gold_actions)

                P,R,F = prf_from_editsets(pred_ed, gold_ed, beta=0.5)
                P_all += P; R_all += R; F_all += F; n_sent += 1

    acc_tok = n_tok_correct / max(1,n_tok)
    acc_edit = n_edit_correct / max(1,n_edit)
    P = P_all / max(1,n_sent); R = R_all / max(1,n_sent); F = F_all / max(1,n_sent)
    return acc_tok, acc_edit, P, R, F

def _reconstruct_src_from_pieces(pieces: List[str]) -> str:
    s=[]
    for i, p in enumerate(pieces):
        tok, sp = _strip_leading(p)
        if i==0: s.append(tok)
        else: s.append((" " if sp else "") + tok)
    return "".join(s)

def preview_samples(model, tokenizer, id2tok, device,
                    items: List[Dict[str, Any]], title: str, k: int = 5, max_len: int = 256):
    """무작위 k개 샘플 프린트: 입력/예측/정답"""
    print("\n=== 예시 문장 (무작위 {}개) ===".format(k))
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

            # 입력/정답 텍스트
            inp_text = meta.get("src") or _reconstruct_src_from_pieces(pieces)
            gold_actions = labels
            gold_text = meta.get("tgt") or decode_apply_token_actions(pieces, gold_actions)

            # 모델 추론
            input_ids = tokenizer.convert_tokens_to_ids(pieces)
            attn = [1]*len(input_ids)
            input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
            attn_t = torch.tensor([attn], dtype=torch.long, device=device)
            logits = model(input_t, attn_t)  # [1, L, C]
            pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
            pred_actions = [id2tok[i] for i in pred_ids[:len(pieces)]]
            pred_text = decode_apply_token_actions(pieces, pred_actions)

            print(f"샘플 {idx}:")
            print(f"> 입력: {inp_text}")
            print(f"> 예측: {pred_text}")
            print(f"> 정답: {gold_text}\n")

# ==============================
# 4) 학습 루프 (+ tqdm with ETA/LR/GradNorm)
# ==============================
def _eta_from_pbar(pbar_start_time: float, n_done: int, total: int) -> str:
    """경과시간 기반 ETA 문자열 생성 (HH:MM:SS)"""
    if n_done <= 0:
        return "--:--"
    elapsed = time.time() - pbar_start_time
    rate = elapsed / max(1, n_done)
    remain = rate * max(0, total - n_done)
    # 형식화
    h = int(remain // 3600); m = int((remain % 3600) // 60); s = int(remain % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # tokenizer (convert_tokens_to_ids만 사용; 재토크나이즈 금지)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)

    # 데이터 로드
    train_path = args.train
    if train_path is None or not os.path.exists(train_path):
        raise FileNotFoundError(f"--train 경로가 없습니다: {train_path}")
    train_items_all = read_json_or_jsonl(train_path)

    valid_items: Optional[List[Dict[str, Any]]] = None
    if args.valid is not None and str(args.valid).lower() != "none":
        if not os.path.exists(args.valid):
            raise FileNotFoundError(f"--valid 경로가 없습니다: {args.valid}")
        valid_items = read_json_or_jsonl(args.valid)

    # src_token 우선 사용, 길이 불일치 시 사전 스킵
    def _pre_screen(items):
        good=[]
        skipped=0
        for rec in items:
            pieces = rec.get("pieces")
            if pieces is None:
                if isinstance(rec.get("src_token"), list) and rec["src_token"]:
                    pieces = rec["src_token"]
                else:
                    src = rec.get("meta",{}).get("src") or rec.get("src") or ""
                    if not src:
                        skipped+=1; continue
                    pieces = tok.tokenize(src)
            labels = rec.get("token_labels") or []
            # 접두 보정 + 복합 태그 단일 의미 축약
            labels = [_collapse_composite_label(_prefix_fix_raw_label(x)) for x in labels]
            if len(labels) != len(pieces):
                skipped += 1; continue
            rec["pieces"] = pieces
            rec["token_labels"] = labels
            good.append(rec)
        if skipped:
            print(f"[pre] 스킵 {skipped}개 (길이 불일치/결측)")
        return good

    train_items_all = _pre_screen(train_items_all)

    # valid가 None이면 train의 10%를 valid로 분리
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

    # label meta (원형)
    meta = build_label_meta(train_items, topn_replace=args.topn_replace, topn_insert=args.topn_insert)
    token2id = meta["token2id"]; id2token = meta["id2token"]
    # eval/preview에서 'KEEP' id 필요
    global id2tok, id2tok_inv
    id2tok = id2token
    id2tok_inv = {v:k for k,v in id2tok.items()}

    train_ds = TagDataset(train_items, tok, token2id, args.max_len)
    valid_ds = TagDataset(valid_items, tok, token2id, args.max_len)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    train_dl = DataLoader(train_ds, batch_size=args.bsz, shuffle=True,
                          collate_fn=lambda b: collate_batch(b, pad_id))
    valid_dl = DataLoader(valid_ds, batch_size=args.bsz, shuffle=False,
                          collate_fn=lambda b: collate_batch(b, pad_id))

    model = Tagger(args.model, num_labels=len(meta["labels"])).to(device)

    # loss (클래스 불균형 완화 옵션)
    weight = None
    if args.non_keep_weight > 0:
        # KEEP 과다 시, 비-KEEP 가중치 부여
        w = torch.ones(len(meta["labels"]))
        keep_id = token2id.get("KEEP", None)
        if keep_id is not None:
            w[keep_id] = 1.0
            non_keep_ids = [i for i in range(len(meta["labels"])) if i!=keep_id]
            w[non_keep_ids] = args.non_keep_weight
        weight = w.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl)*args.epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.05*total_steps),
                                            num_training_steps=total_steps)

    best_f = -1.0
    for ep in range(1, args.epochs+1):
        model.train()
        loss_sum = 0.0
        t0 = time.time()

        # === tqdm: 학습 진행바 ===
        pbar = tqdm(train_dl, desc=f"Epoch {ep} [train]", ncols=120)
        pbar_start = time.time()
        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label_ids"].to(device)

            logits = model(input_ids, attn)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            opt.zero_grad()
            loss.backward()
            # clip & grad norm
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step(); sched.step()

            loss_sum += loss.item()

            # postfix: loss, ETA, lr, grad_norm
            lr_now = opt.param_groups[0]["lr"]
            eta = _eta_from_pbar(pbar_start, pbar.n, pbar.total or 1)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "eta": eta,
                "lr": f"{lr_now:.2e}",
                "grad": f"{grad_norm:.2f}",
            })

        train_loss = loss_sum / max(1,len(train_dl))

        # === tqdm: 검증 진행바 ===
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            pbar_v = tqdm(valid_dl, desc=f"Epoch {ep} [valid]", ncols=120)
            pbar_v_start = time.time()
            for batch in pbar_v:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["label_ids"].to(device)
                logits = model(input_ids, attn)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss_sum += loss.item()

                eta_v = _eta_from_pbar(pbar_v_start, pbar_v.n, pbar_v.total or 1)
                pbar_v.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "eta": eta_v
                })
        val_loss = val_loss_sum / max(1,len(valid_dl))

        acc_tok, acc_edit, P, R, F = evaluate(model, valid_dl, id2token, tok, device)

        print(f"[Epoch {ep}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"acc_token={acc_tok:.4f} | acc_edit={acc_edit:.4f} | P={P:.4f} R={R:.4f} F0.5={F:.4f} "
              f"| time={time.time()-t0:.1f}s")

        # === 에폭별 무작위 프리뷰 (train/valid 각각 k개) ===
        preview_samples(model, tok, id2token, device, train_items, title="train", k=args.preview_k, max_len=args.max_len)
        preview_samples(model, tok, id2token, device, valid_items, title="valid", k=args.preview_k, max_len=args.max_len)

        # save best
        if F > best_f:
            best_f = F
            torch.save({
                "model": model.state_dict(),
                "meta": meta,
                "args": vars(args)
            }, os.path.join(args.outdir, "best.pt"))
            print(f"  -> 새 best 저장 (F0.5={best_f:.4f})")

    # 최종 저장
    torch.save({
        "model": model.state_dict(),
        "meta": meta,
        "args": vars(args)
    }, os.path.join(args.outdir, "last.pt"))
    print("훈련 종료")

# ==============================
# 5) CLI
# ==============================
def get_args():
    p = argparse.ArgumentParser()
    # 기본 train 경로 & valid=None 자동 분리
    p.add_argument("--train", type=str, default="./transformer/out_kobert2.jsonl",
                   help="학습 JSONL/JSON 경로. 기본: ./transformer/out_kobert2.jsonl")
    p.add_argument("--valid", type=str, default=None,
                   help="검증 JSONL/JSON 경로. None이면 train의 1/10을 validation으로 분리")
    p.add_argument("--model", type=str, default="monologg/kobert")
    p.add_argument("--outdir", type=str, default="./runs/kobert_a")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bsz", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--cpu", action="store_true")
    # 불균형 대응
    p.add_argument("--non_keep_weight", type=float, default=3.0,
                   help="비-KEEP 클래스 가중치(KEEP=1.0, 비-KEEP=이 값)")
    # 라벨 어휘 축소 옵션(원형 태그 기반 Top-N 유지)
    p.add_argument("--topn_replace", type=int, default=None)
    p.add_argument("--topn_insert", type=int, default=None)
    # valid None 분할용 시드 & 프리뷰 샘플 수
    p.add_argument("--split_seed", type=int, default=13)
    p.add_argument("--preview_k", type=int, default=5,
                   help="에폭별 train/valid 프리뷰 샘플 수")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
