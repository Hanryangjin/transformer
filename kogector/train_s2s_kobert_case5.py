#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CASE 5: KoBERT 기반 Seq2Seq GEC 파인튜닝 코드

- Encoder: get_kobert_model() (BERT encoder)
- Decoder: nn.TransformerDecoder + LM head (vocab 공유)
- 데이터: CASE 1~4에서 사용한 JSON/JSONL (meta.src, meta.tgt 기준)
- 출력:
    - Epoch마다 콘솔에:
        train_loss | val_loss | acc_token | acc_edit | P/R/F0.5 | space_acc | time_sec
    - outdir/metrics.csv 에 동일 정보 기록
    - validation 예시 몇 개: 입력 / 예측 / 정답
"""

import os
import sys
import time
import json
import math
import csv
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from kobert_transformers import get_kobert_model, get_tokenizer
from transformers import EncoderDecoderModel

# ---------------------------------------
# 유틸 함수들
# ---------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gec_items(path: str) -> List[Dict]:
    """
    CASE 1~4에서 쓰던 포맷을 가정:
    - JSONL:
        {"meta": {"src": "...", "tgt": "..."}, ...}
    - 또는 JSON 배열:
        [{"meta": {...}}, {...}, ...]

    필요하다면 "src", "tgt" 바로 있는 형태도 허용.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    items: List[Dict] = []
    if text[0] == "[":
        # JSON array
        data = json.loads(text)
        items.extend(data)
    else:
        # JSONL
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def extract_src_tgt(item: Dict) -> Tuple[str, str]:
    """한 레코드에서 src, tgt 문자열 추출"""
    if "meta" in item and isinstance(item["meta"], dict):
        src = item["meta"].get("src", "")
        tgt = item["meta"].get("tgt", "")
    else:
        src = item.get("src", "")
        tgt = item.get("tgt", "")
    if not isinstance(src, str):
        src = str(src)
    if not isinstance(tgt, str):
        tgt = str(tgt)
    return src, tgt


def safe_decode(tokenizer, ids: List[int], pad_id: int) -> str:
    if torch.is_tensor(ids):
        ids = ids.tolist()
    # PAD 제거
    ids = [i for i in ids if i != pad_id]
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception as e:
        return f"<decode_error: {e}>"


def eos_to_pad(seq: List[int], eos_id: int, pad_id: int) -> List[int]:
    """EOS 이후는 PAD로 변환 (평가 시 길이 정리용)"""
    if torch.is_tensor(seq):
        seq = seq.tolist()
    if eos_id in seq:
        idx = seq.index(eos_id)
        return seq[: idx + 1] + [pad_id] * (len(seq) - idx - 1)
    return seq


def space_accuracy(gold: str, pred: str) -> float:
    """
    문자열 단위 띄어쓰기 정확도(근사):
    - 각 문자 위치별로 "공백인지 여부"를 비교하여 정확도 계산.
    - 길이 차이는 padding으로 보정.
    """
    if not gold and not pred:
        return 1.0
    total_len = max(len(gold), len(pred), 1)
    correct = 0
    for i in range(total_len):
        g = (i < len(gold) and gold[i].isspace())
        p = (i < len(pred) and pred[i].isspace())
        if g == p:
            correct += 1
    return correct / total_len


# ---------------------------------------
# Dataset
# ---------------------------------------

class Seq2SeqGECDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, max_len: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        src_text, tgt_text = extract_src_tgt(item)

        # KoBERT 토크나이저 사용 (BERT 스타일)
        # [CLS] ... [SEP] + PAD
        enc = self.tokenizer(
            src_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            tgt_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        src_input_ids = enc["input_ids"].squeeze(0)          # (L,)
        src_attention_mask = enc["attention_mask"].squeeze(0)  # (L,)
        src_token_type_ids = enc.get("token_type_ids", torch.zeros_like(src_input_ids)).squeeze(0)

        tgt_input_ids = dec["input_ids"].squeeze(0)          # (L,)
        tgt_attention_mask = dec["attention_mask"].squeeze(0)  # (L,)

        return {
            "src_input_ids": src_input_ids,
            "src_attention_mask": src_attention_mask,
            "src_token_type_ids": src_token_type_ids,
            "tgt_input_ids": tgt_input_ids,
            "tgt_attention_mask": tgt_attention_mask,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


# ---------------------------------------
# KoBERT Seq2Seq 모델 정의
# ---------------------------------------

class KoBERTSeq2Seq(nn.Module):
    """
    기존 커스텀 TransformerDecoder 대신
    HuggingFace EncoderDecoderModel(monologg/kobert ↔ monologg/kobert)을 감싸는 래퍼.

    - forward(
        src_input_ids, src_attention_mask, src_token_type_ids, decoder_input_ids
      ) -> (B, T, V) logits
    - generate(
        src_input_ids, src_attention_mask, src_token_type_ids, bos_id, eos_id, max_len
      ) -> (B, T_gen) ids
    """

    def __init__(
        self,
        max_len: int,
        pretrained_name: str = "monologg/kobert",
        pad_token_id: int = 1,
        bos_token_id: int = None,
        eos_token_id: int = None,
    ):
        super().__init__()
        self.max_len = max_len

        # KoBERT encoder/decoder로 구성된 EncoderDecoderModel
        self.encdec = EncoderDecoderModel.from_encoder_decoder_pretrained(
            pretrained_name, pretrained_name
        )

        config = self.encdec.config

        # pad / bos / eos 세팅
        if config.pad_token_id is None:
            config.pad_token_id = pad_token_id
        self.pad_token_id = config.pad_token_id

        if bos_token_id is not None:
            config.decoder_start_token_id = bos_token_id
        if eos_token_id is not None:
            config.eos_token_id = eos_token_id

        # vocab / hidden_size 등은 encoder 설정을 따라감
        if hasattr(self.encdec.encoder, "config"):
            self.vocab_size = self.encdec.encoder.config.vocab_size
        else:
            self.vocab_size = config.vocab_size

    def forward(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        src_token_type_ids: torch.Tensor = None,  # 들어오긴 하지만 사용하진 않음
        decoder_input_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        run_epoch()에서 사용하는 인터페이스와 동일.
        teacher forcing 시 decoder_input_ids를 넘겨주면 됨.
        """
        outputs = self.encdec(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        # (B, T, V)
        return outputs.logits

    @torch.no_grad()
    def generate(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        src_token_type_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = None,
    ) -> torch.Tensor:
        """
        run_epoch()에서 사용하는 generate() 인터페이스와 동일.
        실제 inference 시에는 여기만 호출됨.
        """
        if max_len is None:
            max_len = self.max_len

        gen_ids = self.encdec.generate(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            max_length=max_len,
            num_beams=1,
            do_sample=False,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=self.pad_token_id,
        )
        return gen_ids

# ---------------------------------------
# 한 epoch 학습/평가
# ---------------------------------------

def run_epoch(
    model: KoBERTSeq2Seq,
    loader: DataLoader,
    tokenizer,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    optimizer=None,
    max_batches: int = None,
    print_examples: bool = False,
    phase: str = "train",
):
    """
    - optimizer가 None이면 eval 모드
    - train: teacher forcing 기반 loss (지표는 선택)
    - val : teacher forcing 기반 loss + generate() 기반 지표 계산
    """
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_loss = 0.0
    total_batches = 0

    correct_tokens = 0
    total_tokens = 0

    total_gold_edits = 0
    total_pred_edits = 0
    correct_edits = 0

    space_correct = 0.0
    space_total = 0.0

    example_printed = 0
    MAX_EXAMPLES = 5

    pbar = tqdm(loader, desc=f"{phase.capitalize()} epoch", ncols=120)
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        src_input_ids = batch["src_input_ids"].to(device)          # (B, L)
        src_attention_mask = batch["src_attention_mask"].to(device)
        src_token_type_ids = batch["src_token_type_ids"].to(device)
        tgt_input_ids = batch["tgt_input_ids"].to(device)          # (B, L)
        tgt_attention_mask = batch["tgt_attention_mask"].to(device)

        B, L = tgt_input_ids.size()

        # 디코더 입력 / 타깃 (teacher forcing용)
        decoder_input_ids = tgt_input_ids[:, :-1]   # (B, L-1)
        target_ids = tgt_input_ids[:, 1:]           # (B, L-1)

        if is_train:
            optimizer.zero_grad()

        # --------- loss (train/val 공통, teacher forcing) ----------
        with torch.set_grad_enabled(is_train):
            logits = model(
                src_input_ids=src_input_ids,
                src_attention_mask=src_attention_mask,
                src_token_type_ids=src_token_type_ids,
                decoder_input_ids=decoder_input_ids,
            )  # (B, L-1, V)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        # -----------------------------
        # metrics 계산
        # -----------------------------
        with torch.no_grad():
            if is_train:
                # 학습 단계에서는 빠른 teacher forcing 기반 지표만 쓰거나,
                # 아예 지표를 계산하지 않아도 된다.
                # 여기서는 간단히 teacher forcing 기반 토큰/편집/space_acc를 계산.
                pred_ids = torch.argmax(logits, dim=-1)  # (B, L-1)
                pred_full = torch.cat([decoder_input_ids[:, :1], pred_ids], dim=1)  # (B, L)
            else:
                # validation 단계에서는 실제 inference(greedy) 기준으로 지표 계산
                gen = model.generate(
                    src_input_ids=src_input_ids,
                    src_attention_mask=src_attention_mask,
                    src_token_type_ids=src_token_type_ids,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    max_len=L,  # max_len과 동일하게
                )  # (B, T_gen)

                # 길이를 L로 맞추기 위해 pad 확장
                pred_full = torch.full((B, L), pad_id, dtype=torch.long, device=device)
                for b in range(B):
                    seq = gen[b]
                    # EOS 이후는 PAD로, 그리고 L 길이까지만 복사
                    seq = eos_to_pad(seq, eos_id, pad_id)
                    seq = seq[:L] + [pad_id] * max(0, L - len(seq))
                    pred_full[b] = torch.tensor(seq, dtype=torch.long, device=device)

            gold_full = tgt_input_ids
            src_full = src_input_ids

            for b in range(B):
                # EOS 이후 PAD로 정리
                pred_seq = eos_to_pad(pred_full[b], eos_id, pad_id)
                gold_seq = eos_to_pad(gold_full[b], eos_id, pad_id)
                src_seq = eos_to_pad(src_full[b], eos_id, pad_id)

                # 길이 차이 보정 (최소 길이 기준)
                max_len = min(len(pred_seq), len(gold_seq), len(src_seq))
                pred_seq = pred_seq[:max_len]
                gold_seq = gold_seq[:max_len]
                src_seq = src_seq[:max_len]

                # 토큰 정확도 (PAD, 첫 토큰 제외)
                for i in range(max_len):
                    if gold_seq[i] == pad_id:
                        continue
                    if i == 0:  # BOS/CLS 위치는 스킵
                        continue
                    total_tokens += 1
                    if pred_seq[i] == gold_seq[i]:
                        correct_tokens += 1

                # 편집 정확도
                for i in range(max_len):
                    if gold_seq[i] == pad_id or src_seq[i] == pad_id:
                        continue
                    gold_edit = (gold_seq[i] != src_seq[i])
                    pred_edit = (pred_seq[i] != src_seq[i] and pred_seq[i] != pad_id)
                    if gold_edit:
                        total_gold_edits += 1
                    if pred_edit:
                        total_pred_edits += 1
                    if gold_edit and pred_seq[i] == gold_seq[i]:
                        correct_edits += 1

                # space_acc (문자열 기준)
                gold_text = safe_decode(tokenizer, gold_seq, pad_id)
                pred_text = safe_decode(tokenizer, pred_seq, pad_id)
                sa = space_accuracy(gold_text, pred_text)
                space_correct += sa
                space_total += 1.0

                # 예시 출력 (validation에서만, 소수)
                if (not is_train) and print_examples and example_printed < MAX_EXAMPLES:
                    src_text = batch["src_text"][b]
                    print("\n[샘플 예시]")
                    print(f"> 입력:  {src_text}")
                    print(f"> 예측:  {pred_text}")
                    print(f"> 정답:  {gold_text}")
                    example_printed += 1

        # tqdm 표시
        avg_loss = total_loss / max(1, total_batches)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # 집계
    avg_loss = total_loss / max(1, total_batches)
    acc_token = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    if total_gold_edits > 0:
        acc_edit = correct_edits / total_gold_edits
    else:
        acc_edit = 0.0

    precision = correct_edits / total_pred_edits if total_pred_edits > 0 else 0.0
    recall = correct_edits / total_gold_edits if total_gold_edits > 0 else 0.0
    beta = 0.5
    if precision + recall > 0:
        f0_5 = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    else:
        f0_5 = 0.0

    space_acc = space_correct / max(1.0, space_total)

    metrics = {
        "loss": avg_loss,         # train: train_loss, val: val_loss (둘 다 teacher forcing 기반)
        "acc_token": acc_token,   # train: teacher forcing 기반, val: generate 기반
        "acc_edit": acc_edit,
        "P": precision,
        "R": recall,
        "F0.5": f0_5,
        "space_acc": space_acc,
    }
    return metrics


# ---------------------------------------
# 메인 학습 루프
# ---------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="train json/jsonl 경로")
    parser.add_argument("--valid", type=str, default="None", help="valid json/jsonl 경로 (None이면 train 1/10 사용)")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="강제로 CPU 사용")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device = {device}")

    # 토크나이저 & special token id
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 1  # 안전장치
    bos_id = tokenizer.cls_token_id  # [CLS]
    eos_id = tokenizer.sep_token_id  # [SEP]

    # ------------------------
    # 데이터 로딩 & split
    # ------------------------
    train_items_all = load_gec_items(args.train)

    if args.valid is None or args.valid == "None":
        idxs = list(range(len(train_items_all)))
        random.shuffle(idxs)
        cut = int(len(idxs) * 0.1)
        valid_idxs = idxs[:cut]
        train_idxs = idxs[cut:]
        train_items = [train_items_all[i] for i in train_idxs]
        valid_items = [train_items_all[i] for i in valid_idxs]
        print(f"[INFO] train split: {len(train_items)} / valid split: {len(valid_items)} (from train)")
    else:
        valid_items = load_gec_items(args.valid)
        train_items = train_items_all
        print(f"[INFO] train = {len(train_items)}, valid = {len(valid_items)}")

    train_dataset = Seq2SeqGECDataset(train_items, tokenizer, args.max_len)
    valid_dataset = Seq2SeqGECDataset(valid_items, tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ------------------------
    # 모델 & 옵티마이저
    # ------------------------
    model = KoBERTSeq2Seq(
        max_len=args.max_len,
        pretrained_name="monologg/kobert",
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics_path = os.path.join(args.outdir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)

    best_f0_5 = -1.0
    best_path = os.path.join(args.outdir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

        epoch_start = time.time()

        # -----------------
        # 1) Train
        # -----------------
        train_start = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            tokenizer,
            pad_id,
            bos_id,
            eos_id,
            device,
            optimizer=optimizer,
            phase="train",
        )
        train_time = time.time() - train_start
        train_loss = train_metrics["loss"]

        # -----------------
        # 2) Validation (generate 기반 지표 포함)
        # -----------------
        val_start = time.time()
        val_metrics = run_epoch(
            model,
            valid_loader,
            tokenizer,
            pad_id,
            bos_id,
            eos_id,
            device,
            optimizer=None,
            phase="val",
            print_examples=True,
        )
        val_time = time.time() - val_start

        val_loss = val_metrics["loss"]
        acc_token = val_metrics["acc_token"]
        acc_edit = val_metrics["acc_edit"]
        P = val_metrics["P"]
        R = val_metrics["R"]
        F05 = val_metrics["F0.5"]
        space_acc = val_metrics["space_acc"]

        epoch_time = time.time() - epoch_start

        log_msg = (
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"acc_token={acc_token:.4f} | acc_edit={acc_edit:.4f} | "
            f"P={P:.4f} R={R:.4f} F0.5={F05:.4f} | "
            f"space_acc={space_acc:.4f} | "
            f"train_time={train_time:.1f}s | val_time={val_time:.1f}s | total={epoch_time:.1f}s"
        )
        print("\n" + log_msg)

        # -------------------
        # CSV 기록
        # -------------------
        try:
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(
                        [
                            "epoch",
                            "train_loss",
                            "val_loss",
                            "acc_token",
                            "acc_edit",
                            "P",
                            "R",
                            "F0.5",
                            "space_acc",
                            "train_time_sec",
                            "val_time_sec",
                            "time_sec",
                        ]
                    )
                    write_header = False

                writer.writerow(
                    [
                        epoch,
                        float(train_loss),
                        float(val_loss),
                        float(acc_token),
                        float(acc_edit),
                        float(P),
                        float(R),
                        float(F05),
                        float(space_acc),
                        float(train_time),
                        float(val_time),
                        float(epoch_time),
                    ]
                )
        except PermissionError:
            print("[경고] metrics.csv 쓰기 실패 (다른 프로그램에서 열려 있을 수 있음). 이 epoch의 기록은 CSV에 저장되지 않습니다.")

        # -------------------
        # best 모델 저장 (F0.5 기준)
        # -------------------
        if F05 > best_f0_5:
            best_f0_5 = F05
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "F0.5": F05,
                },
                best_path,
            )
            print(f"[INFO] 새로운 best 모델 저장: {best_path} (F0.5={F05:.4f})")


if __name__ == "__main__":
    main()
