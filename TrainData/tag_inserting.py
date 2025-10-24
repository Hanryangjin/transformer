#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_data_kobert_only.py
------------------------------
GECToR 전처리 (KoBERT 전용, tgt-토큰 단위 REPLACE)

입력:
  --json_input : {"data":[{"err_sentence":..., "cor_sentence":...}, ...]} 형식의 단일 JSON 파일
                 (최상위가 배열(list)인 경우도 지원)
출력:
  JSONL (레코드: {"src_token": [...], "tgt_token": [...], "token_labels": [...], "meta": {...}})

토크나이저:
  - KoBERT 전용 (기본: 'monologg/kobert')
  - 우선 transformers.AutoTokenizer(trust_remote_code=True) 시도
  - (선택) tokenization_kobert가 설치되어 있으면 우선 사용할 수도 있음(아래 코멘트 참고)

태깅(한 소스 토큰당 1개):
  - KEEP, DELETE, REPLACE_<tgt토큰>, INSERT_<tgt토큰>
  - 치환 구간에서 tgt 토큰 단위로 REPLACE를 부여:
    예) src ["시간","이","부","족"], tgt ["▁시간이","▁부족"]
        → ["REPLACE_▁시간이","DELETE","REPLACE_▁부족","DELETE"]
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

OUTPUT_PATH = "./transformer/out_kobert2.jsonl"
INPUT_PATH = "./transformer/TrainData/combined_train_dataset.json"

# ------------------------------
# KoBERT tokenizer
# ------------------------------

def load_kobert(model_name: str = "monologg/kobert"):
    """
    KoBERT 토크나이저 로더.
    - 보안/호환성 상 이유로 Hugging Face의 원격 커스텀 코드를 실행 허용해야 할 수 있음.
      (monologg/kobert는 커스텀 코드 포함)
    """
    # 1) transformers 경로 (권장: trust_remote_code=True)
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,   # ★ 커스텀 코드 허용
        )
    except Exception:
        # 2) tokenization_kobert (환경에 따라 동작)
        try:
            from tokenization_kobert import KoBertTokenizer
            return KoBertTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                "KoBERT 토크나이저 로드 실패: transformers 또는 tokenization_kobert 설치/환경을 확인하세요."
            ) from e

def tokenize_kobert(kobert_tok, s: str) -> List[str]:
    """
    KoBERT/WordPiece 계열 .tokenize() 사용
    """
    return kobert_tok.tokenize(s)

# ------------------------------
# LCS alignment
# ------------------------------

def lcs_table(a: List[str], b: List[str]):
    """
    dp[i][j] = a[i:], b[j:]의 LCS 길이
    """
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n-1, -1, -1):
        ai = a[i]
        row_i1 = dp[i+1]
        row_i = dp[i]
        for j in range(m-1, -1, -1):
            if ai == b[j]:
                row_i[j] = 1 + row_i1[j+1]
            else:
                # max(dp[i+1][j], dp[i][j+1])
                v1 = row_i1[j]
                v2 = row_i[j+1]
                row_i[j] = v1 if v1 >= v2 else v2
    return dp

def align_tokens(src: List[str], tgt: List[str]) -> List[Tuple[int, int]]:
    """
    LCS 테이블을 역추적하여 동일 토큰 매치의 (src_i, tgt_j) 쌍 리스트를 반환
    """
    dp = lcs_table(src, tgt)
    i = j = 0
    matches = []
    n, m = len(src), len(tgt)
    while i < n and j < m:
        if src[i] == tgt[j]:
            matches.append((i, j))
            i += 1
            j += 1
        else:
            # dp[i+1][j] vs dp[i][j+1]
            down = dp[i+1][j] if i+1 <= n else -1
            right = dp[i][j+1] if j+1 <= m else -1
            if down >= right:
                i += 1
            else:
                j += 1
    return matches

# ------------------------------
# REPLACE per tgt-token helper
# ------------------------------

def _norm(token: str) -> str:
    """
    KoBERT/WordPiece 접두(▁, ##) 제거하여 비교용 표면형으로 사용
    """
    t = token
    if t.startswith("▁"):
        t = t[1:]
    if t.startswith("##"):
        t = t[2:]
    return t

def _assign_replace_by_tgt_units(src, tgt, tags, src_seg_idx, tgt_seg_idx):
    """
    치환 구간에서 'tgt 토큰' 단위로 REPLACE를 부여.
    - 각 tgt 토큰을 만들기 위해 src에서 좌→우로 최소 1개 이상 소비
    - 소비한 src 묶음의 첫 토큰에 REPLACE_<그 tgt>, 나머지는 DELETE
    - src가 남으면 DELETE, tgt가 남으면 마지막 REPLACE 소스에 INSERT 누적
    """
    last_replace_src = None

    i = 0  # src_seg_idx 내부 포인터
    j = 0  # tgt_seg_idx 내부 포인터
    while i < len(src_seg_idx) and j < len(tgt_seg_idx):
        tgt_tok = tgt[tgt_seg_idx[j]]
        tgt_norm = _norm(tgt_tok)

        consumed = []
        accum = ""

        # 최소 1개는 소비하며 tgt_norm과 길이를 맞춰감
        while i < len(src_seg_idx):
            si = src_seg_idx[i]
            consumed.append(si)
            accum += _norm(src[si])
            i += 1
            # 정지 조건: 정확히 같아졌거나 / 길이가 같거나 넘었거나 / 더 이상 src 없음
            if accum == tgt_norm or len(accum) >= len(tgt_norm) or i == len(src_seg_idx):
                break

        # 첫 src에 REPLACE_<tgt_tok>, 나머지 DELETE
        tags[consumed[0]] = f"REPLACE_{tgt_tok}"
        for si in consumed[1:]:
            tags[si] = "DELETE"
        last_replace_src = consumed[0]
        j += 1

    # 남은 src → DELETE
    while i < len(src_seg_idx):
        tags[src_seg_idx[i]] = "DELETE"
        i += 1

    # 남은 tgt → 마지막 REPLACE 소스에 INSERT 누적
    if last_replace_src is not None:
        app_str = ""
        while j < len(tgt_seg_idx):
            rest_tok = tgt[tgt_seg_idx[j]]
            app_str = (app_str + f"|INSERT_{rest_tok}") if app_str else f"INSERT_{rest_tok}"
            j += 1
        if app_str:
            if tags[last_replace_src] in ("", "KEEP"):
                tags[last_replace_src] = (tags[last_replace_src] + ("|" if tags[last_replace_src] else "") + app_str) if tags[last_replace_src] else app_str
            else:
                tags[last_replace_src] += f"|{app_str}"

# ------------------------------
# Tag builder (main logic)
# ------------------------------

def build_tags(src: List[str], tgt: List[str]) -> List[str]:
    """
    변경점:
      (src_seg, tgt_seg) 모두 있는 구간을 '_assign_replace_by_tgt_units'로 처리
      → REPLACE 결과물이 항상 'tgt의 개별 토큰' 단위가 되도록 보장
    나머지 로직(타깃만/소스만/매치 KEEP)은 동일
    """
    n, m = len(src), len(tgt)
    matches = [(-1, -1)] + align_tokens(src, tgt) + [(n, m)]
    tags = [""] * n

    for k in range(len(matches) - 1):
        i0, j0 = matches[k]
        i1, j1 = matches[k + 1]

        src_seg_idx = list(range(i0 + 1, i1))
        tgt_seg_idx = list(range(j0 + 1, j1))

        # (A) 타깃만 있는 구간 → 직전 매치 소스에 INSERT
        if k > 0 and len(src_seg_idx) == 0 and len(tgt_seg_idx) > 0:
            prev_src_i, _ = matches[k]
            if prev_src_i >= 0:
                app_str = ""
                for j in tgt_seg_idx:
                    token = tgt[j]
                    app_str = (app_str + f"|INSERT_{token}") if app_str else f"INSERT_{token}"
                if tags[prev_src_i] in ("", "KEEP"):
                    tags[prev_src_i] = (tags[prev_src_i] + ("|" if tags[prev_src_i] else "") + app_str) if tags[prev_src_i] else app_str
                else:
                    tags[prev_src_i] += f"|{app_str}"

        # (B) 소스와 타깃이 모두 있는 구간 → ★ tgt 단위 REPLACE 적용 ★
        if len(src_seg_idx) > 0 and len(tgt_seg_idx) > 0:
            _assign_replace_by_tgt_units(src, tgt, tags, src_seg_idx, tgt_seg_idx)

        # (C) 소스만 있는 구간 → DELETE
        elif len(src_seg_idx) > 0 and len(tgt_seg_idx) == 0:
            for idx in src_seg_idx:
                tags[idx] = "DELETE"

        # (D) 매치 지점은 KEEP(비어 있으면)
        if i1 < n and tags[i1] == "":
            tags[i1] = "KEEP"

    # 누락 보정
    for i in range(n):
        if tags[i] == "":
            tags[i] = "KEEP"
    return tags

# ------------------------------
# JSON input loader
# ------------------------------

def load_pairs_from_json(path: str, array_key: str, err_key: str, cor_key: str):
    """
    하나의 JSON에서 (err_sentence, cor_sentence) 쌍 리스트 반환
    - 최상위가 배열(list)인 형식도 지원
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        records = obj
    else:
        if array_key not in obj:
            raise ValueError(f"JSON에 '{array_key}' 키가 없습니다.")
        records = obj[array_key]
    pairs = []
    for i, rec in enumerate(records):
        if err_key not in rec or cor_key not in rec:
            raise ValueError(f"{i}번째 레코드에 '{err_key}' 또는 '{cor_key}' 키가 없습니다.")
        pairs.append((rec[err_key], rec[cor_key]))
    return pairs

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()   
    ap.add_argument("--json_input", type=str, default=INPUT_PATH, help="단일 JSON 파일 입력 경로")
    ap.add_argument("--json_array_key", type=str, default="data", help="레코드 배열 키(기본: data)")
    ap.add_argument("--json_err_key", type=str, default="err_sentence", help="오류 문장 키")
    ap.add_argument("--json_cor_key", type=str, default="cor_sentence", help="정답 문장 키")
    ap.add_argument("-o", "--output", default=OUTPUT_PATH, help="출력 JSONL 파일 경로")
    ap.add_argument("--kobert_model", type=str, default="monologg/kobert", help="KoBERT 토크나이저 이름/경로")
    args = ap.parse_args()

    # 입력 로드
    pairs = load_pairs_from_json(args.json_input, args.json_array_key, args.json_err_key, args.json_cor_key)

    # KoBERT tokenizer
    kobert_tok = load_kobert(args.kobert_model)

    # 변환
    with open(args.output, "w", encoding="utf-8") as fo:
        for err, cor in pairs:
            src_tok = tokenize_kobert(kobert_tok, err)
            tgt_tok = tokenize_kobert(kobert_tok, cor)
            labels = build_tags(src_tok, tgt_tok)
            rec = {
                "src_token": src_tok,
                "tgt_token": tgt_tok,
                "token_labels": labels,
                "meta": {"src": err, "tgt": cor, "mode": "kobert"}
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
