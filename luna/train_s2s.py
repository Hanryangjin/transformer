# Luna 사전학습 + 미세조정 (Gradient Checkpointing + FP16 + DeepSpeed ZeRO)

import os, sys, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

drive_path = "/workspace"
# Colab 환경에서는 drive_path로 이동이 필요 -> 아래의 클래스틀을 import 하기 위함
#%cd "$drive_path"

from transformer.luna.model import LunaTransformerEncoder, LunaTransformer
from transformer.code_transformer.dataset import SpellingDataset
from transformer.code_transformer.sentencepiece_tokenizer import SentencePieceTokenizer

# ------------------------
# 안전한 디코딩 함수
# ------------------------
def safe_decode(tokenizer, ids):
    try:
        # 범위를 벗어난 ID 필터링
        vocab_size = tokenizer.get_piece_size()
        valid_ids = [id for id in ids if id < vocab_size]

        # 빈 리스트인 경우 처리
        if not valid_ids:
            return "<빈 시퀀스>"

        return tokenizer.decode(valid_ids)
    except Exception as e:
        return f"디코딩 오류: {e}"

# ------------------------
# 가장 최신에 저장된 체크포인트 path 반환
# ------------------------
"""
    train, evaluate 함수에서 사용하는 양식을 그대로 함수로 옮겨서 정의
"""
def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        raise FileNotFoundError("체크포인트 파일을 찾을 수 없습니다.")

    # 파일 이름에서 에포크 번호 추출
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

# ------------------------
# 1. Luna 모델 정의
# ------------------------
"""
    제거 또는 model.py로 이동
"""
class LunaModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_attention_heads=8):
        super().__init__()
        self.encoder = LunaTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            d_ff=2048,
            dropout_p=0.2,
            project_embedding_length=32,
            max_length=1024
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, input_lengths):
        # 인코더를 통과
        encoder_outputs = self.encoder(input_ids, input_lengths)
        # 언어 모델 헤드를 통과
        logits = self.lm_head(encoder_outputs)
        return logits

# ------------------------
# 2. 학습 설정
# ------------------------
"""
    주석 추가 필요
"""
def label_smoothed_loss(pred, target, epsilon=0.1, ignore_index=0, class_weight=None):
    """
    pred: (B*T, V)  - 로짓
    target: (B*T,)  - 정답 토큰 id
    epsilon: 라벨 스무딩 계수
    ignore_index: PAD id (예: 0)
    class_weight: (V,) or None  - EOS 등에 가중치 부여시 사용
    """
    V = pred.size(-1)
    log_probs = F.log_softmax(pred, dim=-1)          # (N, V)

    # 유효 토큰 마스크
    mask = (target != ignore_index)                  # (N,)
    if mask.sum() == 0:
        return pred.new_tensor(0.0)

    target_clamped = target.clone()
    target_clamped[~mask] = 0                        # one_hot의 인덱스 안전화

    one_hot = F.one_hot(target_clamped, num_classes=V).float()  # (N, V)
    # 라벨 스무딩
    smoothed = (1 - epsilon) * one_hot + (epsilon / V)

    # 클래스 가중치 (예: EOS 1.5)
    if class_weight is not None:
        # (V,) -> (N,V) 브로드캐스트
        smoothed = smoothed * class_weight.unsqueeze(0)

    # 토큰별 손실
    loss_per_tok = -(smoothed * log_probs).sum(dim=-1)          # (N,)
    loss = loss_per_tok[mask].mean()
    return loss

# --- 정렬 기반 근사용 ---
import difflib

def _strip_special(ids, BOS, EOS, PAD):
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if ids and ids[0] == BOS:
        ids = ids[1:]
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return [t for t in ids if t != PAD]

def edit_counts_via_alignment(src_tokens, tgt_tokens, hyp_tokens):
    """
    src→tgt, src→hyp 변환에서 비-동일 구간을 편집으로 보고,
    src 구간 겹침으로 TP를 근사적으로 셉니다. (빠른 모니터링용)
    """
    sm_gold = difflib.SequenceMatcher(a=src_tokens, b=tgt_tokens)
    gold_spans = [op for op in sm_gold.get_opcodes() if op[0] != 'equal']

    sm_pred = difflib.SequenceMatcher(a=src_tokens, b=hyp_tokens)
    pred_spans = [op for op in sm_pred.get_opcodes() if op[0] != 'equal']

    tp = 0
    for tag, gi1, gi2, gj1, gj2 in gold_spans:
        for tag2, pi1, pi2, pj1, pj2 in pred_spans:
            # src 기준 구간이 겹치면 TP로 카운트
            if not (pi2 <= gi1 or pi1 >= gi2):
                tp += 1
                break

    fp = max(0, len(pred_spans) - tp)
    fn = max(0, len(gold_spans) - tp)
    return tp, fp, fn

"""
    train, evaluate 함수는 하나로 두고, 파라미터를 통해 모델을 제어할 수 있도록 수정.
        - train, evaluate에 파라미터 model_type 추가
        - model_type에 대한 사전값은 __init__에서 딕셔너리 구조로 정의
"""
class pNup_s2s:
    def __init__(self):
        # 하이퍼파라미터 설정
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.LEARNING_RATE = 0.0001
        self.D_MODEL = 512
        self.NUM_HEADS = 8
        self.NUM_LAYERS = 6
        self.D_FF = 2048
        self.MAX_SEQ_LENGTH = 128
        self.DROPOUT = 0.2
        self.VOCAB_SIZE = 16000
        self.PAD_TOKEN_ID = 0    # 패딩 토큰 ID
        self.BOS_TOKEN_ID = 2    # 시작 토큰 ID( <s> or [BOS] )
        self.EOS_TOKEN_ID = 3    # 종료 토큰 ID

    def train(self):
        # GPU 사용 가능 여부 확인
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # 데이터셋 및 데이터로더 설정
        """ 
            해당 변수들은 학습/테스트 데이터의 변경이 있지 않는 한 모든 모델에서 공통적으로 사용될 것임. 
                -> __init__에서 정의 또는 전역 변수로 정의
        """
        transformer_path = "/workspace/transformer"
        train_path = os.path.join(transformer_path, 'TrainData/combined_train_dataset.json')
        val_path = os.path.join(transformer_path, 'ValidationData/combined_validation_dataset.json')
        tokenizer = SentencePieceTokenizer(train_path, vocab_size=self.VOCAB_SIZE, max_length=self.MAX_SEQ_LENGTH).tokenizer
        dataset = SpellingDataset(train_path, val_path, tokenizer, self.MAX_SEQ_LENGTH)
        train_loader = DataLoader(dataset.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        checkpoints = [f for f in os.listdir(f"{transformer_path}/checkpoints") if f.endswith('.pt')]
        
        # 모델 객체 생성
        model = LunaTransformer(
            vocab_size=self.VOCAB_SIZE,
            d_model=self.D_MODEL,
            num_layers=self.NUM_LAYERS,
            num_attention_heads=self.NUM_HEADS,
            d_ff = self.D_FF,
            dropout_p = self.DROPOUT,
            project_embedding_length = 32,
            max_length = self.MAX_SEQ_LENGTH
        )
        model = model.to(device)

        # 손실 함수 및 옵티마이저 학습률 스케줄러 설정
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.LEARNING_RATE, weight_decay=0.01)
        scaler = GradScaler(device_type)  # FP16을 위한 Gradient Scaler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        # 체크포인트 로드
        """ $추가. 특정 체크포인트를 불러오는 함수 추가할 것 """
        if not checkpoints:
            print("저장된 체크포인트 없음.")
            latest_checknum = 0
        else:
            # 가장 최근 체크포인트 로드
            checkpoint_path = get_latest_checkpoint(f"{transformer_path}/checkpoints")
            print(f"체크포인트 로드: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # 파일 이름에서 에포크 번호 추출
            latest_checkpoint = os.path.basename(checkpoint_path)
            latest_checknum = int(latest_checkpoint.split('_')[-1].split('.')[0])

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model = model.to(device)

        # 클래스 가중치 설정 (EOS 토큰에 1.5배 가중치 부여)
        eos_weight = 1.5
        class_weight = torch.ones(self.VOCAB_SIZE, device=device)
        class_weight[self.EOS_TOKEN_ID] = eos_weight

        # 학습 루프
        model.train()
        for epoch in range(latest_checknum, self.EPOCHS + latest_checknum):        
            epoch_gold_edit_tok = 0
            epoch_pred_edit_tok = 0
            epoch_nonpad_tok    = 0

            epoch_align_tp = 0
            epoch_align_fp = 0
            epoch_align_fn = 0
            epoch_align_calls = 0

            total_gold_edits = 0
            total_pred_edits = 0

            correct_edit_total = 0
            correct_tokens = 0
            total_tokens = 0

            total_loss_tok = 0
            epoch_loss_tok = 0

            LOG_ALIGN_EVERY = 500
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.EPOCHS+latest_checknum}')

            for batch_idx, batch in enumerate(progress_bar):
                """
                    1. Beam Search 추가.
                    2. 길이 정규화 및 패널티 추가.
                    3. 반복 패널티 수정(초안에서는 적용X. 필요시 추가)
                    4. 스케줄드 샘플링 수정 및 내용 정리(3과 연계)
                    5. 평가 방식 변경(위치별 정확도 → 편집 정확도 : 정렬기반(Levenshtein/ERRANT)으로 편집을 비교하여 P/R/F0.5 산출)
                    6. 유지보수성 향상을 위해 함수로 분리하여 적용.
                        - 대부분의 모델에 사용할 수 있도록 함수화
                        - 해당 부분은 수정 라인 밖의 부분도 동일하게 적용
                    7. 디버깅용 출력문 정리
                        - 중간단계에서의 출력이 필수적이지 않으면 제거 또는 후방으로 이동
                    8. 주석 정리
                        - 함수화 할 경우, 파라미터를 명확히 주석으로 표시
                """
                # 배치 데이터 추출
                input_ids   = batch['input_ids'].to(device)
                output_ids  = batch['output_ids'].to(device)
                
                # input_lengths 계산 (패딩 토큰 0을 제외한 실제 길이)
                input_lengths = (input_ids != self.PAD_TOKEN_ID).sum(dim=1).to(device)

                decoder_input   = output_ids[:, :-1]    # Decoder 입력
                target          = output_ids[:, 1:]     # 정답
                optimizer.zero_grad()

                # ---- 1차 forward: teacher-forcing으로 예측 수집 ----
                with torch.no_grad():
                    logits_tf = model(input_ids, input_lengths, decoder_input)  # (B, T-1, V)
                    next_pred = logits_tf.argmax(dim=-1)                        # (B, T-1)

                # ---- 스케줄드 샘플링 확률: 에포크에 따라 점진 증가 ----
                max_ss = 0.25  # 최댓값(0.1~0.25 권장)
                # latest_checknum이 있는 코드 구조를 고려해 진행률 계산
                ss_prob = min(max_ss, ((epoch - latest_checknum + 1) / self.EPOCHS) * max_ss)

                # ---- 일부 토큰을 모델 예측으로 치환(오토리그레시브 일관성) ----
                if ss_prob > 0.0:
                    # bern: (B, T-1)에서 True인 위치를 치환
                    bern = (torch.rand_like(next_pred.float()) < ss_prob)
                    # 현재 시점 토큰은 "직전 시점의 모델 출력"을 넣는 게 맞음
                    # decoder_input[:, 1:]와 next_pred[:, :-1]를 정렬시켜 치환
                    di_tail  = decoder_input[:, 1:]     # (B, T-2)
                    mix_pred = next_pred[:, :-1]        # (B, T-2)
                    decoder_input[:, 1:] = torch.where(bern[:, :-1], mix_pred, di_tail)

                with autocast(device_type=device_type, dtype=torch.float16):  # 자동 혼합 정밀도 (FP16)      
                    outputs = model(input_ids, input_lengths, decoder_input)

                    # smoothed loss 정의 (1.5배)
                    loss = label_smoothed_loss(outputs, target,
                            epsilon=0.1,
                            ignore_index=self.PAD_TOKEN_ID,
                            class_weight=class_weight)

                # Gradient Clipping
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # --- 훈련 중 진단 로그 계산 ---
                with torch.no_grad():
                    # 1) 자리맞춤 기반 '편집 비율' 빠른 로그 (gold vs pred_tf)  ※ 배치 전체
                    tf_pred = logits_tf.argmax(dim=-1)  # (B, T-1)
                    # BOS + 예측으로 'pred_full' 구성 (출력 길이를 output_ids와 맞춤)
                    pred_full = torch.cat([decoder_input[:, :1], tf_pred], dim=1)  # (B, T)

                    # 자리맟줌 근사(BOS 제외)
                    non_pad = (output_ids[:, 1:] != self.PAD_TOKEN_ID)
                    gold_edits_mask = ((output_ids[:, 1:] != input_ids[:, 1:]) & non_pad)
                    pred_edits_mask = ((pred_full[:, 1:]  != input_ids[:, 1:]) & non_pad)


                    # 토큰 단위로 누적(에포크 전체 기준의 "비율" 계산에 쓰임)
                    epoch_gold_edit_tok += gold_edits_mask.sum().item()
                    epoch_pred_edit_tok += pred_edits_mask.sum().item()
                    epoch_nonpad_tok    += non_pad.sum().item()

                # 2) 정렬 기반 근사 P/R/F0.5
                if (batch_idx % LOG_ALIGN_EVERY or batch_idx == 6697) == 0:
                    with torch.no_grad():
                        S = min(8, input_ids.size(0))
                        tp = fp = fn = 0
                        for b in range(S):
                            src_seq = _strip_special(input_ids[b],  self.BOS_TOKEN_ID, self.EOS_TOKEN_ID, self.PAD_TOKEN_ID)
                            tgt_seq = _strip_special(output_ids[b], self.BOS_TOKEN_ID, self.EOS_TOKEN_ID, self.PAD_TOKEN_ID)
                            hyp_seq = _strip_special(pred_full[b],  self.BOS_TOKEN_ID, self.EOS_TOKEN_ID, self.PAD_TOKEN_ID)
                            tpi, fpi, fni = edit_counts_via_alignment(src_seq, tgt_seq, hyp_seq)
                            tp += tpi; fp += fpi; fn += fni
                        epoch_align_tp += tp
                        epoch_align_fp += fp
                        epoch_align_fn += fn
                        epoch_align_calls += 1

                # 예측
                pred_ids = outputs.argmax(dim=-1)
                pred_ids = pred_ids.view(output_ids.size(0), output_ids.size(1) - 1)

                # 정확도 계산
                target_2d = target.view(pred_ids.size(0), pred_ids.size(1))
                nonpad_loss = (target_2d != self.PAD_TOKEN_ID).sum().item()

                # 토큰 가중 손실 누적
                total_loss_tok += loss.item() * nonpad_loss
                epoch_loss_tok += nonpad_loss
                
                correct_tokens += ((pred_ids == target_2d) & (target_2d != self.PAD_TOKEN_ID)).sum().item()
                total_tokens += (target_2d != self.PAD_TOKEN_ID).sum().item()

                # 편집 정확도(자리맞춤 방식)
                gold_edits = ((output_ids[:, 1:] != input_ids[:, 1:]) & (output_ids[:, 1:] != self.PAD_TOKEN_ID))
                pred_edits = ((pred_ids != input_ids[:, 1:]) & (pred_ids != self.PAD_TOKEN_ID))
                correct_edits = ((pred_ids == output_ids[:, 1:]) & gold_edits)

                total_gold_edits += gold_edits.sum().item()
                total_pred_edits += pred_edits.sum().item()
                correct_edit_total += correct_edits.sum().item()
                # --------------------

                # 진행률 표시줄 업데이트
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    #'edit_ratio': f'{edit_ratio:.2f}'
                })

                # ---- 디버깅용 출력문 ----
                batch_nonpad = (target_2d != self.PAD_TOKEN_ID).sum().item()
                outputs_view = outputs.view(-1, outputs.size(-1))
                target_view = target.contiguous().view(-1)

                # 비정상적인 outputs 텐서 체크
                if not torch.isfinite(outputs_view).all():
                    print("[Debug]배치별 target 비-패딩 토큰 수:", batch_nonpad)
                    print("[Debug]Decoder outputs shape:", outputs_view.shape)
                    print("[Debug]Target shape:", target_view.shape)
                    print("[Debug🚨] outputs 텐서 내 NaN/Inf 존재!")
                    print("예시 출력 (첫 5개):", outputs_view[0][:5])
                    print("최대값:", outputs_view.max().item(), "최소값:", outputs_view.min().item(), "평균값:", outputs_view.mean().item())
                
                # 비정상적인 loss 값 체크
                if not torch.isfinite(loss):
                    print(f"[Debug🚨] 비정상 Loss 발생: {loss.item()}")
                    print("[Debug]현재 learning rate:", scheduler.get_last_lr())
                # --------------------
                
                # 샘플 출력 (각 에포크마다 5개)
                if batch_idx < 5:
                    try:
                        input_text = safe_decode(tokenizer, input_ids[0].cpu().tolist())
                        output_text = safe_decode(tokenizer, output_ids[0].cpu().tolist())
                        pred_ids_with_pad = [self.BOS_TOKEN_ID] + pred_ids[0].cpu().tolist()
                        pred_text = safe_decode(tokenizer, pred_ids_with_pad)

                        # 샘플 출력
                        print(f"\n\t샘플 {batch_idx+1}:")
                        print(f"\t> 입력: {input_text}")
                        print(f"\t> 예측: {pred_text}")
                        print(f"\t> 정답: {output_text}")

                    except Exception as e:
                        print(f"[Error🚨] 샘플 출력 중 오류 발생: {e}")
            
            # ----- 에포크별 평균 손실 및 지표 계산 -----
            # 손실/토큰 정확도
            avg_loss = total_loss_tok / max(1, epoch_loss_tok)
            avg_edit_ratio = (epoch_pred_edit_tok / epoch_nonpad_tok) if epoch_nonpad_tok > 0 else 0.0
            token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

            # 자리 맞춤 기반 P/R/F0.5
            precision = correct_edit_total / total_pred_edits if total_pred_edits > 0 else 0.0
            recall    = correct_edit_total / total_gold_edits if total_gold_edits > 0 else 0.0
            beta = 0.5
            f0_5 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0

            # 에포크 전체 기준 자리맞춤 편집 비율 (토큰 가중 평균)
            epoch_gold_edit_ratio = (epoch_gold_edit_tok / epoch_nonpad_tok) if epoch_nonpad_tok > 0 else 0.0
            epoch_pred_edit_ratio = (epoch_pred_edit_tok / epoch_nonpad_tok) if epoch_nonpad_tok > 0 else 0.0

            # 정렬 기반 근사 P/R/F0.5
            if (epoch_align_tp + epoch_align_fp + epoch_align_fn) > 0:
                align_prec = epoch_align_tp / (epoch_align_tp + epoch_align_fp) if (epoch_align_tp + epoch_align_fp) > 0 else 0.0
                align_reca = epoch_align_tp / (epoch_align_tp + epoch_align_fn) if (epoch_align_tp + epoch_align_fn) > 0 else 0.0
                align_f05  = (1 + beta**2) * align_prec * align_reca / (beta**2 * align_prec + align_reca) if (align_prec + align_reca) > 0 else 0.0
            else:
                align_prec = align_reca = align_f05 = float('nan')

            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Avg Edit Ratio(pred/token-wt): {avg_edit_ratio:.4f}")
            print(f"    Token Acc(2nd pass): {token_acc:.4f}, (pos-based) Precision: {precision:.4f}, Recall: {recall:.4f}, F0.5: {f0_5:.4f}")
            print(f"    gold_edit_ratio={epoch_gold_edit_ratio:.3f} | pred_edit_ratio={epoch_pred_edit_ratio:.3f} | ratio={epoch_pred_edit_ratio/epoch_gold_edit_ratio if epoch_gold_edit_ratio>0 else 0:.3f}")
            print(f"    Align(P/R/F0.5)={align_prec:.3f}/{align_reca:.3f}/{align_f05:.3f} (calls={epoch_align_calls})")

            # CSV 기록
            csv_path = f"{transformer_path}/epoch_metrics.csv"
            write_header = (epoch == 0) and (not os.path.exists(csv_path))
            with open(csv_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow([
                        "epoch", "loss", "edit_ratio", "token_acc", "precision", "recall", "f0.5",
                        "gold_edit_ratio_token_wt", "pred_edit_ratio_token_wt",
                        "align_prec", "align_reca", "align_f0.5", "align_calls"
                    ])
                writer.writerow([
                    epoch + 1,
                    avg_loss,
                    avg_edit_ratio,
                    token_acc,
                    precision,
                    recall,
                    f0_5,
                    epoch_gold_edit_ratio,
                    epoch_pred_edit_ratio,
                    align_prec,
                    align_reca,
                    align_f05,
                    epoch_align_calls
                ])
            # --------------------

            # 학습률 조정
            scheduler.step(avg_loss)

            # 체크포인트 저장
            if((epoch + 1) != 0):
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    #'edit_ratio': avg_edit_ratio,
                    'tokenizer': tokenizer
                }
                torch.save(checkpoint, f"{transformer_path}/checkpoints/luna_model_epoch_{epoch+1}.pt")

    # ------------------------
    # 3. 평가 설정
    # ------------------------
    def evaluate(self,
        beam_size: int = 4,
        max_gen_len: int = 127,         # 디코더 토큰 수(BOS 제외) 상한 (MAX_LENGTH-1 권장)
        length_alpha: float = 0.6,      # 길이 패널티 alpha (0.6~1.0 튜닝)
        repetition_penalty: float = 1.1,# 1.0이면 비활성
        no_repeat_ngram_size: int = 3,  # 0이면 비활성(권장: 2~4)
        diag_tf: bool = False,          # True면 TF-loss/TF-acc도 병행 계산(느려짐)
        dump_for_errant: bool = True,   # ERRANT 입력 덤프 저장
        dump_dir_name: str = "eval_dumps"
    ):
        """
        1) BeamSearch(+length penalty, repetition penalty, no-repeat n-gram)로 프리러닝 생성
        2) 정렬 기반 편집지표(근사)로 Precision/Recall/F0.5 계산
        3) (옵션) TF-loss/TF-Acc 진단
        4) (옵션) ERRANT 덤프(src/hyp/ref) 저장
        5) 샘플 5개: 입력/예측/정답 형식 출력 (train과 동일 스타일)
        """
        import os, math, difflib, csv
        from tqdm import tqdm
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # ----- 환경/경로(기존 train 설정과 일치) -----
        CHECKPOINT_DIR   = f"{drive_path}/transformer/checkpoints"
        transformer_path = f"{drive_path}/transformer"

        # ----- 데이터/토크나이저/로더 -----
        train_path = os.path.join(transformer_path, 'TrainData/combined_train_dataset.json')
        val_path   = os.path.join(transformer_path, 'ValidationData/combined_validation_dataset.json')

        tokenizer = SentencePieceTokenizer(train_path, vocab_size=self.VOCAB_SIZE, max_length=self.MAX_SEQ_LENGTH).tokenizer
        dataset   = SpellingDataset(train_path, val_path, tokenizer, self.MAX_SEQ_LENGTH)
        val_loader = DataLoader(dataset.val_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        # ----- 모델 -----
        model = LunaTransformer(
            vocab_size=self.VOCAB_SIZE,
            d_model=512,
            num_layers=6,
            num_attention_heads=8,
            d_ff=2048,
            dropout_p=0.1,
            project_embedding_length=32,
            max_length=self.MAX_SEQ_LENGTH
        ).to(device)

        # (옵션) TF 진단에서만 사용
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_TOKEN_ID)

        # ----- 체크포인트 로드 -----
        ckpt_path = get_latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path is None:
            print(">> 체크포인트를 찾을 수 없습니다:", CHECKPOINT_DIR)
            return
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        latest_checkpoint = os.path.basename(ckpt_path)
        try:
            epoch_num = int(latest_checkpoint.split('_')[-1].split('.')[0])
        except:
            epoch_num = -1
        print(f">> 체크포인트 로드: {ckpt_path} (epoch {epoch_num})")

        # ===================== 헬퍼들 =====================

        def _strip_special(ids, BOS, EOS, PAD):
            if torch.is_tensor(ids): ids = ids.tolist()
            if ids and ids[0] == BOS: ids = ids[1:]
            if EOS in ids:
                ids = ids[:ids.index(EOS)]
            return [t for t in ids if t != PAD]

        def _first_token_text(ids):
            """첫 비-특수토큰을 텍스트로 (샘플 출력용)"""
            seq = ids.tolist() if torch.is_tensor(ids) else list(ids)
            first = ""
            for t in seq:
                if t in (self.BOS_TOKEN_ID, self.EOS_TOKEN_ID, self.PAD_TOKEN_ID): continue
                try:
                    first = safe_decode(tokenizer, [t])
                except:
                    first = ""
                break
            return first

        def _length_penalty(length, alpha=length_alpha):
            # GNMT length penalty
            return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

        def _apply_repetition_penalty(logits_row, generated, penalty):
            # CTRL 방식: 이미 생성된 토큰의 logit을 조정
            if penalty == 1.0 or len(generated) == 0:
                return logits_row
            unique_tokens = set(generated)
            # in-place 전환 전 clone
            logits_row = logits_row.clone()
            with torch.no_grad():
                for t in unique_tokens:
                    if t < 0 or t >= logits_row.numel():  # 안전장치
                        continue
                    val = logits_row[t]
                    logits_row[t] = val / penalty if val > 0 else val * penalty
            return logits_row

        def _banned_tokens_for_ngram(seq, n):
            # no-repeat n-gram: 마지막 n-1 토큰 prefix와 과거 next-token들을 금지
            if n <= 0 or len(seq) < n - 1:
                return set()
            banned = set()
            prefix = tuple(seq[-(n - 1):]) if n - 1 > 0 else tuple()
            # build n-gram dict
            hist = {}
            for i in range(len(seq) - n + 1):
                gram = tuple(seq[i:i + n])
                pfx = gram[:-1]
                nxt = gram[-1]
                hist.setdefault(pfx, set()).add(nxt)
            if prefix in hist:
                banned = hist[prefix]
            return banned

        @torch.no_grad()
        def beam_search_generate(src_ids_b, src_len_b):
            """
            src_ids_b : (Tsrc,) Long
            src_len_b : () Long
            return: List[int] (BOS ... EOS)
            """
            beams = [ (0.0, [self.BOS_TOKEN_ID]) ]  # (cum_logprob, seq)

            for step in range(max_gen_len):
                new_beams = []
                all_ended = True
                for score, seq in beams:
                    if seq[-1] == self.EOS_TOKEN_ID:
                        new_beams.append((score, seq))
                        continue
                    all_ended = False

                    # 디코더 입력 구성
                    dec_inp = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
                    # 모델 forward (간단구현: 전체 길이 재실행)
                    logits = model(
                        src_ids_b.unsqueeze(0),         # (1, Tsrc)
                        src_len_b.unsqueeze(0),         # (1,)
                        dec_inp                         # (1, t)
                    )                                   # (1, t, V)
                    next_logits = logits[:, -1, :].squeeze(0)  # (V,)

                    # 반복 패널티
                    next_logits = _apply_repetition_penalty(next_logits, seq, repetition_penalty)

                    # no-repeat n-gram 금지 토큰 -inf 처리
                    banned = _banned_tokens_for_ngram(seq, no_repeat_ngram_size)
                    if banned:
                        next_logits[list(banned)] = float('-inf')

                    # 패딩은 생성 금지
                    next_logits[self.PAD_TOKEN_ID] = float('-inf')

                    logprobs = F.log_softmax(next_logits, dim=-1)  # (V,)
                    topk_logprobs, topk_ids = torch.topk(logprobs, beam_size)

                    for lp, idx in zip(topk_logprobs.tolist(), topk_ids.tolist()):
                        new_seq = seq + [idx]
                        new_score = score + lp  # raw logprob 누적
                        new_beams.append((new_score, new_seq))

                if all_ended:
                    break

                # 빔 정렬 및 상위 K 유지
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_size]

            # 완결 빔 선택(길이 패널티 반영)
            def finalized_score(b):
                sc, seq = b
                eff_len = len(seq) if self.EOS_TOKEN_ID not in seq else (seq.index(self.EOS_TOKEN_ID) + 1)
                return sc / _length_penalty(eff_len, length_alpha)

            best = max(beams, key=finalized_score)
            return best[1]

        # ===================== 평가 루프 =====================

        model.eval()

        total_tp = total_fp = total_fn = 0
        num_samples = 0

        # (옵션) TF-loss/acc 진단
        tf_total_loss = 0.0
        tf_total_tok  = 0
        tf_correct_tok = 0

        # (옵션) ERRANT 덤프 준비
        dump_src, dump_hyp, dump_ref = [], [], []

        # 샘플 출력(입력/예측/정답) 5개만
        printed_samples = 0
        MAX_PRINT = 5

        pbar = tqdm(val_loader, desc=f"Validation (beam={beam_size}, α={length_alpha}, rep={repetition_penalty}, ngr={no_repeat_ngram_size})")
        for batch in pbar:
            input_ids  = batch['input_ids'].to(device)   # (B, T)
            output_ids = batch['output_ids'].to(device)  # (B, T)
            input_lengths = (input_ids != self.PAD_TOKEN_ID).sum(dim=1)  # (B,)

            B = input_ids.size(0)
            # 1) 프리러닝 생성 (beam search) — 샘플 단위
            preds = []
            for b in range(B):
                hyp_ids = beam_search_generate(input_ids[b], input_lengths[b])
                preds.append(hyp_ids)

            # 2) 정렬 기반 편집지표(근사) — 마이크로 평균
            for b in range(B):
                src_seq = _strip_special(input_ids[b],  self.BOS_TOKEN_ID, self.EOS_TOKEN_ID, self.PAD_TOKEN_ID)
                ref_seq = _strip_special(output_ids[b], self.BOS_TOKEN_ID, self.EOS_TOKEN_ID, self.PAD_TOKEN_ID)
                hyp_seq = _strip_special(torch.tensor(preds[b], device=device), self.BOS_TOKEN_ID, self.EOS_TOKEN_ID, self.PAD_TOKEN_ID)

                sm_gold = difflib.SequenceMatcher(a=src_seq, b=ref_seq)
                gold_spans = [op for op in sm_gold.get_opcodes() if op[0] != 'equal']

                sm_pred = difflib.SequenceMatcher(a=src_seq, b=hyp_seq)
                pred_spans = [op for op in sm_pred.get_opcodes() if op[0] != 'equal']

                # src 기준 span overlap으로 TP 근사
                tp = 0
                for tag, gi1, gi2, gj1, gj2 in gold_spans:
                    for tag2, pi1, pi2, pj1, pj2 in pred_spans:
                        if not (pi2 <= gi1 or pi1 >= gi2):  # overlap on src
                            tp += 1
                            break
                fp = max(0, len(pred_spans) - tp)
                fn = max(0, len(gold_spans) - tp)

                total_tp += tp
                total_fp += fp
                total_fn += fn

                # 3) ERRANT 덤프 텍스트 저장
                if dump_for_errant:
                    src_txt = safe_decode(tokenizer, [self.BOS_TOKEN_ID] + src_seq + [self.EOS_TOKEN_ID])
                    ref_txt = safe_decode(tokenizer, [self.BOS_TOKEN_ID] + ref_seq + [self.EOS_TOKEN_ID])
                    hyp_txt = safe_decode(tokenizer, preds[b])

                    dump_src.append(src_txt)
                    dump_ref.append(ref_txt)
                    dump_hyp.append(hyp_txt)

                # 4) 샘플 출력(최대 5개, train과 동일 스타일)
                if printed_samples < MAX_PRINT:
                    input_text  = safe_decode(tokenizer, input_ids[b].detach().cpu().tolist())
                    pred_text   = safe_decode(tokenizer, preds[b])
                    output_text = safe_decode(tokenizer, output_ids[b].detach().cpu().tolist())

                    # 첫 토큰 비교(비-특수)
                    pred_first  = _first_token_text(preds[b])
                    gold_first  = _first_token_text(output_ids[b])

                    print(f"> 첫 토큰 비교 | 예측: {pred_first} / 정답: {gold_first}\n")
                    print(f"\t샘플 {printed_samples+1}:")
                    print(f"\t> 입력: {input_text}")
                    print(f"\t> 예측: {pred_text}")
                    print(f"\t> 정답: {output_text}")
                    printed_samples += 1

            num_samples += B

            # 5) (옵션) TF-loss/acc 진단 — 빠르게 보고 싶을 때만
            if diag_tf:
                with torch.no_grad():
                    dec_inp = output_ids[:, :-1]
                    target  = output_ids[:,  1:]
                    logits  = model(input_ids, input_lengths, dec_inp)  # (B, T-1, V)
                    logits_f = logits.view(-1, logits.size(-1))
                    target_f = target.contiguous().view(-1)
                    loss = criterion(logits_f, target_f)
                    nonpad = (target_f != self.PAD_TOKEN_ID).sum().item()
                    tf_total_loss += loss.item() * nonpad
                    tf_total_tok  += nonpad

                    pred_tf = logits.argmax(dim=-1)  # (B, T-1)
                    tf_correct_tok += ((pred_tf == target) & (target != self.PAD_TOKEN_ID)).sum().item()

        # --------- 집계 ---------
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        beta = 0.5
        f0_5 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n>> Validation 결과 (epoch {epoch_num}) [beam={beam_size}, α={length_alpha}, rep={repetition_penalty}, ngr={no_repeat_ngram_size}]")
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F0.5: {f0_5:.4f}")

        if diag_tf and tf_total_tok > 0:
            tf_avg_loss = tf_total_loss / tf_total_tok
            tf_acc = tf_correct_tok / tf_total_tok
            print(f"   (TF 진단) Avg Loss: {tf_avg_loss:.4f}, Token Acc: {tf_acc:.4f}")

        # --------- ERRANT 덤프 저장 ---------
        if dump_for_errant:
            dump_root = os.path.join(transformer_path, dump_dir_name, f"epoch_{epoch_num if epoch_num>=0 else 'NA'}")
            os.makedirs(dump_root, exist_ok=True)
            src_path = os.path.join(dump_root, "src.txt")
            hyp_path = os.path.join(dump_root, "hyp.txt")
            ref_path = os.path.join(dump_root, "ref.txt")

            with open(src_path, "w", encoding="utf-8") as f:
                f.write("\n".join(dump_src))
            with open(hyp_path, "w", encoding="utf-8") as f:
                f.write("\n".join(dump_hyp))
            with open(ref_path, "w", encoding="utf-8") as f:
                f.write("\n".join(dump_ref))

            print(f"\n[ERRANT 덤프] 저장 완료:")
            print(f"  SRC: {src_path}")
            print(f"  HYP: {hyp_path}")
            print(f"  REF: {ref_path}")
            print("  → 외부 스크립트로 ERRANT 채점 실행 권장 (언어 리소스/토크나이저 준비 필요)")


if __name__ == '__main__':
    tne = pNup_s2s()
    tne.train()
    tne.evaluate()