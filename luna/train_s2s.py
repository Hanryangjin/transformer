# Luna 사전학습 + 미세조정 템플릿 (Gradient Checkpointing + FP16 + DeepSpeed ZeRO)

"""
# $수정필. train과 evaluate 모두 하나의 클래스로 통합.
# 하이퍼 파라미터를 동일한 곳에서 초기화 하고 기능적인 부분은 분리.
# 각 함수의 내부 파라미터의 변동값에 주의.
"""

import os, sys, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

drive_path = "/workspace"
# Colab 환경에서는 drive_path로 이동이 필요 -> 아래의 클래스틀을 import 하기 위함
#%cd "$drive_path"

from transformer.luna.model import LunaTransformerEncoder, EditBasedLunaModel, LunaTransformer
from transformer.code_transformer.dataset import SpellingDataset
from transformer.code_transformer.sentencepiece_tokenizer import SentencePieceTokenizer
from transformer.code_transformer.WeightedCELossForGEC import WeightedCELossForGEC
from transformer.code_transformer.EditBasedLoss import EditBasedLoss
from transformer.code_transformer.EditBasedDecoder import EditBasedDecoder

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

"""
def label_smoothed_loss(pred, target, epsilon=0.1):
    n_class = pred.size(-1)
    log_probs = F.log_softmax(pred, dim=-1)

    one_hot = F.one_hot(target, num_classes=n_class).float()
    smoothed_target = (1 - epsilon) * one_hot + epsilon / n_class

    loss = -(smoothed_target * log_probs).sum(dim=-1)
    return loss.mean()
"""
class pNup_s2s:
    def __init__(self):
        # 하이퍼파라미터 설정
        self.BATCH_SIZE = 32
        self.EPOCHS = 25
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
        print(f"Using device: {device}")

        # 데이터셋 및 데이터로더 설정
        transformer_path = "/workspace/transformer"
        train_path = os.path.join(transformer_path, 'TrainData/combined_train_dataset.json')
        val_path = os.path.join(transformer_path, 'ValidationData/combined_validation_dataset.json')
        tokenizer = SentencePieceTokenizer(train_path, vocab_size=self.VOCAB_SIZE, max_length=self.MAX_SEQ_LENGTH).tokenizer
        dataset = SpellingDataset(train_path, val_path, tokenizer, self.MAX_SEQ_LENGTH)
        train_loader = DataLoader(dataset.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        checkpoints = [f for f in os.listdir(f"{drive_path}/transformer/checkpoints") if f.endswith('.pt')]
        
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
        """
        weight = torch.ones(self.VOCAB_SIZE, device=device)
        weight[self.EOS_TOKEN_ID] = 1.5  # 1.2~2.0 범위에서 탐색
        criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_TOKEN_ID, weight=weight)
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.LEARNING_RATE, weight_decay=0.01)
        scaler = GradScaler()  # FP16을 위한 Gradient Scaler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # 체크포인트 로드
        if not checkpoints:
            print("저장된 체크포인트 없음.")
            latest_checknum = 0
        else:
            # 가장 최근 체크포인트 로드
            checkpoint_path = get_latest_checkpoint(f"{drive_path}/transformer/checkpoints")
            print(f"체크포인트 로드: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # 파일 이름에서 에포크 번호 추출
            latest_checkpoint = os.path.basename(checkpoint_path)
            latest_checknum = int(latest_checkpoint.split('_')[-1].split('.')[0])

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model = model.to(device)

        # 학습 루프
        model.train()
        for epoch in range(latest_checknum, self.EPOCHS + latest_checknum):
            total_loss = 0
            total_edit_ratio = 0
            
            total_gold_edits = 0
            total_pred_edits = 0
            correct_edit_total = 0
            correct_tokens = 0
            total_tokens = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.EPOCHS+latest_checknum}')

            for batch_idx, batch in enumerate(progress_bar):
                # 배치 데이터 추출
                input_ids = batch['input_ids'].to(device)
                output_ids = batch['output_ids'].to(device)
                
                # input_lengths 계산 (패딩 토큰 0을 제외한 실제 길이)
                input_lengths = (input_ids != self.PAD_TOKEN_ID).sum(dim=1).to(device)

                decoder_input = output_ids[:, :-1]  # Decoder 입력
                target = output_ids[:, 1:]          # 정답
                optimizer.zero_grad()

                # 1) 1차 forward (teacher-forcing) - 예측을 얻기 위한 용도
                logits_tf = model(input_ids, input_lengths, decoder_input)     # (B, T-1, V)
                with torch.no_grad():
                    next_pred = logits_tf.argmax(dim=-1)                       # (B, T-1)

                # 2) 스케줄드 샘플링 확률(에폭별로 점감)
                #    예: 0ep=0.00 → 25ep=0.25까지 증가 (원하면 반대로도 가능)
                max_ss = 0.25
                ss_prob = min(max_ss, (epoch - latest_checknum + 1) / self.EPOCHS * max_ss)

                # 3) 섞기: BOS 다음 시점부터 일부를 모델 예측으로 치환
                if ss_prob > 0.0:
                    bern = torch.rand_like(next_pred.float(), device=device) < ss_prob  # (B, T-1)
                    # decoder_input[:,1:] 위치에 next_pred[:,:-1]을 매칭시켜 주입
                    #  - 현재 시점 토큰은 "직전 시점의 모델 출력"을 넣는 것이 맞음
                    mix_pred = torch.zeros_like(decoder_input[:, 1:])
                    mix_pred.copy_(next_pred[:, :-1])
                    # 치환
                    di_tail = decoder_input[:, 1:]
                    decoder_input[:, 1:] = torch.where(bern[:, :-1], mix_pred, di_tail)

                ### 디버깅용 임시 변경 : autocast 제거
                #with autocast():  # 자동 혼합 정밀도 (FP16)      
                outputs = model(input_ids, input_lengths, decoder_input)
                outputs = outputs.view(-1, outputs.size(-1))
                target = target.contiguous().view(-1)

                if not torch.isfinite(outputs).all():
                    print("[Debug]배치별 target 비-패딩 토큰 수:", (target != self.PAD_TOKEN_ID).sum(dim=1).tolist())
                    print("[Debug]Decoder outputs shape:", outputs.shape)
                    print("[Debug]Target shape:", target.shape)
                    print("[Debug🚨] outputs 텐서 내 NaN/Inf 존재!")
                    print("예시 출력 (첫 5개):", outputs[0][:5])
                    print("최대값:", outputs.max().item(), "최소값:", outputs.min().item(), "평균값:", outputs.mean().item())

                eos_weight = 1.5
                class_weight = torch.ones(self.VOCAB_SIZE, device=device)
                class_weight[self.EOS_TOKEN_ID] = eos_weight

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

                # 수정 비율 계산
                edit_ratio = ((output_ids != input_ids) & (output_ids != 0)).float().mean().item()
                total_edit_ratio += edit_ratio

                total_loss += loss.item()
                if not torch.isfinite(loss):
                    print(f"[Debug🚨] 비정상 Loss 발생: {loss.item()}")
                    print("[Debug]현재 learning rate:", scheduler.optimizer.param_groups[0]['lr'])
                
                # 예측
                pred_ids = outputs.argmax(dim=-1)
                pred_ids = pred_ids.view(output_ids.size(0), output_ids.size(1) - 1)

                # 수정 비율
                edit_ratio = ((output_ids != input_ids) & (output_ids != self.PAD_TOKEN_ID)).float().mean().item()
                total_edit_ratio += edit_ratio

                # 정확도 계산
                target_2d = target.view(pred_ids.size(0), pred_ids.size(1))

                correct_tokens += ((pred_ids == target_2d) & (target_2d != self.PAD_TOKEN_ID)).sum().item()
                total_tokens += (target_2d != self.PAD_TOKEN_ID).sum().item()

                # 편집 정확도
                gold_edits = ((output_ids[:, 1:] != input_ids[:, 1:]) & (output_ids[:, 1:] != self.PAD_TOKEN_ID))
                pred_edits = ((pred_ids != input_ids[:, 1:]) & (pred_ids != self.PAD_TOKEN_ID))
                correct_edits = ((pred_ids == output_ids[:, 1:]) & gold_edits)

                total_gold_edits += gold_edits.sum().item()
                total_pred_edits += pred_edits.sum().item()
                correct_edit_total += correct_edits.sum().item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    #'edit_ratio': f'{edit_ratio:.2f}'
                })
                
                # 샘플 출력 (각 에포크마다 5개)
                if batch_idx < 5:
                    try:
                        input_text = safe_decode(tokenizer, input_ids[0].cpu().tolist())
                        output_text = safe_decode(tokenizer, output_ids[0].cpu().tolist())
                        pred_ids_with_pad = [self.BOS_TOKEN_ID] + pred_ids[0].cpu().tolist()
                        pred_text = safe_decode(tokenizer, pred_ids_with_pad)

                        # === 첫 토큰 비교 디버그 추가 ===
                        pred_first = safe_decode(tokenizer, [pred_ids[0, 0].item()])
                        gold_first = safe_decode(tokenizer, [output_ids[0, 1].item()])  # [BOS] 다음 토큰
                        print(f"\t> 첫 토큰 비교 | 예측: {pred_first}-")
                        print(f"\t> 첫 토큰 비교 | 정답: {gold_first}-")
                        # =============================

                        # 샘플 출력
                        print(f"\t샘플 {batch_idx+1}:")
                        print(f"\t> 입력: {input_text}")
                        print(f"\t> 예측: {pred_text}")
                        print(f"\t> 정답: {output_text}")

                    except Exception as e:
                        print(f"[Error🚨] 샘플 출력 중 오류 발생: {e}")

            avg_loss = total_loss / len(train_loader)
            avg_edit_ratio = total_edit_ratio / len(train_loader)
            
            token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

            precision = correct_edit_total / total_pred_edits if total_pred_edits > 0 else 0.0
            recall = correct_edit_total / total_gold_edits if total_gold_edits > 0 else 0.0
            beta = 0.5
            if precision + recall > 0:
                f0_5 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            else:
                f0_5 = 0.0

            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Avg Edit Ratio: {avg_edit_ratio:.4f}")
            print(f"   Token Acc: {token_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F0.5: {f0_5:.4f}")
            #print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Average Edit Ratio: {avg_edit_ratio:.4f}")
            
            with open(f"{transformer_path}\\epoch_metrics.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                if epoch == 0:
                    writer.writerow(["epoch", "loss", "edit_ratio", "token_acc", "precision", "recall", "f0.5"])
                writer.writerow([
                    epoch + 1,
                    avg_loss,
                    avg_edit_ratio,
                    token_acc,
                    precision,
                    recall,
                    f0_5
                ])

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
                torch.save(checkpoint, f"{drive_path}/transformer/checkpoints/luna_model_epoch_{epoch+1}.pt")

    # ------------------------
    # 3. 평가 설정
    # ------------------------
    def evaluate(self):
        import os, torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        import torch.nn.functional as F

        # ===== 기본 설정 =====
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        CHECKPOINT_DIR   = f"{drive_path}/transformer/checkpoints"
        transformer_path = f"{drive_path}/transformer"
        train_path = os.path.join(transformer_path, 'TrainData/combined_train_dataset.json')
        val_path   = os.path.join(transformer_path, 'ValidationData/combined_validation_dataset.json')
        printed_guard = False

        # ===== 토크나이저/데이터셋 =====
        tokenizer = SentencePieceTokenizer(val_path, vocab_size=self.VOCAB_SIZE, max_length=self.MAX_SEQ_LENGTH).tokenizer
        dataset = SpellingDataset(train_path, val_path, tokenizer, self.MAX_SEQ_LENGTH)
        val_loader = DataLoader(dataset.val_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        # ===== 모델/criterion =====
        model = LunaTransformer(
            vocab_size=self.VOCAB_SIZE,
            d_model=self.D_MODEL,
            num_layers=self.NUM_LAYERS,
            num_attention_heads=self.NUM_HEADS,
            d_ff=self.D_FF,
            dropout_p=self.DROPOUT,
            project_embedding_length=32,
            max_length=self.MAX_SEQ_LENGTH
        ).to(device)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_TOKEN_ID)

        # ===== 체크포인트 로드 =====
        ckpt_path = get_latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path is None:
            print(">> 체크포인트를 찾을 수 없습니다:", CHECKPOINT_DIR)
            return
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        latest_checkpoint = os.path.basename(ckpt_path)
        epoch_num = int(latest_checkpoint.split('_')[-1].split('.')[0])
        print(f">> 체크포인트 로드: {ckpt_path} (epoch {epoch_num})")

        # ==== [추가] free-running 생성 함수 ====
        @torch.no_grad()
        def free_run_generate(model, input_ids, input_lengths,
                            max_len, bos_id, eos_id, pad_id):
            """
            입력 배치에 대해 오토리그레시브로 디코딩(teacher-forcing 없음).
            반환: pred_full (B, T_full) : [BOS] ... EOS ... PAD
            """
            B = input_ids.size(0)
            # 시작: [BOS] + PAD...  형태로 초기화
            dec = torch.full((B, 1), bos_id, dtype=input_ids.dtype, device=input_ids.device)

            # 이미 EOS를 낸 샘플 마스크
            finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

            # 디코딩 제약 하이퍼파라미터
            repetition_penalty = 1.12     # 1.1~1.2 추천
            eos_bonus = 0.25              # 0.2~0.5 사이 실험
            no_repeat_ngram = 3           # 2나 3 추천
            max_same_token = 8            # 동일 토큰 N회 연속 방지

            # 최대 길이-1 만큼 반복 (BOS 포함하므로 -1)
            for t in range(1, max_len):
                logits = model(input_ids, input_lengths, dec)      # (B, t, V)
                step_logits = logits[:, -1, :].clone()             # (B, V)

                # --- (1) repetition penalty ---
                # 지금까지 쓴 토큰(out=dec)의 빈도를 이용해 사용한 토큰의 로짓을 조금 낮춤
                for b in range(B):
                    used = dec[b].unique()
                    step_logits[b, used] /= repetition_penalty

                # --- (2) no-repeat n-gram (아주 간단 버전) ---
                # prefix가 동일한 n-gram 반복을 억제: 직전 토큰 시퀀스가 충분히 길면 일부 상위 후보를 -inf 처리
                if no_repeat_ngram >= 2 and dec.size(1) >= (no_repeat_ngram - 1):
                    prefix = dec[:, -(no_repeat_ngram - 1):]    # (B, n-1)
                    # 간단 버전: 각 배치마다 상위 K 후보를 막아 보수적으로 반복 억제
                    K = 5
                    topk = step_logits.topk(K, dim=-1).indices  # (B, K)
                    # prefix가 짧을 땐 실제 n-gram 인덱싱을 하지 않고 보수적으로 topK를 낮춤
                    step_logits.scatter_(1, topk, -1e9)

                # --- (3) EOS 보너스 ---
                step_logits[:, eos_id] += eos_bonus

                # --- (4) 동일 토큰 연속 방지 ---
                if dec.size(1) >= max_same_token:
                    last_tok = dec[:, -1]                      # (B,)
                    run = (dec[:, -(max_same_token-1):] == last_tok.unsqueeze(1)).all(dim=1)  # True면 직전 N-1이 전부 같음
                    # 연속 run인 배치에 대해 해당 토큰을 강제로 배제
                    step_logits[run, last_tok] = -1e9

                next_ids = step_logits.argmax(dim=-1)          # (B,)
                next_ids = torch.where(finished, torch.full_like(next_ids, pad_id), next_ids)
                finished |= (next_ids == eos_id)

                dec = torch.cat([dec, next_ids.unsqueeze(1)], dim=1)
                if finished.all():
                    break

            # 길이 모자라면 PAD로 우측 패딩
            if dec.size(1) < max_len:
                pad_cols = max_len - dec.size(1)
                pad = torch.full((B, pad_cols), pad_id, dtype=dec.dtype, device=dec.device)
                dec = torch.cat([dec, pad], dim=1)

            return dec  # (B, max_len)

        # ===== 평가 루프 =====
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_eval_non_pad = 0
        correct_tokens = 0

        total_gold_edits = 0
        total_pred_edits = 0
        correct_edits = 0

        first_batch_guard_printed = False

        # 샘플 출력 개수 컨트롤 (원하면 3으로 바꿔도 됨)
        N_SAMPLES = 5
        printed_examples = 0

        # 디코딩용 함수: PAD 제거 + BOS/EOS 잘라내기
        def _strip_special(ids, bos_id=self.BOS_TOKEN_ID, eos_id=self.EOS_TOKEN_ID, pad_id=self.PAD_TOKEN_ID):
            # 리스트/텐서 모두 지원
            if torch.is_tensor(ids):
                ids = ids.tolist()
            # 앞쪽 BOS 제거
            if ids and ids[0] == bos_id:
                ids = ids[1:]
            # EOS 이후 잘라내기
            if eos_id in ids:
                cut = ids.index(eos_id)
                ids = ids[:cut]
            # PAD 제거
            ids = [t for t in ids if t != pad_id]
            return ids

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids  = batch['input_ids'].to(device)     # (B, T_full)
                output_ids = batch['output_ids'].to(device)    # (B, T_full) = [BOS ... EOS PAD...]

                input_lengths = (input_ids != self.PAD_TOKEN_ID).sum(dim=1)

                # ---- teacher-forcing 기반 loss만 계산 (진단용) ----
                decoder_input = output_ids[:, :-1].contiguous()
                target        = output_ids[:,  1:].contiguous()   # (B, T-1)

                logits = model(input_ids, input_lengths, decoder_input)  # (B, T-1, V)

                if not first_batch_guard_printed:
                    # 정렬/shape 가드
                    eos_pos = (output_ids[0] != self.PAD_TOKEN_ID).sum() - 1
                    print("[VAL] BOS/EOS (should be 2,3):", output_ids[0,0].item(), output_ids[0, eos_pos].item())
                    print("[VAL] shapes (logits vs target):", logits.shape, target.shape)
                    print("[VAL] non-pad ratio in target:", (target != self.PAD_TOKEN_ID).float().mean().item())
                    print("[VAL] sample target head:", target[0, :10].tolist())
                    print("[VAL] sample pred head  :", logits.argmax(-1)[0, :10].tolist())
                    print("[VAL] sample decoder_inp:", decoder_input[0, :10].tolist())
                    first_batch_guard_printed = True

                loss = criterion(logits.view(-1, self.VOCAB_SIZE), target.view(-1))
                non_pad_mask = (target != self.PAD_TOKEN_ID)
                num_non_pad = non_pad_mask.sum().item()
                total_loss += loss.item() * num_non_pad
                total_tokens += num_non_pad

                # ---- (E’) free-running 생성으로 예측/지표 계산 ----
                pred_full = free_run_generate(
                    model=model,
                    input_ids=input_ids,
                    input_lengths=input_lengths,
                    max_len=output_ids.size(1),   # 정답 길이에 맞춰 생성
                    bos_id=self.BOS_TOKEN_ID,
                    eos_id=self.EOS_TOKEN_ID,
                    pad_id=self.PAD_TOKEN_ID
                )  # (B, T_full)

                # 토큰 정확도는 free-running 예측 기준으로(실사용과 일치)
                non_pad_full = (output_ids != self.PAD_TOKEN_ID)
                correct_tokens += ((pred_full == output_ids) & non_pad_full).sum().item()

                total_eval_non_pad += non_pad_full.sum().item()

                # 편집 지표(입력/정답/예측을 같은 프레임에서 비교)
                valid_mask = (output_ids != self.PAD_TOKEN_ID) & (output_ids != self.BOS_TOKEN_ID) & (output_ids != self.EOS_TOKEN_ID)
                gold_edits = (output_ids != input_ids) & valid_mask
                pred_edits = (pred_full  != input_ids) & valid_mask

                correct_edits   += ((pred_full == output_ids) & gold_edits).sum().item()
                total_gold_edits += gold_edits.sum().item()
                total_pred_edits += pred_edits.sum().item()

                # 배치에서 몇 개 뽑아 "입력/예측/정답 + 첫 토큰 비교" 출력
                if printed_examples < N_SAMPLES:
                    bsz = input_ids.size(0)
                    # 가능한 만큼만 출력
                    take = min(N_SAMPLES - printed_examples, bsz)
                    for i in range(take):
                        # 첫 토큰 비교(예측은 pred_full[:,1], 정답은 output_ids[:,1])
                        try:
                            pred_first_id = pred_full[i, 1].item()
                            gold_first_id = output_ids[i, 1].item()
                            pred_first = safe_decode(tokenizer, [pred_first_id])
                            gold_first = safe_decode(tokenizer, [gold_first_id])
                            print(f"> 첫 토큰 비교 | 예측: {pred_first} / 정답: {gold_first}")
                        except Exception as e:
                            print(f"> 첫 토큰 비교 중 오류: {e}")

                        # 본문 디코딩(특수토큰/패딩 제거)
                        in_ids   = _strip_special(input_ids[i])
                        pr_ids   = _strip_special(pred_full[i])
                        out_ids  = _strip_special(output_ids[i])

                        input_text  = safe_decode(tokenizer, in_ids)
                        pred_text   = safe_decode(tokenizer, pr_ids)
                        output_text = safe_decode(tokenizer, out_ids)

                        print(f"\n\t샘플 {printed_examples + 1}:")
                        print(f"\t> 입력: {input_text}")
                        print(f"\t> 예측: {pred_text}")
                        print(f"\t> 정답: {output_text}")

                        printed_examples += 1
                        if printed_examples >= N_SAMPLES:
                            break

                if not printed_guard:
                    eos_rate = (pred_full[:, 1:] == self.EOS_TOKEN_ID).float().mean().item()
                    print("[VAL] EOS rate(after first token):", eos_rate)

                    # 입력 무시 여부를 간단히 체크(샘플 몇 개만 섞어서)
                    idx = torch.arange(input_ids.size(0), device=input_ids.device)
                    idx = idx[torch.randperm(idx.numel())[: min(8, idx.numel())]]
                    pred_shuf = free_run_generate(
                        model, input_ids[idx], (input_ids[idx] != self.PAD_TOKEN_ID).sum(1),
                        max_len=output_ids.size(1),
                        bos_id=self.BOS_TOKEN_ID, eos_id=self.EOS_TOKEN_ID, pad_id=self.PAD_TOKEN_ID
                    )
                    # EOS rate가 0에 가깝다면 EOS가 거의 안 나옴 → 디코딩 무한반복 경향
                    same_ratio = (pred_full[idx] == pred_shuf).float().mean().item()
                    print("[VAL] Input-agnostic ratio:", same_ratio)

                    def _strip(ids, BOS, EOS, PAD):
                        if torch.is_tensor(ids): ids = ids.tolist()
                        if ids and ids[0] == BOS: ids = ids[1:]
                        if EOS in ids:
                            ids = ids[:ids.index(EOS)]
                        return [t for t in ids if t != PAD]

                    # 배치에서 4개만 샘플 비교
                    k = min(4, pred_full.size(0))
                    same = 0
                    for j in range(k):
                        a = _strip(pred_full[j], self.BOS_TOKEN_ID, self.EOS_TOKEN_ID, self.PAD_TOKEN_ID)
                        b = _strip(pred_shuf[j], self.BOS_TOKEN_ID, self.EOS_TOKEN_ID, self.PAD_TOKEN_ID)
                        same += 1.0 if a == b else 0.0
                    print("[VAL] Input-agnostic (stripped) mean:", same / k)

                    printed_guard = True

        # ===== 지표 집계 =====
        if total_tokens > 0:
            avg_loss  = total_loss / total_tokens        # teacher-forcing loss (진단용)
        else:
            avg_loss = float('nan')

        # free-running 기준 토큰 정확도/편집 지표
        token_acc = correct_tokens / total_eval_non_pad if total_eval_non_pad > 0 else 0.0
        precision = (correct_edits / total_pred_edits) if total_pred_edits > 0 else 0.0
        recall    = (correct_edits / total_gold_edits) if total_gold_edits > 0 else 0.0
        beta = 0.5
        f0_5 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n>> Validation 결과 (epoch {epoch_num}) [free-running]:")
        print(f"   (TF-loss 진단) 평균 손실: {avg_loss:.4f}")
        print(f"   Token Acc: {token_acc:.4f}")
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F0.5: {f0_5:.4f}")

if __name__ == '__main__':
    tne = pNup_s2s()
    tne.train()
    tne.evaluate()

