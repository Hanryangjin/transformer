# Luna 사전학습 + 미세조정 템플릿 (Gradient Checkpointing + FP16 + DeepSpeed ZeRO)

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

drive_path = "/content/drive/Othercomputers/학교 컴/대학원 과목/[Coding]"
# Colab 환경에서는 drive_path로 이동이 필요 -> 아래의 클래스틀을 import 하기 위함
#%cd "$drive_path"

from transformer.luna.model import LunaTransformerEncoder, EditBasedLunaModel, LunaTransformer
from transformer.code_transformer.dataset import SpellingDataset
from transformer.code_transformer.sentencepiece_tokenizer import SentencePieceTokenizer
from transformer.code_transformer.WeightedCELossForGEC import WeightedCELossForGEC
from transformer.code_transformer.EditBasedLoss import EditBasedLoss
from transformer.code_transformer.EditBasedDecoder import EditBasedDecoder

# 안전한 디코딩 함수 추가
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
            dropout_p=0.1,
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
def train():
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 하이퍼파라미터 설정
    BATCH_SIZE = 8
    EPOCHS = 25
    LEARNING_RATE = 0.0001
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    D_FF = 2048
    MAX_SEQ_LENGTH = 1024
    DROPOUT = 0.1
    VOCAB_SIZE = 16000
    max_length = 1024
    PAD_TOKEN_ID = 0  # 패딩 토큰 ID

    """
    # Edit Base Training 하이퍼파라미터
    OPERATION_WEIGHT = 2.0
    TOKEN_WEIGHT = 1.0
    BEAM_SIZE = 5
    """

    # 데이터셋 및 데이터로더 설정
    transformer_path = "/content/drive/Othercomputers/학교 컴/대학원 과목/[Coding]/transformer"
    train_path = os.path.join(transformer_path, 'TrainData/combined_train_dataset.json')
    val_path = os.path.join(transformer_path, 'ValidationData/combined_validation_dataset.json')
    tokenizer = SentencePieceTokenizer(train_path, vocab_size=VOCAB_SIZE, max_length=max_length).tokenizer
    dataset = SpellingDataset(train_path, val_path, tokenizer, max_length)
    train_loader = DataLoader(dataset.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    checkpoints = [f for f in os.listdir(f"{drive_path}/transformer/checkpoints") if f.endswith('.pt')]
    
    model = LunaTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        d_ff = D_FF,
        dropout_p = DROPOUT,
        project_embedding_length = 32,
        max_length = MAX_SEQ_LENGTH
    )
    model = model.to(device)

    # 손실 함수 및 옵티마이저 학습률 스케줄러 설정
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()  # FP16을 위한 Gradient Scaler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

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
    for epoch in range(latest_checknum, EPOCHS + latest_checknum):
        total_loss = 0
        total_edit_ratio = 0
        
        total_gold_edits = 0
        total_pred_edits = 0
        correct_edit_total = 0
        correct_tokens = 0
        total_tokens = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS+latest_checknum}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 배치 데이터 추출
            input_ids = batch['input_ids'].to(device)
            output_ids = batch['output_ids'].to(device)
            
            # input_lengths 계산 (패딩 토큰 0을 제외한 실제 길이)
            input_lengths = (input_ids != PAD_TOKEN_ID).sum(dim=1).to(device)

            decoder_input = output_ids[:, :-1]  # Decoder 입력
            target = output_ids[:, 1:]          # 정답
            
            optimizer.zero_grad()

            with autocast():  # 자동 혼합 정밀도 (FP16)      
                outputs = model(input_ids, input_lengths, decoder_input)
                outputs = outputs.view(-1, outputs.size(-1))
                target = target.contiguous().view(-1)
                """
                outputs = model(input_ids, input_lengths, output_ids)
                outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_len, vocab_size)
                target = output_ids.view(-1)  # (batch_size * seq_len)
                """
                loss = criterion(outputs, target)

            # FP16 학습을 위한 스케일링된 역전파
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 수정 비율 계산
            edit_ratio = ((output_ids != input_ids) & (output_ids != 0)).float().mean().item()
            total_edit_ratio += edit_ratio

            total_loss += loss.item()
            
            # 예측
            pred_ids = outputs.argmax(dim=-1)

            # 수정 비율
            edit_ratio = ((output_ids != input_ids) & (output_ids != PAD_TOKEN_ID)).float().mean().item()
            total_edit_ratio += edit_ratio

            # 정확도 계산용
            correct_tokens += ((pred_ids == target) & (target != PAD_TOKEN_ID)).sum().item()
            total_tokens += (target != PAD_TOKEN_ID).sum().item()

            # 편집 정확도
            gold_edits = ((output_ids[:, 1:] != input_ids[:, 1:]) & (output_ids[:, 1:] != PAD_TOKEN_ID))
            pred_edits = ((pred_ids != input_ids[:, 1:]) & (pred_ids != PAD_TOKEN_ID))
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
                    pred_text = safe_decode(tokenizer, pred_ids[0].cpu().tolist())
                    """
                    # 배치의 첫 번째 샘플 선택
                    input_sample = input_ids[0].cpu().numpy().tolist()
                    output_sample = batch['output_ids'][0].cpu().numpy().tolist()

                    # 예측 결과 계산
                    with torch.no_grad():
                        pred_outputs = model(input_ids[0:1], input_lengths[0:1], output_ids[0:1])
                        pred_ids = pred_outputs.argmax(dim=-1)[0].cpu().numpy().tolist()

                    # 토큰을 텍스트로 변환 (안전하게 처리)
                    input_text = safe_decode(tokenizer, input_sample)
                    output_text = safe_decode(tokenizer, output_sample)
                    pred_text = safe_decode(tokenizer, pred_ids)

                    # 패딩 토큰 제거 (출력용)
                    input_text = input_text.replace('<pad>', '').strip()
                    output_text = output_text.replace('<pad>', '').strip()
                    pred_text = pred_text.replace('<pad>', '').strip()
                    """
                    # 샘플 출력
                    print(f"\n\t샘플 {batch_idx+1}:")
                    print(f"\t> 입력: {input_text}")
                    print(f"\t> 예측: {pred_text}")
                    print(f"\t> 정답: {output_text}")

                except Exception as e:
                    print(f"샘플 출력 중 오류 발생: {e}")

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
        
        # 학습률 조정
        scheduler.step(avg_loss)

        # 체크포인트 저장
        if((epoch + 1) % 5 == 0):
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
def evaluate():
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # (1) 사용자 환경에 맞춰 설정할 부분
    # - CHECKPOINT_DIR: 체크포인트(.pt) 파일들이 저장된 디렉토리 경로
    # - transformer_path: 데이터셋 경로 설정 (기존 train 코드와 일관되도록)
    # - 하이퍼파라미터: VOCAB_SIZE, max_length, BATCH_SIZE, PAD_TOKEN_ID 등
    CHECKPOINT_DIR = f"{drive_path}/transformer/checkpoints"
    transformer_path = f"{drive_path}/transformer"
    VOCAB_SIZE = 16000
    max_length = 1024
    BATCH_SIZE = 8
    PAD_TOKEN_ID = 0

    # (3) 데이터셋 및 DataLoader 준비
    # 사용자 정의 클래스: SentencePieceTokenizer, SpellingDataset, LunaTransformer 등을 이미 import/정의한 상태여야 함
    # 예시:
    # tokenizer = SentencePieceTokenizer(train_path, vocab_size=VOCAB_SIZE, max_length=max_length).tokenizer
    # dataset = SpellingDataset(train_path, val_path, tokenizer, max_length)
    # val_loader = DataLoader(dataset.val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_path = os.path.join(transformer_path, 'TrainData/combined_train_dataset.json')
    val_path   = os.path.join(transformer_path, 'ValidationData/combined_validation_dataset.json')

    # 토크나이저 초기화 (환경에 맞춰 구현체 사용)
    tokenizer = SentencePieceTokenizer(train_path, vocab_size=VOCAB_SIZE, max_length=max_length).tokenizer

    # 데이터셋 객체 생성
    dataset = SpellingDataset(train_path, val_path, tokenizer, max_length)
    val_loader = DataLoader(dataset.val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # (4) 모델 및 criterion 준비
    # LunaTransformer 생성자 인자는 train 시와 동일하게 맞춰야 함
    model = LunaTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=512,           # train 코드에 맞춰 조정
        num_layers=6,
        num_attention_heads=8,
        d_ff=2048,
        dropout_p=0.1,
        project_embedding_length=32,
        max_length=1024        # train 시 positional encoding 길이 등과 일치
    )
    model = model.to(device)

    # 손실 함수: 패딩 토큰을 무시하도록 ignore_index 설정
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    # (5) 최신 체크포인트 로드
    ckpt_path = get_latest_checkpoint(CHECKPOINT_DIR)
    if ckpt_path is None:
        print(">> 체크포인트를 찾을 수 없습니다:", CHECKPOINT_DIR)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        # checkpoint에 저장된 키가 'model_state_dict'일 경우:
        model.load_state_dict(checkpoint['model_state_dict'])
        # 파일 이름에서 에포크 번호 추출
        latest_checkpoint = os.path.basename(ckpt_path)
        epoch_num = int(latest_checkpoint.split('_')[-1].split('.')[0])
        print(f">> 체크포인트 로드: {ckpt_path} (epoch {epoch_num})")

        # (6) Validation 평가 루프
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0

        total_gold_edits = 0
        total_pred_edits = 0
        correct_edits = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)    # (batch, seq_len)
                output_ids = batch['output_ids'].to(device)  # (batch, seq_len)

                # input_lengths 계산: 모델 forward에 필요한 경우
                input_lengths = (input_ids != PAD_TOKEN_ID).sum(dim=1).to(device)  # (batch,)

                # forward: teacher-forcing 형태로 구현된 모델이라면
                logits = model(input_ids, input_lengths, output_ids)  
                # logits shape: (batch, seq_len, vocab_size)

                # 손실 계산
                logits_flat = logits.view(-1, logits.size(-1))    # (batch*seq_len, vocab_size)
                target_flat = output_ids.view(-1)                 # (batch*seq_len,)
                loss = criterion(logits_flat, target_flat)
                # 토큰 개수: padding이 아닌 위치만 카운트
                non_pad_mask = target_flat != PAD_TOKEN_ID
                num_non_pad = non_pad_mask.sum().item()
                total_loss += loss.item() * num_non_pad
                total_tokens += num_non_pad

                # 예측
                pred_ids = logits.argmax(dim=-1)  # (batch, seq_len)

                # 정확히 예측한 토큰 수
                correct_tokens += ((pred_ids == output_ids) & (output_ids != PAD_TOKEN_ID)).sum().item()

                # GEC 편집 지표: 
                # gold edit 위치: output_ids != input_ids, output_ids != PAD
                gold_edits = ((output_ids != input_ids) & (output_ids != PAD_TOKEN_ID))
                # pred edit 위치: pred_ids != input_ids, pred_ids != PAD
                pred_edits = ((pred_ids != input_ids) & (pred_ids != PAD_TOKEN_ID))
                correct_edits += ((pred_ids == output_ids) & gold_edits).sum().item()
                total_gold_edits += gold_edits.sum().item()
                total_pred_edits += pred_edits.sum().item()

        # (7) 지표 계산 및 출력
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            token_acc = correct_tokens / total_tokens
        else:
            avg_loss = float('nan')
            token_acc = float('nan')

        precision = correct_edits / total_pred_edits if total_pred_edits > 0 else 0.0
        recall = correct_edits / total_gold_edits if total_gold_edits > 0 else 0.0
        beta = 0.5
        if precision + recall > 0:
            f0_5 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        else:
            f0_5 = 0.0

        print(f"\n>> Validation 결과 (epoch {epoch_num}):")
        print(f"   평균 손실: {avg_loss:.4f}, 토큰 정확도: {token_acc:.4f}")
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F0.5: {f0_5:.4f}")

if __name__ == '__main__':
    train()
