import os
import sys
import json
import numpy as np
from tqdm import tqdm
import sentencepiece as spm
from torch.serialization import add_safe_globals

# 현재 디렉토리를 시스템 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# PyTorch 임포트
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from basic_transformer.model import Transformer
from code_transformer.dataset import SpellingDataset

# 새로운 손실 함수 클래스 정의
class LengthAwareCrossEntropyLoss(nn.Module):
    def __init__(self, pad_id=0, length_weight=0.1):
        super(LengthAwareCrossEntropyLoss, self).__init__()
        self.pad_id = pad_id
        self.length_weight = length_weight  # 길이 차이에 대한 가중치
        
    def forward(self, logits, targets):
        # 배치 크기와 시퀀스 길이 가져오기
        batch_size, seq_len, vocab_size = logits.size()
        
        # 패딩 마스크 생성 (패딩 토큰이 아닌 위치는 1, 패딩 토큰 위치는 0)
        pad_mask = (targets != self.pad_id).float()
        
        # 각 시퀀스의 실제 길이 계산 (패딩 토큰 제외)
        target_lengths = pad_mask.sum(dim=1)  # [batch_size]
        
        # 예측 토큰의 길이 계산 (가장 높은 확률을 가진 토큰이 패딩이 아닌 경우)
        pred_tokens = logits.argmax(dim=-1)  # [batch_size, seq_len]
        pred_mask = (pred_tokens != self.pad_id).float()
        pred_lengths = pred_mask.sum(dim=1)  # [batch_size]
        
        # 길이 차이에 대한 손실 계산
        length_diff = torch.abs(target_lengths - pred_lengths)
        length_loss = length_diff.mean()
        
        # 크로스 엔트로피 손실 계산 (패딩 토큰 제외)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # 패딩 토큰이 아닌 위치만 선택
        non_pad_indices = (targets_flat != self.pad_id)
        logits_flat = logits_flat[non_pad_indices]
        targets_flat = targets_flat[non_pad_indices]
        
        # 크로스 엔트로피 손실 계산
        ce_loss = F.cross_entropy(logits_flat, targets_flat)
        
        # 최종 손실 계산 (크로스 엔트로피 손실 + 길이 차이 손실)
        total_loss = ce_loss + self.length_weight * length_loss
        
        return total_loss

def create_masks(src, tgt, device):
    # 소스 시퀀스의 패딩 마스크 생성
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
    
    # 타겟 시퀀스의 패딩 마스크 생성
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2).to(device)
    
    # 타겟 시퀀스의 디코더 마스크 생성 (미래 토큰을 보지 못하도록)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    
    return src_mask, tgt_mask

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        raise FileNotFoundError("체크포인트 파일을 찾을 수 없습니다.")
    
    # 파일 이름에서 에포크 번호 추출
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

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

def train(model, train_loader, optimizer, criterion, device, epoch, tokenizer=None):
    model.train()
    total_loss = 0
    sample_count = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # 배치 데이터 추출
        input_ids = batch['input_ids'].to(device)
        output_ids = batch['output_ids'].to(device)
        
        optimizer.zero_grad()
        
        # 마스크 생성
        src_mask, tgt_mask = create_masks(input_ids, output_ids, device)
        
        # 모델 실행
        outputs = model(input_ids, output_ids, src_mask, tgt_mask)
        
        # 손실 계산 (새로운 손실 함수는 배치 형태 그대로 사용)
        loss = criterion(outputs, output_ids)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        # 샘플 출력 (각 에포크마다 5개)
        if tokenizer is not None and sample_count < 5:
            try:
                # 배치의 첫 번째 샘플 선택
                input_sample = input_ids[0].cpu().numpy().tolist()
                output_sample = output_ids[0].cpu().numpy().tolist()
                
                # 예측 결과 계산
                with torch.no_grad():
                    pred_outputs = model(input_ids[0:1], input_ids[0:1], src_mask[0:1], tgt_mask[0:1])
                    pred_ids = pred_outputs.argmax(dim=-1)[0].cpu().numpy().tolist()
                
                # 토큰을 텍스트로 변환 (안전하게 처리)
                input_text = safe_decode(tokenizer, input_sample)
                output_text = safe_decode(tokenizer, output_sample)
                pred_text = safe_decode(tokenizer, pred_ids)
                
                # 패딩 토큰 제거 (출력용)
                input_text = input_text.replace('<pad>', '').strip()
                output_text = output_text.replace('<pad>', '').strip()
                pred_text = pred_text.replace('<pad>', '').strip()
                
                # 샘플 출력
                print(f"\n샘플 {sample_count+1}:")
                print(f"입력: {input_text}")
                print(f"예측: {pred_text}")
                print(f"정답: {output_text}")
                
                sample_count += 1
            except Exception as e:
                print(f"샘플 출력 중 오류 발생: {e}")
                # 오류 발생 시 디버깅 정보 출력
                if tokenizer is not None:
                    try:
                        vocab_size = tokenizer.get_piece_size()
                        print(f"토크나이저 어휘 크기: {vocab_size}")
                        print(f"입력 샘플 ID 범위: {min(input_sample)} ~ {max(input_sample)}")
                        print(f"출력 샘플 ID 범위: {min(output_sample)} ~ {max(output_sample)}")
                        print(f"예측 ID 범위: {min(pred_ids)} ~ {max(pred_ids)}")
                    except:
                        pass
    
    return {'loss': total_loss / len(train_loader)}

def main():
    drive_path = "/content/gdrive/Othercomputers/학교 컴/대학원 과목/[Coding]"
    
    # 하이퍼파라미터 설정
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    D_FF = 2048
    MAX_SEQ_LENGTH = 512
    DROPOUT = 0.1
    VOCAB_SIZE = 16000

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA 사용 불가능, CPU 사용")

    # 데이터셋 및 데이터로더 생성
    dataset = SpellingDataset(f'{drive_path}/transformer/TrainData/맞춤법오류_SNS.json', MAX_SEQ_LENGTH, VOCAB_SIZE)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    add_safe_globals([spm.SentencePieceProcessor])

    checkpoints = [f for f in os.listdir(f"{drive_path}/transformer/checkpoints") if f.endswith('.pt')]
    if not checkpoints:
        print("저장된 체크포인트 없음.")
        # 모델 초기화
        model = Transformer(
            src_vocab_size=dataset.vocab_size,
            tgt_vocab_size=dataset.vocab_size,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            d_ff=D_FF,
            max_seq_length=MAX_SEQ_LENGTH,
            dropout=DROPOUT
        ).to(device)
        latest_checknum = 0
        tokenizer = dataset.tokenizer
    else:
        # 가장 최근 체크포인트 로드
        checkpoint_path = get_latest_checkpoint(f"{drive_path}/transformer/checkpoints")
        print(f"체크포인트 로드: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 파일 이름에서 에포크 번호 추출
        latest_checkpoint = os.path.basename(checkpoint_path)
        latest_checknum = int(latest_checkpoint.split('_')[-1].split('.')[0])
        
        # 토크나이저 로드
        tokenizer = checkpoint['tokenizer']

        # Transformer 모델 초기화
        model = Transformer(
            src_vocab_size=8000,  # SentencePiece 어휘 크기
            tgt_vocab_size=8000,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_length=512,
            dropout=0.1
        ).to(device)
        
        # 모델 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 손실 함수 및 옵티마이저 설정
    criterion = LengthAwareCrossEntropyLoss(pad_id=SpellingDataset.PAD_ID, length_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 체크포인트에서 옵티마이저 상태 로드
    if checkpoints:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 모델 학습
    for epoch in range(latest_checknum, EPOCHS):
        # 학습
        train_metrics = train(model, train_loader, optimizer, criterion, device, epoch, tokenizer)
        
        # 체크포인트 저장
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_metrics['loss'],
            'tokenizer': tokenizer
        }
        torch.save(checkpoint, f"{drive_path}/transformer/checkpoints/model_epoch_{epoch+1}.pt")
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {train_metrics["loss"]:.4f}')

if __name__ == "__main__":
    main() 