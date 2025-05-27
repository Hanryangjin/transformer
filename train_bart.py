import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import time
import os
import json
from tqdm import tqdm

from bart_code.bart_model import BartModel
from code_transformer.dataset import SpellingDataset
from code_transformer.sentencepiece_tokenizer import SentencePieceTokenizer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_masks(src, tgt, device):
    # 소스 마스크 생성 [batch_size, 1, 1, seq_len]
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
    
    # 타겟 마스크 생성 [batch_size, 1, seq_len, seq_len]
    tgt_mask = (tgt != 0).unsqueeze(1).to(device)
    
    # 타겟 시퀀스의 어텐션 마스크 생성
    n = tgt.size(1)
    tgt_sub_mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
    tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    tgt_mask = tgt_mask & ~tgt_sub_mask
    
    return src_mask, tgt_mask

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, clip=1):
    model.train()
    total_loss = 0
    batch_count = 0
    
    # tqdm을 사용하여 진행상황 표시
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in progress_bar:
        try:
            # 배치 데이터를 디바이스로 이동하고 long 타입으로 변환
            src = batch['input_ids'].to(device).long()
            tgt = batch['output_ids'].to(device).long()
            
            # 디버깅: 텐서 타입과 shape 확인
            print(f"\n디버깅 정보:")
            print(f"src 타입: {src.dtype}, shape: {src.shape}, device: {src.device}")
            print(f"tgt 타입: {tgt.dtype}, shape: {tgt.shape}, device: {tgt.device}")
            
            # 마스크 생성
            src_mask, tgt_mask = create_masks(src, tgt, device)
            
            # 디버깅: 마스크 정보 확인
            print(f"src_mask 타입: {src_mask.dtype}, shape: {src_mask.shape}, device: {src_mask.device}")
            print(f"tgt_mask 타입: {tgt_mask.dtype}, shape: {tgt_mask.shape}, device: {tgt_mask.device}")
            
            # 순전파 전 텐서 타입 확인
            print(f"모델 입력 전 src 타입: {src.dtype}")
            print(f"모델 입력 전 tgt 타입: {tgt.dtype}")
            
            # 순전파
            optimizer.zero_grad()
            
            # 디버깅: 모델 파라미터 확인
            print("모델 파라미터 타입 확인:")
            for name, param in model.named_parameters():
                if 'embedding' in name:
                    print(f"{name}: {param.dtype}, device: {param.device}")
            
            try:
                _, decoder_output = model(src, src_mask, tgt, tgt_mask)
                print("모델 순전파 성공")
            except Exception as e:
                print(f"모델 순전파 중 오류 발생: {e}")
                print(f"오류 타입: {type(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # 손실 계산
            loss = criterion(decoder_output, tgt)          
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
            # tqdm 진행상황 업데이트
            progress_bar.set_postfix({"손실": f"{loss.item():.4f}"})
            
        except Exception as e:
            print(f"train_epoch - 배치 {batch_idx} 처리 중 오류 발생: {e}")
            print(f"오류 타입: {type(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if batch_count == 0:
        return float('inf')
    
    return total_loss / batch_count

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    batch_count = 0
    
    # tqdm을 사용하여 진행상황 표시
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="검증 진행 중")
    
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            try:
                # 배치 데이터를 디바이스로 이동
                src = batch['input_ids'].to(device).long()
                tgt = batch['output_ids'].to(device).long()
                
                # 디버깅: 텐서 타입과 shape 확인
                print(f"\n디버깅 정보:")
                print(f"src 타입: {src.dtype}, shape: {src.shape}, device: {src.device}")
                print(f"tgt 타입: {tgt.dtype}, shape: {tgt.shape}, device: {tgt.device}")
                
                # 마스크 생성
                src_mask, tgt_mask = create_masks(src, tgt, device)
                
                # 디버깅: 마스크 정보 확인
                print(f"src_mask 타입: {src_mask.dtype}, shape: {src_mask.shape}, device: {src_mask.device}")
                print(f"tgt_mask 타입: {tgt_mask.dtype}, shape: {tgt_mask.shape}, device: {tgt_mask.device}")
                
                # 순전파
                try:
                    _, decoder_output = model(src, src_mask, tgt, tgt_mask)
                    print("모델 순전파 성공")
                except Exception as e:
                    print(f"모델 순전파 중 오류 발생: {e}")
                    print(f"오류 타입: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # 손실 계산
                loss = criterion(decoder_output.view(-1, decoder_output.size(-1)), tgt.view(-1))
                total_loss += loss.item()
                batch_count += 1
                
                # tqdm 진행상황 업데이트
                progress_bar.set_postfix({"손실": f"{loss.item():.4f}"})
                
            except Exception as e:
                print(f"evaluate - 배치 {batch_idx} 처리 중 오류 발생: {e}")
                print(f"오류 타입: {type(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    if batch_count == 0:
        return float('inf')
    
    return total_loss / batch_count

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir, filename):
    """체크포인트를 저장하는 함수"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'체크포인트 저장됨: {checkpoint_path}')
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """체크포인트를 로드하는 함수"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        print(f'체크포인트 로드됨: {checkpoint_path}')
        return epoch, train_loss, val_loss
    except (RuntimeError, FileNotFoundError, KeyError) as e:
        print(f'체크포인트 로드 실패: {e}')
        print('새로운 학습을 시작합니다.')
        return 0, float('inf'), float('inf')

def train_bart(train_data_path, val_data_path, vocab_size, batch_size=32, d_model=768, num_heads=12, 
               num_layers=12, d_ff=3072, dropout=0.1, num_epochs=10, device='cuda', 
               lambda_param=3.0, checkpoint_dir='checkpoints'):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # SentencePiece 토크나이저 초기화 및 학습
    tokenizer = SentencePieceTokenizer(train_data_path, val_data_path, vocab_size=vocab_size, max_length=512)
    dataset = SpellingDataset(train_data_path, val_data_path, tokenizer, max_length=512)
    
    # 데이터로더 생성
    train_dataloader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset.val_dataset, batch_size=batch_size)
    
    # 모델 초기화
    model = BartModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        device=device,
        lambda_param=lambda_param
    ).to(device)
    
    # 모델 토크나이저 설정 -> Mask infilling에서 사용
    model.set_tokenizer(tokenizer)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0은 패딩 토큰 => 패딩 토큰은 loss 계산에서 제외
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 체크포인트에서 학습 재개
    start_epoch = 0
    best_val_loss = float('inf')
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    
    # 최신 체크포인트가 있으면 로드
    if os.path.exists(latest_checkpoint_path):
        start_epoch, _, best_val_loss = load_checkpoint(model, optimizer, latest_checkpoint_path, device)
        print(f'최신 체크포인트에서 학습 재개: 에포크 {start_epoch}')
    
    # 에포크 진행상황 표시
    epoch_progress = tqdm(range(start_epoch, num_epochs), desc="에포크 진행 중")
    
    # 학습 루프
    for epoch in epoch_progress:
        start_time = time.time()
        
        # 학습
        model.train()
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch)
        
        # 검증
        model.eval()
        val_loss = evaluate(model, val_dataloader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # 에포크 진행상황 업데이트
        epoch_progress.set_postfix({
            "학습 손실": f"{train_loss:.4f}",
            "검증 손실": f"{val_loss:.4f}",
            "시간": f"{epoch_time:.2f}s"
        })
        
        # 최신 체크포인트 저장
        save_checkpoint(
            model, optimizer, epoch + 1, train_loss, val_loss,
            checkpoint_dir, 'latest_checkpoint.pt'
        )
        
        # 5 에포크마다 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'
            )
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                checkpoint_dir, 'best_bart_model.pt'
            )
            print('최고 성능 모델 저장됨')
    
    print(f"\n학습 완료! 총 {num_epochs} 에포크 학습됨")
    return model

if __name__ == '__main__':
    # 데이터 경로 설정
    #drive_path = "/content/drive/Othercomputers/학교 컴/대학원 과목/[Coding]"
    drive_path = None
    if drive_path is not None:
        train_data_path = f'{drive_path}/transformer/TrainData/맞춤법오류_SNS.json'
        val_data_path = f'{drive_path}/transformer/ValidationData/맞춤법오류_SNS.json'
    else:
        train_data_path = 'transformer/TrainData/맞춤법오류_SNS.json'
        val_data_path = 'transformer/ValidationData/맞춤법오류_SNS.json'
    
    # 모델 파라미터 설정
    vocab_size = 32000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 체크포인트 디렉토리 설정
    checkpoint_dir = 'checkpoints'
    
    # 모델 학습
    model = train_bart(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        vocab_size=vocab_size,
        batch_size=32,
        device=device,
        checkpoint_dir=checkpoint_dir,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        dropout=0.1,
        num_epochs=50,
        lambda_param=3.0,
    ) 