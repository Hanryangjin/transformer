import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from code_transformer.transformer import Transformer
from code_transformer.attention import _Embedding
import math
import time

def create_masks(src, tgt):
    # 소스 시퀀스의 패딩 마스크 생성
    src_mask = (src != 0).unsqueeze(-2)
    
    # 타겟 시퀀스의 패딩 마스크 생성
    tgt_mask = (tgt != 0).unsqueeze(-2)
    
    # 타겟 시퀀스의 어텐션 마스크 생성 (미래 토큰을 볼 수 없도록)
    n = tgt.size(1)
    tgt_sub_mask = torch.triu(torch.ones((n, n)), diagonal=1).bool()
    tgt_sub_mask = tgt_sub_mask.to(tgt.device)
    tgt_mask = tgt_mask & ~tgt_sub_mask
    
    return src_mask, tgt_mask

def train_epoch(model, dataloader, optimizer, criterion, device, clip=1):
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        # 마스크 생성
        src_mask, tgt_mask = create_masks(src, tgt)
        
        # 순전파
        optimizer.zero_grad()
        output = model(src, tgt, src_mask, tgt_mask)
        
        # 손실 계산
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        
        # 역전파
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            # 마스크 생성
            src_mask, tgt_mask = create_masks(src, tgt)
            
            # 순전파
            output = model(src, tgt, src_mask, tgt_mask)
            
            # 손실 계산
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_transformer(train_dataloader, val_dataloader, vocab_size, d_model=512, num_heads=8, 
                     num_layers=6, d_ff=2048, dropout=0.1, num_epochs=10, device='cuda'):
    
    # 모델 초기화
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0은 패딩 토큰
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 학습 루프
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 학습
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        
        # 검증
        val_loss = evaluate(model, val_dataloader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Time: {epoch_time:.2f}s')
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pt')
            print('모델 저장됨')
    
    return model

if __name__ == '__main__':
    # 데이터로더와 모델 파라미터 설정
    vocab_size = 32000  # 예시 값
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터로더 생성 (실제 데이터셋으로 교체 필요)
    train_dataloader = DataLoader([], batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader([], batch_size=batch_size)
    
    # 모델 학습
    model = train_transformer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        vocab_size=vocab_size,
        device=device
    ) 