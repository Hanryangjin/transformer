import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from bert_code.bert_model import BertModel, BertForMaskedLM
from bert_code.tokenizer import BertTokenizer

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

class BertDataset(Dataset):
    # 토큰 ID 상수 정의
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    
    def __init__(self, texts, tokenizer, max_len=512, mlm_probability=0.15, device=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        self.device = device
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = self.tokenizer.encode(text)
        
        # MLM을 위한 마스킹
        labels = input_ids.clone()
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 마스킹되지 않은 토큰은 손실 계산에서 제외
        
        # 80%는 [MASK]로, 10%는 랜덤 토큰으로, 10%는 원래 토큰 유지
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.vocab['[MASK]']
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.vocab), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 디바이스 설정이 있는 경우 텐서를 해당 디바이스로 이동
        if self.device is not None:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

def train_bert(model, train_dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            # 데이터가 이미 디바이스에 있는지 확인하고, 없으면 이동
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            if input_ids.device != device:
                input_ids = input_ids.to(device)
            if labels.device != device:
                labels = labels.to(device)
            
            optimizer.zero_grad()
            prediction_scores, _ = model(input_ids, labels=labels)
            
            # 손실 계산 (새로운 손실 함수 사용)
            loss = criterion(prediction_scores, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

def main():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 초기화
    vocab_size = 30522  # BERT 기본 어휘 크기
    model = BertModel(vocab_size=vocab_size)
    model = BertForMaskedLM(model)
    model = model.to(device)
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # 데이터셋 및 데이터로더 설정
    tokenizer = BertTokenizer()
    # 여기에 실제 학습 데이터를 로드하는 코드 추가
    texts = ["예시 문장 1", "예시 문장 2"]  # 실제 데이터로 교체 필요
    dataset = BertDataset(texts, tokenizer, device=device)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 손실 함수 초기화
    criterion = LengthAwareCrossEntropyLoss()
    
    # 학습 실행
    train_bert(model, train_dataloader, optimizer, criterion, device)

if __name__ == '__main__':
    main() 