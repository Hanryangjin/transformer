import os
import sys
import torch
import json
from tqdm import tqdm
import sentencepiece as spm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Google Drive 경로 설정
drive_path = "/content/gdrive/Othercomputers/학교 컴/대학원 과목/[Coding]"

# 트랜스포머 모듈 경로 추가
transformer_path = os.path.join(drive_path, "transformer")
sys.path.append(transformer_path)

# 트랜스포머 모듈 임포트
from code_transformer.attention import MultiHeadAttention
from code_transformer.encoder import Encoder, EncoderBlock
from code_transformer.decoder import Decoder, DecoderBlock
from code_transformer.model import Transformer

# PyTorch 임포트
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SpellingDataset(Dataset):
    def __init__(self, train_data_path, val_data_path, max_length=512, vocab_size=16000):
        self.data = []
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # 학습 데이터 로드
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_raw_data = json.load(f)
            
        # 검증 데이터 로드
        with open(val_data_path, 'r', encoding='utf-8') as f:
            val_raw_data = json.load(f)
            
        # 데이터 전처리
        for item in train_raw_data['data']:
            annotation = item['annotation']
            if 'err_sentence' in annotation and 'cor_sentence' in annotation:
                self.data.append({
                    'input': annotation['err_sentence'],
                    'output': annotation['cor_sentence'],
                    'errors': annotation['errors']
                })
        
        # SentencePiece 모델 학습 및 토큰화
        self.train_sentencepiece(train_raw_data, val_raw_data)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(os.path.join(transformer_path, 'spm_model.model'))
        
    def train_sentencepiece(self, train_data, val_data):
        # 학습 데이터 준비
        train_texts = []
        for item in train_data['data']:
            annotation = item['annotation']
            if 'err_sentence' in annotation and 'cor_sentence' in annotation:
                train_texts.append(annotation['err_sentence'])
                train_texts.append(annotation['cor_sentence'])
        
        # 검증 데이터 준비
        val_texts = []
        for item in val_data['data']:
            annotation = item['annotation']
            if 'err_sentence' in annotation and 'cor_sentence' in annotation:
                val_texts.append(annotation['err_sentence'])
                val_texts.append(annotation['cor_sentence'])
        
        # 모든 텍스트를 하나의 파일에 저장
        train_data_path = os.path.join(transformer_path, 'train_data.txt')
        with open(train_data_path, 'w', encoding='utf-8') as f:
            for text in train_texts:
                f.write(text + '\n')
            for text in val_texts:
                f.write(text + '\n')
        
        # SentencePiece 모델 학습
        spm.SentencePieceTrainer.train(
            input=train_data_path,
            model_prefix=os.path.join(transformer_path, 'spm_model'),
            vocab_size=self.vocab_size,
            character_coverage=1.0,
            model_type='unigram',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            normalization_rule_name='nmt_nfkc',
            num_threads=4,
            shuffle_input_sentence=True,
            input_sentence_size=10000000,
            split_by_unicode_script=True,
            split_by_whitespace=True,
            byte_fallback=True,
            allow_whitespace_only_pieces=True,
            hard_vocab_limit=False,
            min_piece_length=1,
            max_piece_length=16,
            train_extremely_large_corpus=False
        )
    
    def tokenize_text(self, text):
        tokens = self.tokenizer.encode_as_ids(text)
        if len(tokens) > self.max_length - 2:  # <s>와 </s> 토큰을 위한 공간 확보
            tokens = tokens[:self.max_length-2]
        
        # <s>와 </s> 토큰 추가
        tokens = [2] + tokens + [3]  # 2: <s>, 3: </s>
        
        # 패딩 추가
        padding_length = self.max_length - len(tokens)
        if padding_length > 0:
            tokens += [0] * padding_length  # 0: <pad>
        
        return tokens
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 입력과 출력 텍스트 토큰화
        input_ids = self.tokenize_text(item['input'])
        output_ids = self.tokenize_text(item['output'])
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long)
        }

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

def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            output_ids = batch['output_ids'].to(device)
            
            # 마스크 생성
            src_mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = (output_ids != 0).unsqueeze(1).unsqueeze(2)
            
            # 순전파
            outputs = model(input_ids, output_ids, src_mask, tgt_mask)
            
            # 손실 계산 (새로운 손실 함수는 배치 형태 그대로 사용)
            loss = criterion(outputs, output_ids)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        # 체크포인트 저장
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader),
            'tokenizer': train_loader.dataset.tokenizer
        }
        torch.save(checkpoint, os.path.join(transformer_path, 'checkpoints', f'model_epoch_{epoch+1}.pt'))
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

def evaluate_model(model, tokenizer, validation_data, device, max_length=512):
    model.eval()
    total_correct = 0
    total_samples = 0
    error_types = {
        '띄어쓰기': 0,
        '유사 발음': 0,
        '유사 모양': 0,
        '문장부호': 0
    }
    
    with torch.no_grad():
        for sample in tqdm(validation_data):
            err_sentence = sample['annotation']['err_sentence']
            cor_sentence = sample['annotation']['cor_sentence']
            errors = sample['annotation']['errors']
            
            # 입력 텍스트 토큰화
            input_ids = tokenizer.encode_as_ids(err_sentence)
            if len(input_ids) > max_length - 2:  # <s>와 </s> 토큰을 위한 공간 확보
                input_ids = input_ids[:max_length-2]
            
            # <s>와 </s> 토큰 추가
            input_ids = [2] + input_ids + [3]  # 2: <s>, 3: </s>
            
            # 패딩 추가
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids += [0] * padding_length  # 0: <pad>
            
            # 텐서 변환
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
            
            # 마스크 생성
            src_mask = (input_tensor != 0).unsqueeze(1).unsqueeze(2)
            
            # 모델 예측
            output = model(input_tensor, input_tensor, src_mask, src_mask)
            predictions = output.argmax(dim=-1)
            
            # 예측 결과를 텍스트로 변환
            pred_ids = predictions[0].cpu().numpy().tolist()
            pred_text = tokenizer.decode(pred_ids)
            
            # 정답과 비교
            if pred_text == cor_sentence:
                total_correct += 1
                
                # 오류 유형별 통계
                for error in errors:
                    for err_type in error['err_details']:
                        if err_type in error_types:
                            error_types[err_type] += 1
            
            total_samples += 1
            
            # 디버깅을 위한 출력
            if total_samples <= 5:  # 처음 5개 샘플만 출력
                print(f"\n입력: {err_sentence}")
                print(f"예측: {pred_text}")
                print(f"정답: {cor_sentence}")
    
    accuracy = total_correct / total_samples
    
    # 오류 유형별 통계 출력
    print("\n오류 유형별 통계:")
    for err_type, count in error_types.items():
        print(f"{err_type}: {count}건")
    
    return accuracy

def load_validation_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['data']

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        raise FileNotFoundError("체크포인트 파일을 찾을 수 없습니다.")
    
    # 파일 이름에서 에포크 번호 추출
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

def main():
    # 하이퍼파라미터 설정
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 10
    vocab_size = 16000
    max_length = 512
    
    # 디바이스 설정 (Colab에서는 GPU 사용 가능)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 중인 디바이스: {device}")
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = os.path.join(transformer_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 데이터셋 및 데이터로더 생성
    dataset = SpellingDataset(
        os.path.join(transformer_path, 'TrainData/맞춤법오류_SNS.json'),
        os.path.join(transformer_path, 'ValidationData/맞춤법오류_SNS.json'),
        max_length,
        vocab_size
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 초기화
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_length=max_length,
        dropout=0.1
    ).to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = LengthAwareCrossEntropyLoss(pad_id=SpellingDataset.PAD_ID, length_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 모델 학습
    train_model(model, train_loader, optimizer, criterion, device, num_epochs)
    
    # 모델 평가
    validation_data = load_validation_data(os.path.join(transformer_path, 'ValidationData/맞춤법오류_SNS.json'))
    accuracy = evaluate_model(model, dataset.tokenizer, validation_data, device)
    print(f"\nSNS 데이터에 대한 모델 정확도: {accuracy:.4f}")

if __name__ == "__main__":
    main() 