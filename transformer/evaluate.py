import json
import torch
import os
import sys
from tqdm import tqdm
import sentencepiece as spm

# 현재 디렉토리를 시스템 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, 'code')
sys.path.append(code_dir)

# 모듈 임포트 - 이름 충돌 방지
import code_transformer.attention as attention_module
import code_transformer.encoder as encoder_module
import code_transformer.decoder as decoder_module
import code_transformer.model as transformer_model

# PyTorch 임포트
import torch.nn as nn

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_path = "D:/김태호/대학원 과목/[Coding]"
    
    # 가장 최근 체크포인트 로드
    checkpoint_path = get_latest_checkpoint(f"{main_path}/transformer/checkpoints")
    print(f"체크포인트 로드: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 토크나이저 로드
    tokenizer = checkpoint['tokenizer']
    
    # Transformer 모델 초기화 - 모듈 이름 충돌 방지
    transformer = transformer_model.Transformer(
        src_vocab_size=32000,  # SentencePiece 어휘 크기
        tgt_vocab_size=32000,
        d_model=256,
        num_heads=4,
        num_layers=6,
        d_ff=1024,
        max_seq_length=256,
        dropout=0.1
    )
    
    transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer = transformer.to(device)
    
    # 검증 데이터 로드
    validation_data = load_validation_data(f"{main_path}/transformer/ValidationData/맞춤법오류_SNS.json")
    
    # 모델 평가
    accuracy = evaluate_model(transformer, tokenizer, validation_data, device, 256)
    print(f"\nSNS 데이터에 대한 모델 정확도: {accuracy:.4f}")

if __name__ == "__main__":
    main() 