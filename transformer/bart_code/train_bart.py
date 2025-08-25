import torch
import torch.nn as nn
import torch.optim as optim
from bert_code.tokenizer import BertTokenizer
from bart_code.bart_model import BartModel

def train_bart(model, train_data, tokenizer, num_epochs=10, batch_size=32, learning_rate=1e-4, device='cuda'):
    model = model.to(device)
    model.set_tokenizer(tokenizer)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.sp.piece_to_id('[PAD]'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            
            # 배치 데이터 처리
            input_texts = [data['input'] for data in batch_data]
            target_texts = [data['target'] for data in batch_data]
            
            # 입력 텍스트 마스킹
            masked_inputs = []
            original_inputs = []
            for text in input_texts:
                masking_info = model.generate_masked_input(text)
                masked_inputs.append(masking_info['masked_token_ids'])
                original_inputs.append(masking_info['original_token_ids'])
            
            # 배치 내에서 가장 긴 시퀀스의 길이에 맞춰 패딩 추가
            max_len_input = min(max(len(ids) for ids in masked_inputs), tokenizer.max_len)
            max_len_target = min(max(len(ids) for ids in original_inputs), tokenizer.max_len)
            
            # 패딩 추가
            padded_input_ids = []
            padded_target_ids = []
            
            for ids in masked_inputs:
                if len(ids) > max_len_input:
                    ids = ids[:max_len_input]
                else:
                    ids = ids + [tokenizer.sp.piece_to_id('[PAD]')] * (max_len_input - len(ids))
                padded_input_ids.append(ids)
            
            for ids in original_inputs:
                if len(ids) > max_len_target:
                    ids = ids[:max_len_target]
                else:
                    ids = ids + [tokenizer.sp.piece_to_id('[PAD]')] * (max_len_target - len(ids))
                padded_target_ids.append(ids)
            
            # 텐서 변환
            input_ids = torch.tensor(padded_input_ids).to(device)
            target_ids = torch.tensor(padded_target_ids).to(device)
            
            # 디코더 입력 준비 (시프트된 타겟)
            decoder_input_ids = torch.roll(target_ids, shifts=1, dims=1)
            decoder_input_ids[:, 0] = tokenizer.sp.piece_to_id('[CLS]')
            
            # 어텐션 마스크 생성
            attention_mask = (input_ids != tokenizer.sp.piece_to_id('[PAD]')).float()
            decoder_mask = (decoder_input_ids != tokenizer.sp.piece_to_id('[PAD]')).float()
            
            # 순전파
            optimizer.zero_grad()
            encoder_output, decoder_output = model(input_ids, attention_mask=attention_mask, 
                                                 decoder_input_ids=decoder_input_ids, 
                                                 decoder_mask=decoder_mask)
            
            # 손실 계산
            loss = criterion(decoder_output.view(-1, decoder_output.size(-1)), target_ids.view(-1))
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 에포크 결과 출력
        avg_loss = total_loss / (len(train_data) / batch_size)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

def generate_text(model, input_text, tokenizer, max_length=100, device='cuda'):
    """BART 모델을 사용하여 텍스트를 생성합니다."""
    model.eval()
    model = model.to(device)
    
    # 입력 텍스트 마스킹
    masking_info = model.generate_masked_input(input_text)
    input_ids = torch.tensor(masking_info['masked_token_ids']).unsqueeze(0).to(device)
    
    # 어텐션 마스크 생성
    attention_mask = (input_ids != tokenizer.sp.piece_to_id('[PAD]')).float()
    
    # 디코더 입력 초기화
    decoder_input_ids = torch.tensor([[tokenizer.sp.piece_to_id('[CLS]')]]).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # 디코더 마스크 생성
            decoder_mask = (decoder_input_ids != tokenizer.sp.piece_to_id('[PAD]')).float()
            
            # 인코더-디코더 순전파
            encoder_output, decoder_output = model(input_ids, attention_mask=attention_mask, 
                                                 decoder_input_ids=decoder_input_ids, 
                                                 decoder_mask=decoder_mask)
            
            # 다음 토큰 예측
            next_token_logits = decoder_output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # 디코더 입력 업데이트
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            
            # [SEP] 토큰이 생성되면 중단
            if next_token.item() == tokenizer.sp.piece_to_id('[SEP]'):
                break
    
    # 생성된 텍스트 디코딩
    generated_ids = decoder_input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text 