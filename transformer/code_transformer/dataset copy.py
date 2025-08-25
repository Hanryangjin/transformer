import json
from torch.utils.data import Dataset
import torch
import numpy as np
import os

class SpellingDataset:
    def __init__(self, train_data_path, val_data_path, tokenizer, max_length):
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # 학습 데이터와 검증 데이터 로드
        self.train_data = self._load_data(train_data_path)
        self.val_data = self._load_data(val_data_path)
        
        # 학습 데이터와 검증 데이터 전처리
        self.train_input_ids, self.train_output_ids = self._preprocess_data(self.train_data)
        self.val_input_ids, self.val_output_ids = self._preprocess_data(self.val_data)
    
    def _load_data(self, data_path):
        """JSON 파일에서 데이터를 로드하는 함수"""
        print(f"데이터 파일 로드 중: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 데이터가 'data' 키를 가지고 있는지 확인
        if 'data' not in raw_data:
            raise ValueError(f"데이터 파일 {data_path}에 'data' 키가 없습니다.")
        
        data = []
        for item in raw_data['data']:
            # err_sentence와 cor_sentence가 있는지 확인
            if 'err_sentence' not in item or 'cor_sentence' not in item:
                continue
            
            # 입력과 출력 텍스트 추출
            input_text = item['err_sentence']
            output_text = item['cor_sentence']
            
            # 토큰화
            input_ids = self.tokenizer.encode(input_text)
            output_ids = self.tokenizer.encode(output_text)
            
            # 최대 길이 제한
            if len(input_ids) > self.max_length or len(output_ids) > self.max_length:
                continue
            
            data.append({
                'input_ids': input_ids,
                'output_ids': output_ids
            })
        
        print(f"로드된 데이터 수: {len(data)}")
        if len(data) == 0:
            raise ValueError(f"유효한 데이터가 없습니다. 파일 경로: {data_path}")
        
        return data
    
    def _preprocess_data(self, data):
        input_ids = []
        output_ids = []
        
        for item in data:
            input_ids.append(item['input_ids'])
            output_ids.append(item['output_ids'])
        
        return input_ids, output_ids
    
    @property
    def train_dataset(self):
        return TrainDataset(self.train_input_ids, self.train_output_ids, self.max_length)
    
    @property
    def val_dataset(self):
        return ValidationDataset(self.val_input_ids, self.val_output_ids, self.max_length)

class TrainDataset(Dataset):
    def __init__(self, input_ids, output_ids, max_length):
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.max_length = max_length
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        if idx >= len(self.input_ids):
            raise IndexError(f"인덱스 {idx}가 학습 데이터셋 범위를 벗어났습니다.")
        
        input_ids = self.input_ids[idx]
        output_ids = self.output_ids[idx]
        
        # 패딩 처리
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))
            
        if len(output_ids) > self.max_length:
            output_ids = output_ids[:self.max_length]
        else:
            output_ids = output_ids + [0] * (self.max_length - len(output_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long)
        }

class ValidationDataset(Dataset):
    def __init__(self, input_ids, output_ids, max_length):
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.max_length = max_length
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        if idx >= len(self.input_ids):
            raise IndexError(f"인덱스 {idx}가 검증 데이터셋 범위를 벗어났습니다.")
        
        input_ids = self.input_ids[idx]
        output_ids = self.output_ids[idx]
        
        # 패딩 처리
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))
            
        if len(output_ids) > self.max_length:
            output_ids = output_ids[:self.max_length]
        else:
            output_ids = output_ids + [0] * (self.max_length - len(output_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long)
        }