import torch
import sentencepiece as spm
import os

class BertTokenizer:
    def __init__(self, model_path=None, max_len=512):
        self.max_len = max_len
        self.sp = None
        
        if model_path and os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
        else:
            raise ValueError("SentencePiece 모델 파일이 필요합니다.")
    
    def tokenize(self, text):
        return self.sp.encode_as_pieces(text)
    
    def convert_tokens_to_ids(self, tokens):
        return self.sp.piece_to_id(tokens)
    
    def convert_ids_to_tokens(self, ids):
        return self.sp.id_to_piece(ids)
    
    def encode(self, text, add_special_tokens=True):
        ids = self.sp.encode_as_ids(text)
        
        if add_special_tokens:
            ids = [self.sp.piece_to_id('[CLS]')] + ids + [self.sp.piece_to_id('[SEP]')]
        
        # 패딩 처리
        if len(ids) < self.max_len:
            ids = ids + [self.sp.piece_to_id('[PAD]')] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        
        return torch.tensor(ids)
    
    def decode(self, ids):
        tokens = self.convert_ids_to_tokens(ids)
        return ''.join([token for token in tokens if token not in ['[PAD]', '[CLS]', '[SEP]']])
    
    def batch_encode(self, texts, add_special_tokens=True):
        """배치 내에서 가장 긴 시퀀스의 길이에 맞춰 패딩을 추가합니다."""
        # 모든 텍스트를 토큰화
        all_ids = []
        for text in texts:
            ids = self.sp.encode_as_ids(text)
            if add_special_tokens:
                ids = [self.sp.piece_to_id('[CLS]')] + ids + [self.sp.piece_to_id('[SEP]')]
            all_ids.append(ids)
        
        # 배치 내에서 가장 긴 시퀀스의 길이 찾기
        max_len_in_batch = min(max(len(ids) for ids in all_ids), self.max_len)
        
        # 패딩 추가
        padded_ids = []
        for ids in all_ids:
            if len(ids) > max_len_in_batch:
                ids = ids[:max_len_in_batch]
            else:
                ids = ids + [self.sp.piece_to_id('[PAD]')] * (max_len_in_batch - len(ids))
            padded_ids.append(ids)
        
        return torch.tensor(padded_ids) 