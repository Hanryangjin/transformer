import torch
import numpy as np
from bert_code.tokenizer import BertTokenizer

class MaskInfilling:
    def __init__(self, tokenizer=None, lambda_param=3.0, mask_token='[MASK]'):
        """
        Mask Infilling을 처리하는 클래스
        
        Args:
            tokenizer (BertTokenizer, optional): 토크나이저 객체. None인 경우 나중에 set_tokenizer로 설정해야 함.
            lambda_param (float): 포아송 분포의 파라미터
            mask_token (str): 마스크 토큰 문자열
        """
        self.tokenizer = tokenizer
        self.lambda_param = lambda_param  # 포아송 분포의 파라미터
        self.mask_token = mask_token
        self.mask_token_id = None  # 토크나이저가 설정될 때 초기화됨
        
        if tokenizer is not None:
            self._initialize_mask_token_id()
    
    def _initialize_mask_token_id(self):
        """마스크 토큰 ID를 초기화하는 함수"""
        if self.tokenizer is None:
            raise ValueError("토크나이저가 설정되지 않았습니다. set_tokenizer()를 호출하세요.")
        self.mask_token_id = self.tokenizer.sp.piece_to_id(self.mask_token)
    
    def set_tokenizer(self, tokenizer):
        """
        토크나이저를 설정하는 함수
        
        Args:
            tokenizer (BertTokenizer): 설정할 토크나이저 객체
        """
        if not isinstance(tokenizer, BertTokenizer):
            raise TypeError("tokenizer는 BertTokenizer 인스턴스여야 합니다.")
        self.tokenizer = tokenizer
        self._initialize_mask_token_id()
    
    def _sample_span_length(self):
        return np.random.poisson(self.lambda_param)
    
    def _create_masking_plan(self, text_length):
        masking_plan = []
        current_pos = 0
        
        while current_pos < text_length:
            # 포아송 분포에서 구간 길이 샘플링
            span_length = self._sample_span_length()
            
            # 구간의 끝 위치 계산
            end_pos = min(current_pos + span_length, text_length)
            
            # 구간 정보 저장 (시작 위치, 끝 위치, 마스킹 여부)
            masking_plan.append({
                'start': current_pos,
                'end': end_pos,
                'mask': span_length > 0  # 길이가 0이면 마스킹하지 않음
            })
            
            current_pos = end_pos
        
        return masking_plan
    
    def mask_text(self, text):
        if self.tokenizer is None:
            raise ValueError("토크나이저가 설정되지 않았습니다. set_tokenizer()를 호출하세요.")
            
        # 텍스트를 토큰으로 분리
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # 마스킹 계획 생성
        masking_plan = self._create_masking_plan(len(token_ids))
        
        # 마스킹 적용
        masked_token_ids = []
        for span in masking_plan:
            if span['mask']:
                # 구간의 길이가 0인 경우 단일 [MASK] 토큰 삽입
                if span['end'] - span['start'] == 0:
                    masked_token_ids.append(self.mask_token_id)
                else:
                    # 구간의 모든 토큰을 [MASK]로 대체
                    masked_token_ids.extend([self.mask_token_id] * (span['end'] - span['start']))
            else:
                # 마스킹하지 않는 구간은 원래 토큰 유지
                masked_token_ids.extend(token_ids[span['start']:span['end']])
        
        # 마스킹된 텍스트 생성
        masked_tokens = self.tokenizer.convert_ids_to_tokens(masked_token_ids)
        masked_text = ''.join(masked_tokens)
        
        return masked_text, masked_token_ids, token_ids
    
    def get_masking_info(self, text):
        masked_text, masked_token_ids, original_token_ids = self.mask_text(text)
        
        # 마스킹된 위치 정보 추출
        masked_positions = []
        for i, token_id in enumerate(masked_token_ids):
            if token_id == self.mask_token_id:
                masked_positions.append(i)
        
        return {
            'masked_text': masked_text,
            'masked_token_ids': masked_token_ids,
            'original_token_ids': original_token_ids,
            'masked_positions': masked_positions
        } 