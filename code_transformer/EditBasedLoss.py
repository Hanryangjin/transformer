import torch
import torch.nn as nn
import torch.nn.functional as F

class EditBasedLoss(nn.Module):
    def __init__(self, pad_token_id: int, operation_weight: float = 2.0, token_weight: float = 1.0):
        """
        Args:
            pad_token_id: 패딩 토큰 ID
            operation_weight: operation 예측에 대한 가중치
            token_weight: 토큰 생성에 대한 가중치
        """
        super().__init__()
        self.pad_token_id = pad_token_id
        self.operation_weight = operation_weight
        self.token_weight = token_weight
        
        # Operation ID 매핑
        self.operation_map = {
            'KEEP': 0,
            'DELETE': 1,
            'INSERT': 2,
            'REPLACE': 3
        }
    
    def forward(self, model_outputs, target_operations, target_tokens, input_tokens):
        """
        Args:
            model_outputs: 모델 출력 (operation_logits, token_logits)
            target_operations: 목표 operation [batch_size, seq_len]
            target_tokens: 목표 토큰 [batch_size, seq_len]
            input_tokens: 입력 토큰 [batch_size, seq_len]
        """
        operation_logits = model_outputs['operation_logits']
        token_logits = model_outputs['token_logits']
        
        # 패딩 마스크 생성
        valid_mask = (target_tokens != self.pad_token_id)
        
        # Operation 손실 계산
        operation_loss = F.cross_entropy(
            operation_logits.view(-1, operation_logits.size(-1)),
            target_operations.view(-1),
            reduction='none'
        )
        operation_loss = (operation_loss * valid_mask.view(-1)).mean()
        
        # 토큰 손실 계산
        token_loss = F.cross_entropy(
            token_logits.view(-1, token_logits.size(-1)),
            target_tokens.view(-1),
            reduction='none'
        )
        token_loss = (token_loss * valid_mask.view(-1)).mean()
        
        # 가중치를 적용한 최종 손실
        total_loss = self.operation_weight * operation_loss + self.token_weight * token_loss
        
        return {
            'total_loss': total_loss,
            'operation_loss': operation_loss,
            'token_loss': token_loss
        } 