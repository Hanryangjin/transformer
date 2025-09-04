import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCELossForGEC(nn.Module):
    def __init__(self, pad_token_id: int, edit_weight: float = 1.5, normal_weight: float = 1.0, min_edit_ratio: float = 0.1):
        """
        Args:
            pad_token_id: 패딩 토큰 ID (loss에서 무시)
            edit_weight: 수정된 토큰에 부여할 가중치
            normal_weight: 수정되지 않은 토큰에 부여할 가중치
            min_edit_ratio: 최소 수정 비율 (이 비율보다 적은 수정은 페널티)
        """
        super().__init__()
        self.pad_token_id = pad_token_id
        self.edit_weight = edit_weight
        self.normal_weight = normal_weight
        self.min_edit_ratio = min_edit_ratio

    def forward(self, logits, target_ids, input_ids):
        """
        Args:
            logits: [batch_size, seq_len, vocab_size]
            target_ids: [batch_size, seq_len] - 정답 토큰
            input_ids: [batch_size, seq_len] - 입력 토큰 (수정 여부 확인용)
        """
        vocab_size = logits.size(-1)
        batch_size = logits.size(0)

        # 예측값 -> [batch * seq_len, vocab_size], 정답 -> [batch * seq_len]
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target_ids.view(-1)
        input_flat = input_ids.view(-1)

        # 패딩 위치 무시
        valid_mask = (target_flat != self.pad_token_id)
        logits_flat = logits_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        input_flat = input_flat[valid_mask]

        # 수정 여부 계산
        is_edited = (target_flat != input_flat).float()
        
        # 배치별 수정 비율 계산
        seq_len = logits.size(1)
        edit_ratios = is_edited.view(batch_size, -1).mean(dim=1)
        
        # 수정 비율이 너무 낮은 경우 페널티
        ratio_penalty = torch.where(edit_ratios < self.min_edit_ratio,
                                  (self.min_edit_ratio - edit_ratios) * 2.0,
                                  torch.zeros_like(edit_ratios))
        
        # 가중치 계산: 수정 여부와 수정 비율을 모두 고려
        weights = self.normal_weight + is_edited * (self.edit_weight - self.normal_weight)
        
        # 배치별 페널티 적용
        ratio_penalty = ratio_penalty.repeat_interleave(seq_len)[valid_mask]
        weights = weights * (1.0 + ratio_penalty)

        # CrossEntropyLoss 계산
        loss = F.cross_entropy(logits_flat, target_flat, reduction='none')
        weighted_loss = (loss * weights).mean()

        return weighted_loss