import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

class EditBasedDecoder:
    def __init__(self, tokenizer, beam_size: int = 5):
        """
        Args:
            tokenizer: 토크나이저
            beam_size: 빔 서치 크기
        """
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        
        # Operation ID 매핑
        self.operation_map = {
            0: 'KEEP',
            1: 'DELETE',
            2: 'INSERT',
            3: 'REPLACE'
        }
    
    def apply_operation(self, tokens: List[int], operation: int, new_token: int = None) -> List[int]:
        """
        단일 operation을 적용하는 함수
        """
        if operation == 0:  # KEEP
            return tokens
        elif operation == 1:  # DELETE
            return []
        elif operation == 2:  # INSERT
            return tokens + [new_token]
        elif operation == 3:  # REPLACE
            return [new_token]
        return tokens
    
    def beam_search(self, model_outputs: Dict[str, torch.Tensor], input_tokens: List[int]) -> List[int]:
        """
        빔 서치를 사용한 디코딩
        """
        operation_logits = model_outputs['operation_logits']
        token_logits = model_outputs['token_logits']
        
        # 초기 빔 설정
        beams = [([], torch.tensor(0.0, device=operation_logits.device))]  # (토큰 시퀀스, 로그 확률)
        
        for t in range(len(input_tokens)):
            new_beams = []
            
            for beam_tokens, beam_score in beams:
                # 현재 위치의 operation과 토큰 확률
                op_probs = F.softmax(operation_logits[0, t], dim=-1)
                token_probs = F.softmax(token_logits[0, t], dim=-1)
                
                # 상위 k개의 operation과 토큰 조합
                for op_id in range(4):
                    op_prob = op_probs[op_id]
                    
                    if op_id in [2, 3]:  # INSERT나 REPLACE의 경우
                        # 상위 k개의 토큰 선택
                        top_tokens = torch.topk(token_probs, self.beam_size)
                        for token_id, token_prob in zip(top_tokens.indices, top_tokens.values):
                            new_tokens = self.apply_operation(beam_tokens, op_id, token_id.item())
                            new_score = beam_score + torch.log(op_prob * token_prob)
                            new_beams.append((new_tokens, new_score))
                    else:  # KEEP이나 DELETE의 경우
                        new_tokens = self.apply_operation(beam_tokens, op_id)
                        new_score = beam_score + torch.log(op_prob)
                        new_beams.append((new_tokens, new_score))
            
            # 상위 k개의 빔 선택
            beams = sorted(new_beams, key=lambda x: x[1].item(), reverse=True)[:self.beam_size]
        
        # 최종 결과 선택
        best_tokens, _ = max(beams, key=lambda x: x[1].item())
        return best_tokens
    
    def decode(self, model_outputs: Dict[str, torch.Tensor], input_tokens: List[int]) -> str:
        """
        모델 출력을 디코딩하여 최종 문장 생성
        """
        # 빔 서치로 토큰 시퀀스 생성
        output_tokens = self.beam_search(model_outputs, input_tokens)
        
        # 토큰을 텍스트로 변환
        output_text = self.tokenizer.decode(output_tokens)
        
        return output_text 